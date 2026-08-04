# DMD 式 On-Policy 蒸馏实现计划

## Context

当前 on-policy 实现的 loss 流为：
1. Student rollout (no_grad) → 采样 tokens y1'
2. Student fresh forward(y1') → y2_s (with grad)
3. Teacher forward(y1') → y2_t (no grad, via disable_adapter)
4. Loss = KL(y2_s, y2_t) on decoded positions + CE on remaining

**问题**：梯度来自第二次前向 y2_s（看到的是**完整解码序列**），不是 rollout 本身的 logits y1。这更像 off-policy 蒸馏，不是真正的 on-policy。

**目标**：改为 DMD (Distribution Matching Distillation) 式 loss：
- 梯度直接流过 **rollout logits** y1（通过 Gumbel-softmax 直通估计器让采样可微）
- Score c = log p_teacher(y1') - log p_fake(y1')（EMA fake model）
- Loss = -sg(c) · log p_student(y1')
- Remaining positions 也用 DMD loss（用 fresh forward 的 logits）

## 核心发现（探索阶段）

1. **LLaDA 模型已支持 `inputs_embeds`**（[modeling_llada.py:1226](model/modeling_llada.py#L1226)）：
   ```python
   x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings
   ```
   可以直接传 soft embeddings，无需改模型代码。

2. **PeftModel.forward 透传所有参数**：`inputs_embeds` 可通过 PeftModel → LLaDAModelLM → LLaDAModel 传递。

3. **Teacher 已实现**：`denoiser.disable_adapter()` 获取 base model logits。

4. **无现成 EMA / Gumbel-softmax 代码**，需新建。

## 实现方案

### 1. 新文件 `utils/ema_lora.py` — EMA LoRA 权重管理器

只 EMA LoRA 参数（33.5M = 67MB），共享 base model（16GB），通过权重交换实现 fake forward：

```python
class EMALoRA:
    """EMA copy of LoRA params for DMD fake model."""
    def __init__(self, student_model, decay=0.999):
        base = _unwrap(student_model)
        self._param_names = [n for n, p in base.named_parameters() if p.requires_grad]
        self.shadow = {n: base.get_parameter(n).data.clone() for n in self._param_names}
        self.decay = decay

    @torch.no_grad()
    def update(self, student_model):
        base = _unwrap(student_model)
        for n in self._param_names:
            self.shadow[n].mul_(self.decay).add_(base.get_parameter(n).data, alpha=1 - self.decay)

    @contextmanager
    def swap(self, student_model):
        """Context manager: swap in EMA weights, restore after."""
        base = _unwrap(student_model)
        backup = {n: base.get_parameter(n).data.clone() for n in self._param_names}
        for n in self._param_names:
            base.get_parameter(n).data.copy_(self.shadow[n])
        yield
        for n in self._param_names:
            base.get_parameter(n).data.copy_(backup[n])
```

**内存开销**：67MB shadow params + 67MB backup during swap = 134MB（可忽略）。

### 2. 修改 `utils/on_policy_rollout.py` — Gumbel-softmax 可微 rollout

新增 `student_blockwise_rollout_dmd` 函数：

**核心改动**：
- 去掉 `torch.no_grad()`
- 用 `F.gumbel_softmax(logits, tau=temperature, hard=True)` 替代 `_top_p_sample`
- 用 `inputs_embeds`（soft embeddings at decoded positions）替代 `input_ids`
- 收集每个 decoded position 的 rollout logits y1[pos]
- 对 remaining positions，也收集最后一次 forward 的 logits

**流程**：
```python
def student_blockwise_rollout_dmd(...):
    embed_weight = model.get_input_embeddings().weight  # [vocab, dim]

    # 初始 inputs_embeds（全 detach，无梯度）
    inputs_embeds = embed_weight(student_decoded).detach()  # [B, L, dim]

    # 存储每个 decoded position 的 rollout logits（WITH grad）
    rollout_logits = torch.zeros(B, L, vocab_size, device=device)
    soft_embeds = inputs_embeds.clone()  # 会被 decoded positions 的 soft embed 覆盖

    for block_idx in range(max_blocks):
        # 构建 subsequence 的 inputs_embeds
        soft_embeds_sub = soft_embeds[:, :max_needed, :].contiguous()
        attn_kw_sub = _attn_kwargs(is_llada, attention_mask_sub)

        # Forward WITH GRAD
        outputs = model(input_ids=None, inputs_embeds=soft_embeds_sub, **attn_kw_sub)
        logits = outputs.logits  # [B, max_needed, vocab] — WITH grad

        for step in range(max_steps_block):
            for i in active:
                pos = start_i + step
                current_logits = logits[i, pos, :]  # [vocab] — WITH grad

                # Gumbel-softmax straight-through
                soft = F.gumbel_softmax(current_logits.unsqueeze(0),
                                        tau=temperature, hard=True)  # [1, vocab]
                token_id = soft.argmax(-1).squeeze()  # hard token ID

                # Soft embedding (WITH grad through Gumbel-softmax)
                soft_embed = soft @ embed_weight  # [1, dim] — WITH grad

                # 更新序列
                student_decoded[i, pos] = token_id
                soft_embeds[i, pos] = soft_embed.squeeze(0)  # 覆盖为 soft embedding
                decoded_positions[i, pos] = True
                rollout_logits[i, pos] = current_logits  # 保存 logits（WITH grad）

    return student_decoded, decoded_positions, rollout_logits
```

**梯度链**：loss → rollout_logits[pos] → model weights → soft_embeds[pos] → Gumbel-softmax → previous block's logits → ... → first block's logits

**内存优化**：用 `torch.utils.checkpoint` 对每个 block 的 forward 做 gradient checkpointing，将 8 个 forward 的激活内存从 8× 降到 ~1×。

### 3. 修改 `utils/loss.py` — DMD 式 loss

新增 `compute_dmd_loss` 函数，替代 `compute_on_policy_loss`：

```python
def compute_dmd_loss(input_ids, denoiser, ema_lora, question_length, ...):
    # Step 1: Gumbel-softmax rollout (WITH grad)
    student_decoded, decoded_positions, rollout_logits = student_blockwise_rollout_dmd(...)

    # Step 2: Teacher forward (no grad, disable_adapter)
    with torch.no_grad():
        with denoiser.disable_adapter():
            logits_teacher = denoiser(student_decoded, **teacher_kwargs).logits

    # Step 3: Fake forward (no grad, EMA LoRA swap)
    with torch.no_grad():
        with ema_lora.swap(denoiser):
            logits_fake = denoiser(student_decoded, **teacher_kwargs).logits

    # Step 4: Fresh student forward for remaining positions (WITH grad)
    logits_student_fresh = denoiser(student_decoded, **student_kwargs).logits

    # Step 5: DMD loss on decoded positions
    # log p_student from rollout logits (Gumbel-softmax grad chain)
    log_p_student_decoded = F.log_softmax(rollout_logits[decoded_positions], dim=-1)
    sampled_tokens = student_decoded[decoded_positions]  # y1'
    log_p_student = log_p_student_decoded.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

    # Score c = log p_teacher - log p_fake (NO grad)
    with torch.no_grad():
        log_p_teacher = F.log_softmax(logits_teacher[decoded_positions], dim=-1)
        log_p_teacher = log_p_teacher.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        log_p_fake = F.log_softmax(logits_fake[decoded_positions], dim=-1)
        log_p_fake = log_p_fake.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        c = log_p_teacher - log_p_fake  # density ratio

    loss_decoded = -(c.detach() * log_p_student).mean()  # sg(c) * log p_student

    # Step 6: DMD loss on remaining positions (using fresh forward logits)
    remaining_mask = (student_decoded == mask_id) & (~decoded_positions)
    if remaining_mask.any():
        gt_tokens = input_ids[remaining_mask]
        log_p_student_rem = F.log_softmax(logits_student_fresh[remaining_mask], dim=-1)
        log_p_student_rem = log_p_student_rem.gather(-1, gt_tokens.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            log_p_teacher_rem = F.log_softmax(logits_teacher[remaining_mask], dim=-1)
            log_p_teacher_rem = log_p_teacher_rem.gather(-1, gt_tokens.unsqueeze(-1)).squeeze(-1)
            log_p_fake_rem = F.log_softmax(logits_fake[remaining_mask], dim=-1)
            log_p_fake_rem = log_p_fake_rem.gather(-1, gt_tokens.unsqueeze(-1)).squeeze(-1)
            c_rem = log_p_teacher_rem - log_p_fake_rem

        loss_remaining = -(c_rem.detach() * log_p_student_rem).mean()
        loss = loss_decoded + loss_remaining
    else:
        loss = loss_decoded

    return {'loss': loss}
```

### 4. 修改 `train.py` — 集成 EMA + DMD loss

```python
# 在 model setup 后创建 EMA
ema_lora = EMALoRA(denoiser, decay=0.999)

# 在 train_one_step 中:
# 1. 调用 compute_dmd_loss (传入 ema_lora)
# 2. backward + optimizer.step()
# 3. 更新 EMA: ema_lora.update(denoiser)

def train_one_step(batch, rollout_results):
    losses = compute_dmd_loss(
        input_ids, denoiser, ema_lora, question_length, ...
    )
    loss_tgt = losses['loss']
    accelerator.backward(loss_tgt)
    optimizer.step()
    optimizer.zero_grad()
    ema_lora.update(denoiser)  # EMA update after optimizer step
```

### 5. 配置 `config/llada_on_policy_dmd.yaml`

新增字段：
```yaml
train:
  dmd_loss: true              # 启用 DMD loss（vs 旧 KL loss）
  dmd_ema_decay: 0.999        # EMA fake model decay rate
  dmd_gumbel_tau: 1.0         # Gumbel-softmax temperature
  dmd_grad_checkpoint: true   # gradient checkpointing for rollout
```

## 关键设计决策

1. **Gumbel-softmax hard=True（直通估计器）**：前向用 hard one-hot（等价于 argmax 采样），反向用 soft 梯度。这让采样的 token embedding 可微，梯度可从后续 block 流回前序 block。

2. **EMA 只管 LoRA 参数**：base model 16GB 共享，只复制 67MB LoRA params。通过 `swap()` context manager 在 forward 前后交换权重。

3. **Remaining positions 用 fresh forward**：rollout 时 remaining positions 的 context 不完整（部分 masked），不可靠。用 fully-decoded 序列做 fresh forward 更准确。logits 来源不同但 loss 形式相同（DMD）。

4. **Gradient checkpointing**：8 个 block × 32 层的激活图无法放进 32GB V100。用 `torch.utils.checkpoint` 将每层激活在反向时重算，内存从 ~8× 降到 ~1×，计算开销 +33%。

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `utils/ema_lora.py` | **新建** | EMA LoRA 权重管理器 |
| `utils/on_policy_rollout.py` | **修改** | 新增 `student_blockwise_rollout_dmd` 函数 |
| `utils/loss.py` | **修改** | 新增 `compute_dmd_loss` 函数 + `compute_loss_by_config` 分支 |
| `train.py` | **修改** | 集成 EMA 创建/更新 + DMD loss 调用 |
| `config/llada_on_policy_dmd.yaml` | **新建** | DMD 专用配置 |

## 验证方案

1. **语法检查**：`python -m py_compile` 所有修改的文件
2. **小规模训练测试**：`num_iters=3, batch_size=1`，确认：
   - Gumbel-softmax rollout 不崩溃
   - EMA swap 正确（teacher/fake/student logits 不同）
   - Loss 数值合理（非 NaN/Inf，应与 log-probability 量级一致）
   - 梯度正确流动（LoRA 参数 grad 非 None）
3. **内存检查**：监控 GPU 显存，确认 gradient checkpointing 生效
4. **Loss 曲线**：跑 30 步，确认 loss 下降趋势
