import torch
from utils.util import forward_process_length, shift_logits, forward_process, build_custom_float_attention_mask
import torch.nn.functional as F

def compute_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    logits=denoiser(noisy_batch,attention_mask=attention_mask).logits
    logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                # denoiser.eval()
                ref_logits=denoiser(noisy_batch,attention_mask=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
                # denoiser.train()
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
def compute_normal_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    logits=denoiser(noisy_batch).logits
    logits=shift_logits(logits)
    token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
import torch
def compute_llada_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    mask_id=126336
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    # print(noisy_batch)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    # print(type(denoiser),noisy_batch.shape,attention_mask.shape)
    logits=denoiser(noisy_batch,attention_bias=attention_mask).logits
    # logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                ref_logits=denoiser(noisy_batch,attention_bias=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                # ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 


def compute_on_policy_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        student_decode_steps=1,
        teacher_rollout_steps=1,
        temperature=1.0,
        top_p=0.95,
        config=None,
        rollout_results=None,
        lengths=None
):
    """Compute on-policy distillation loss.

    In on-policy distillation:
    1. Student model decodes n steps within each block using block-wise causal
       attention (batched across blocks — only ``num_decode_steps`` forward
       passes total).
    2. A single teacher forward pass (``disable_adapter``) provides the
       distillation target on the student-decoded sequence.
    3. Distillation loss on decoded positions + supervised loss on remaining
       masked positions.

    Args:
        input_ids: Original input tokens [B, L].
        denoiser: Student model (also used as teacher via disable_adapter).
        question_length: Prompt length per sample [B].
        mask_id: Mask token ID.
        block_size: Block size.
        enable_shift: Whether to shift logits.
        share_steps: Number of shared steps.
        self_align: Whether to use self-alignment.
        feature_align: Whether to use feature alignment.
        self_step: Whether to use self-step.
        eos_id: EOS token ID.
        student_decode_steps: Student decode steps per block.
        teacher_rollout_steps: (unused — teacher rollout is disabled).
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        config: Configuration object.
        rollout_results: Externally-provided rollout (async pipeline). When not
            None, the internal rollout call is skipped.
        lengths: Real-token length per sample [B] (question + answer, excl.
            pure padding). When provided, positions >= length are excluded
            from the loss so the reported loss is consistent across batch
            sizes and padding lengths (dynamic padding). None = no exclusion.

    Returns:
        Dictionary containing loss values.
    """
    B, L = input_ids.shape
    device = input_ids.device

    # Import on_policy_distillation_step lazily to avoid circular imports
    from utils.on_policy_rollout import on_policy_distillation_step

    # Model-family dispatch: LLaDA's forward takes the 4D block mask as
    # ``attention_bias`` (added to SDPA scores) and uses masked-LM-style
    # prediction (no logit shift). Dream's forward takes it as
    # ``attention_mask`` and predicts the next token (shift required).
    training_mode = config.get('training_mode', 'dream') if config is not None else 'dream'
    is_llada = (training_mode == 'llada')
    # LLaDA already predicts the token at each position -> never shift.
    shift = (not is_llada) and enable_shift

    # Resolve vocab size for the (unused) log-prob buffers allocated in the rollout.
    try:
        base = denoiser.get_base_model() if hasattr(denoiser, 'get_base_model') else denoiser
        rollout_vocab_size = base.config.vocab_size
    except Exception:
        rollout_vocab_size = 128000

    # Step 1: Perform on-policy rollout (student + teacher)
    # If rollout_results are provided externally (async pipeline), skip
    # the internal rollout call.
    if rollout_results is not None:
        student_decoded = rollout_results['student_decoded']
        decoded_positions = rollout_results['decoded_positions']
    else:
        rollout_results = on_policy_distillation_step(
            input_ids=input_ids,
            student_model=denoiser,
            teacher_model=denoiser,
            question_length=question_length,
            block_size=block_size,
            student_decode_steps=student_decode_steps,
            teacher_rollout_steps=teacher_rollout_steps,
            mask_id=mask_id,
            eos_id=eos_id,
            temperature=temperature,
            top_p=top_p,
            device=device,
            vocab_size=rollout_vocab_size,
            is_llada=is_llada,
            shift=shift,
        )
        student_decoded = rollout_results['student_decoded']
        decoded_positions = rollout_results['decoded_positions']

    # Free rollout intermediates before the gradient-bearing forward pass
    del rollout_results
    torch.cuda.empty_cache()

    # Step 2: Compute forward pass for student on its own decoded sequence
    # (to get gradients for training). Pass the 4D block mask through the
    # kwarg expected by the model family.
    attention_mask_student = build_custom_float_attention_mask(
        student_decoded, question_length, block_size, device=device
    )
    attention_mask_student = attention_mask_student.to(torch.float16)
    student_kwargs = (
        {"attention_bias": attention_mask_student} if is_llada
        else {"attention_mask": attention_mask_student}
    )

    # Get student predictions (with gradients)
    logits_student = denoiser(student_decoded, **student_kwargs).logits
    if shift:
        logits_student = shift_logits(logits_student)

    # Step 3: Compute forward pass for teacher on the student-decoded sequence
    # Teacher uses full bidirectional attention (4D all-zero bias).
    L = student_decoded.shape[1]
    attention_mask_teacher = torch.zeros(
        [1, 1, L, L], dtype=torch.float16, device=device
    )
    teacher_kwargs = (
        {"attention_bias": attention_mask_teacher} if is_llada
        else {"attention_mask": attention_mask_teacher}
    )

    with torch.no_grad():
        # ``disable_adapter`` lives on the PeftModel; under DeepSpeed the engine
        # forwards the attribute to ``.module``. Match the off-policy path.
        with denoiser.disable_adapter():
            logits_teacher = denoiser(student_decoded, **teacher_kwargs).logits
            if shift:
                logits_teacher = shift_logits(logits_teacher)
            teacher_probs = F.softmax(logits_teacher, dim=-1)

    # Step 4: Compute distillation loss
    # Build a valid-position mask that excludes pure padding (positions >=
    # ``length``) when ``lengths`` is provided. With dynamic padding the pad
    # region shrinks/grows per batch; including it would make the mean loss
    # depend on how much padding a batch happens to carry. Excluding pad
    # positions yields a loss that is comparable across batch sizes and
    # padding strategies (only real answer tokens + the closing EOS count).
    if lengths is not None:
        token_positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        valid_mask = token_positions < lengths.to(device).unsqueeze(1)  # [B, L]
        decoded_positions = decoded_positions & valid_mask
    else:
        valid_mask = None

    # Only compute loss on positions that were decoded by student
    if decoded_positions.any():
        token_loss = F.cross_entropy(
            logits_student[decoded_positions],
            teacher_probs[decoded_positions],
            reduction='none'
        )
    else:
        token_loss = torch.tensor([], device=device)

    # Also compute supervised loss on remaining masked positions (positions
    # the student did not decode — target is the ground-truth token).
    remaining_mask = (student_decoded == mask_id) & (~decoded_positions)
    if valid_mask is not None:
        remaining_mask = remaining_mask & valid_mask
    if remaining_mask.any():
        token_loss_remaining = F.cross_entropy(
            logits_student[remaining_mask],
            input_ids[remaining_mask],
            reduction='none'
        )
        token_loss = torch.cat([token_loss, token_loss_remaining])

    # Compute mean loss
    loss = token_loss.mean()

    losses = {
        'loss': loss,
        'student_decoded_length': decoded_positions.sum().float() / B,
        'remaining_mask_length': remaining_mask.sum().float() / B,
    }

    return losses


def compute_dmd_loss(
    input_ids,
    denoiser,
    ema_lora,
    question_length,
    mask_id,
    block_size,
    enable_shift,
    share_steps,
    self_align,
    feature_align,
    self_step,
    eos_id,
    student_decode_steps=1,
    teacher_rollout_steps=1,
    temperature=1.0,
    top_p=0.95,
    config=None,
    lengths=None,
):
    """Three-model Concrete transition-DMD on successor rollout states.

    Student rollout transitions are detached. After every unmask transition
    x_t -> x_{t+1}, this loss evaluates the current block's remaining masks
    in x_{t+1}: the student receives the Concrete Score Matching surrogate,
    while the EMA fake and frozen teacher define its detached DMD vector
    field. A small ground-truth masked-token term remains an explicit,
    configurable auxiliary objective.
    """
    from utils.on_policy_rollout import student_blockwise_rollout_dmd

    B, L = input_ids.shape
    device = input_ids.device

    training_mode = config.get('training_mode', 'dream') if config is not None else 'dream'
    is_llada = (training_mode == 'llada')
    shift = (not is_llada) and enable_shift

    # Step 1: Detached rollout. transitions[-1] is the sole final state after
    # every block has completed its configured student_decode_steps updates.

    student_decoded, decoded_positions, transitions = student_blockwise_rollout_dmd(
        input_ids=input_ids,
        student_model=denoiser,
        question_length=question_length,
        block_size=block_size,
        num_decode_steps=student_decode_steps,
        mask_id=mask_id,
        eos_id=eos_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        is_llada=is_llada,
        shift=shift,
    )

    # Step 2: Compute Concrete DMD once on the final rollout state.
    # Teacher is full-bidirectional; fake and student use block-causal masks.
    if lengths is not None:
        token_positions = torch.arange(L, device=device).unsqueeze(0)
        valid_mask = token_positions < lengths.to(device).unsqueeze(1)
    else:
        valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    final_transition = transitions[-1]
    successor_ids = final_transition["input_ids"]
    successor_mask = final_transition["remaining_mask"] & valid_mask

    attention_mask_student = build_custom_float_attention_mask(
        successor_ids, question_length, block_size, device=device
    ).to(torch.float16)
    attention_mask_teacher = torch.zeros(
        [1, 1, L, L], dtype=torch.float16, device=device
    )
    student_kwargs = (
        {"attention_bias": attention_mask_student} if is_llada
        else {"attention_mask": attention_mask_student}
    )
    teacher_kwargs = (
        {"attention_bias": attention_mask_teacher} if is_llada
        else {"attention_mask": attention_mask_teacher}
    )

    with torch.no_grad():
        with denoiser.disable_adapter():
            teacher_logits = denoiser(successor_ids, **teacher_kwargs).logits
            if shift:
                teacher_logits = shift_logits(teacher_logits)
        with ema_lora.swap(denoiser):
            fake_logits = denoiser(successor_ids, **student_kwargs).logits
            if shift:
                fake_logits = shift_logits(fake_logits)

    student_logits_full = denoiser(successor_ids, **student_kwargs).logits
    if shift:
        student_logits_full = shift_logits(student_logits_full)

    if successor_mask.any():
        student_logits = student_logits_full[successor_mask]
        teacher_logits = teacher_logits[successor_mask].float()
        fake_logits = fake_logits[successor_mask].float()

        # Concrete score matching: student probabilities provide the measure;
        # EMA fake minus teacher defines the detached DMD score vector field.
        student_probs = F.softmax(student_logits.detach(), dim=-1, dtype=torch.float32)
        fake_mean = (student_probs * fake_logits).sum(dim=-1, keepdim=True)
        teacher_mean = (student_probs * teacher_logits).sum(dim=-1, keepdim=True)
        fake_score = fake_logits - fake_mean
        teacher_score = teacher_logits - teacher_mean
        grad_coeff = 2.0 * student_probs * (fake_score - teacher_score)
        loss_transition = (
            grad_coeff.detach() * student_logits.float()
        ).sum(dim=-1).mean()
    else:
        loss_transition = student_logits_full[0, 0, 0] * 0.0

    # Step 3: Optional ground-truth masked-token auxiliary objective on the
    # final rollout state. It is not part of the transition-DMD objective.
    aux_remaining_weight = config.train.get('aux_remaining_weight', 0.0) if config is not None else 0.0
    remaining_mask = successor_mask
    loss_remaining = torch.zeros((), device=device)
    if aux_remaining_weight and remaining_mask.any():
        # Reuse the final-state student forward already computed for CSM.
        loss_remaining = F.cross_entropy(
            student_logits_full[remaining_mask], input_ids[remaining_mask]
        )

    loss = loss_transition + aux_remaining_weight * loss_remaining
    # DMD successor masks vary across batches. Keep every trainable adapter
    # parameter in the autograd graph with a zero contribution so DeepSpeed's
    # partitioned gradient reducer sees a stable parameter set each step.
    loss = loss + sum(
        (parameter.reshape(-1)[0] * 0.0)
        for parameter in denoiser.parameters()
        if parameter.requires_grad
    )
    losses = {
        'loss': loss,
        'transition_dmd_loss': loss_transition.detach(),
        'aux_remaining_loss': loss_remaining.detach(),
        'student_decoded_length': decoded_positions.sum().float() / B,
        'remaining_mask_length': remaining_mask.sum().float() / B,
        'transition_count': torch.tensor(1, device=device),
    }

    return losses


def compute_loss_by_config(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        config,
        rollout_results=None,
        lengths=None,
        ema_lora=None
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    distillation_mode = config.train.get('distillation_mode', 'off-policy') if hasattr(config, 'train') else 'off-policy'
    dmd_loss = config.train.get('dmd_loss', False) if hasattr(config, 'train') else False

    # LLaDA's mask token id (from config.json) is 126336; the config file's
    # ``denoiser.encoder.mask_id`` is only correct for Dream. Mirror the
    # off-policy ``compute_llada_loss`` which hardcodes 126336.
    if training_mode == 'llada':
        mask_id = 126336

    # Check if on-policy distillation is requested
    if distillation_mode == 'on-policy':
        student_decode_steps = config.train.get('student_decode_steps', 1)
        teacher_rollout_steps = config.train.get('teacher_rollout_steps', 1)
        temperature = config.train.get('temperature', 1.0)
        top_p = config.train.get('top_p', 0.95)

        # DMD-style loss (Gumbel-softmax rollout + EMA fake model)
        if dmd_loss:
            if ema_lora is None:
                raise ValueError("dmd_loss requires ema_lora to be passed to compute_loss_by_config")
            return compute_dmd_loss(
                input_ids, denoiser, ema_lora, question_length, mask_id, block_size,
                enable_shift, share_steps, self_align, feature_align, self_step, eos_id,
                student_decode_steps=student_decode_steps,
                teacher_rollout_steps=teacher_rollout_steps,
                temperature=temperature,
                top_p=top_p,
                config=config,
                lengths=lengths,
            )

        # Standard on-policy KL loss
        if training_mode == 'llada':
            return compute_on_policy_loss(
                input_ids, denoiser, question_length, mask_id, block_size,
                enable_shift, share_steps, self_align, feature_align, self_step, eos_id,
                student_decode_steps=student_decode_steps,
                teacher_rollout_steps=teacher_rollout_steps,
                temperature=temperature,
                top_p=top_p,
                config=config,
                rollout_results=rollout_results,
                lengths=lengths,
            )
        elif training_mode == 'dream':
            return compute_on_policy_loss(
                input_ids, denoiser, question_length, mask_id, block_size,
                enable_shift, share_steps, self_align, feature_align, self_step, eos_id,
                student_decode_steps=student_decode_steps,
                teacher_rollout_steps=teacher_rollout_steps,
                temperature=temperature,
                top_p=top_p,
                config=config,
                rollout_results=rollout_results,
                lengths=lengths,
            )
        else:
            raise ValueError(f"Unsupported training mode: {training_mode}")
    
    # Original off-policy logic
    if training_mode == 'llada':
        return compute_llada_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'dream':
        return compute_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")


if __name__ == "__main__":
    seq_len = 10
    input_ids = torch.randint(0, 100, (2, seq_len))  # 示例输入
    block_size = 4
    prompt_length = torch.tensor([2, 4])  # 示例prompt长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_mask = build_custom_float_attention_mask(input_ids, prompt_length, block_size, device)
    print(attn_mask)