import torch
from utils.util import forward_process_length, shift_logits, forward_process, build_custom_float_attention_mask
import torch.nn.functional as F


def _unwrap_model(model):
    while hasattr(model, 'module'):
        model = model.module
    return model

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
        lengths=lengths,
    )

    # Step 2: Compute Concrete DMD once on the final rollout state.
    # Teacher is full-bidirectional; fake and student use block-causal masks.
    if lengths is not None:
        valid_lengths = lengths.to(device=device, dtype=torch.long).clamp(0, L)
        token_positions = torch.arange(L, device=device).unsqueeze(0)
        valid_mask = token_positions < valid_lengths.unsqueeze(1)
    else:
        valid_lengths = torch.full((B,), L, dtype=torch.long, device=device)
        valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    final_transition = transitions[-1]
    successor_ids = final_transition["input_ids"]
    successor_mask = final_transition["remaining_mask"] & valid_mask

    prompt_lengths = question_length.to(device=device, dtype=torch.long).clamp(0, L)
    prompt_lengths = torch.minimum(prompt_lengths, valid_lengths)
    attention_mask_student = build_custom_float_attention_mask(
        successor_ids, prompt_lengths, block_size, device=device
    ).to(torch.float16)
    attention_mask_student = attention_mask_student.masked_fill(
        ~valid_mask[:, None, None, :], float('-inf')
    )
    attention_mask_teacher = torch.zeros(
        [B, 1, L, L], dtype=torch.float16, device=device
    ).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
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
            teacher_selected = teacher_logits[successor_mask].float()
        with ema_lora.swap(denoiser):
            fake_logits = denoiser(successor_ids, **student_kwargs).logits
            if shift:
                fake_logits = shift_logits(fake_logits)
            fake_selected = fake_logits[successor_mask].float()

    student_logits_full = denoiser(successor_ids, **student_kwargs).logits
    if shift:
        student_logits_full = shift_logits(student_logits_full)

    if successor_mask.any():
        student_logits = student_logits_full[successor_mask]
        teacher_logits = teacher_selected
        fake_logits = fake_selected

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


def compute_transition_csm_loss(
    input_ids, denoiser, ema_lora, question_length, mask_id, block_size,
    enable_shift, eos_id, student_decode_steps=1, temperature=1.0, top_p=0.95,
    config=None, lengths=None, backward_callback=None, gradient_probe_callback=None,
):
    """Run CSM on one uniformly sampled hard rollout transition."""
    from utils.on_policy_rollout import student_blockwise_rollout_dmd

    if backward_callback is None:
        raise ValueError("transition_csm requires a per-transition backward callback")
    B, L = input_ids.shape
    device = input_ids.device
    is_llada = config.get('training_mode', 'dream') == 'llada'
    shift = (not is_llada) and enable_shift
    student_decoded, decoded_positions, transitions = student_blockwise_rollout_dmd(
        input_ids=input_ids, student_model=denoiser, question_length=question_length,
        block_size=block_size, num_decode_steps=student_decode_steps, mask_id=mask_id,
        eos_id=eos_id, temperature=temperature, top_p=top_p, device=device,
        is_llada=is_llada, shift=shift, lengths=lengths, transition_csm=True,
        transition_sample_ratio=config.train.get('transition_sample_ratio', 0.01),
    )
    if lengths is not None:
        valid_lengths = lengths.to(device=device, dtype=torch.long).clamp(0, L)
    else:
        valid_lengths = torch.full((B,), L, dtype=torch.long, device=device)
    prompt_lengths = torch.minimum(
        question_length.to(device=device, dtype=torch.long).clamp(0, L), valid_lengths
    )
    valid_mask = torch.arange(L, device=device).unsqueeze(0) < valid_lengths.unsqueeze(1)
    local_transition_count = len(transitions)
    synced_transition_count = local_transition_count
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        main_transition_count = torch.tensor(
            [local_transition_count], dtype=torch.long, device=device
        )
        torch.distributed.broadcast(main_transition_count, src=0)
        synced_transition_count = main_transition_count.item()

    # Rank 0 caps real CSM transitions. Ranks with fewer local transitions pad
    # the missing backwards with zero gradients so every rank performs the same
    # number of DDP/DeepSpeed gradient-reduction collectives.
    transitions = transitions[:synced_transition_count]
    transition_count = len(transitions)
    transition_total = torch.zeros((), device=device)
    aux_remaining_total = torch.zeros((), device=device)
    remaining_mask_total = torch.zeros((), device=device)
    aux_remaining_weight = config.train.get('aux_remaining_weight', 0.0)
    adapter_model = _unwrap_model(denoiser)
    teacher_mask = torch.zeros([B, 1, L, L], dtype=torch.float16, device=device).masked_fill(
        ~valid_mask[:, None, None, :], float('-inf')
    )
    teacher_kwargs = {'attention_bias': teacher_mask} if is_llada else {'attention_mask': teacher_mask}
    def zero_term():
        return sum(
            parameter.reshape(-1)[0] * 0.0 for parameter in denoiser.parameters()
            if parameter.requires_grad
        )

    for transition in transitions:
        predecessor_ids = transition['predecessor_ids']
        successor_ids = transition['successor_ids']
        csm_mask = transition['answer_mask'] & transition['advanced_mask'][:, None]
        predecessor_mask = build_custom_float_attention_mask(
            predecessor_ids, prompt_lengths, block_size, device=device
        ).to(torch.float16).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
        successor_mask = build_custom_float_attention_mask(
            successor_ids, prompt_lengths, block_size, device=device
        ).to(torch.float16).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
        student_kwargs = {'attention_bias': predecessor_mask} if is_llada else {'attention_mask': predecessor_mask}
        fake_kwargs = {'attention_bias': successor_mask} if is_llada else {'attention_mask': successor_mask}
        with torch.no_grad():
            with adapter_model.disable_adapter():
                teacher_logits = denoiser(successor_ids, **teacher_kwargs).logits
                if shift:
                    teacher_logits = shift_logits(teacher_logits)
                teacher_selected = teacher_logits[csm_mask].float()
            with ema_lora.swap(denoiser):
                fake_logits = denoiser(successor_ids, **fake_kwargs).logits
                if shift:
                    fake_logits = shift_logits(fake_logits)
                fake_selected = fake_logits[csm_mask].float()
        student_logits_full = denoiser(predecessor_ids, **student_kwargs).logits
        if shift:
            student_logits_full = shift_logits(student_logits_full)
        if csm_mask.any():
            student_logits = student_logits_full[csm_mask]
            student_probs = F.softmax(student_logits.detach(), dim=-1, dtype=torch.float32)
            fake_score = fake_selected - (student_probs * fake_selected).sum(dim=-1, keepdim=True)
            teacher_score = teacher_selected - (student_probs * teacher_selected).sum(dim=-1, keepdim=True)
            grad_coeff = (2.0 * student_probs * (fake_score - teacher_score)).detach()
            transition_loss = (grad_coeff * student_logits.float()).sum(dim=-1).mean()
        else:
            transition_loss = student_logits_full[0, 0, 0] * 0.0
        transition_total = transition_total + transition_loss.detach() / synced_transition_count
        backward_callback(transition_loss / synced_transition_count + zero_term())

    for _ in range(synced_transition_count - transition_count):
        backward_callback(zero_term())

    csm_grads = None
    if gradient_probe_callback is not None:
        csm_grads = {
            parameter: parameter.grad.detach().clone()
            for parameter in denoiser.parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        for parameter in denoiser.parameters():
            if parameter.requires_grad:
                parameter.grad = None

    # CE anchors the endpoint of the latest sampled transition, independently
    # of the CSM gradients computed for every sampled transition above. Every
    # rank invokes its CE backward once, including ranks without a local
    # endpoint, to keep distributed gradient collectives aligned.
    final_remaining_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    loss_remaining = torch.zeros((), device=device)
    if transition_count:
        final_transition = transitions[-1]
        final_successor_ids = final_transition['successor_ids']
        final_remaining_mask = (
            (final_successor_ids == mask_id)
            & final_transition['answer_mask']
            & valid_mask
        )
        if aux_remaining_weight and final_remaining_mask.any():
            final_student_mask = build_custom_float_attention_mask(
                final_successor_ids, prompt_lengths, block_size, device=device
            ).to(torch.float16).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
            final_student_kwargs = (
                {'attention_bias': final_student_mask} if is_llada
                else {'attention_mask': final_student_mask}
            )
            final_student_logits = denoiser(final_successor_ids, **final_student_kwargs).logits
            if shift:
                final_student_logits = shift_logits(final_student_logits)
            loss_remaining = F.cross_entropy(
                final_student_logits[final_remaining_mask], input_ids[final_remaining_mask]
            )
    backward_callback(aux_remaining_weight * loss_remaining + zero_term())
    if gradient_probe_callback is not None:
        ce_grads = {
            parameter: parameter.grad.detach().clone()
            for parameter in denoiser.parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        ce_grad_sq = torch.zeros((), device=device)
        for grad in ce_grads.values():
            ce_grad_sq = ce_grad_sq + grad.float().square().sum()
        ce_norm = torch.nan_to_num(
            ce_grad_sq.sqrt() / max(abs(aux_remaining_weight), 1e-8),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        csm_norm = torch.zeros((), device=device)
        if csm_grads is not None:
            for grad in csm_grads.values():
                csm_norm = csm_norm + grad.float().square().sum()
            csm_norm = csm_norm.sqrt()
        csm_norm = torch.nan_to_num(csm_norm, nan=0.0, posinf=0.0, neginf=0.0)
        gradient_probe_callback(csm_norm, ce_norm)
        for parameter in denoiser.parameters():
            if parameter.requires_grad:
                parameter.grad = csm_grads.get(parameter, torch.zeros_like(parameter))
                if parameter in ce_grads:
                    parameter.grad.add_(ce_grads[parameter])

    aux_remaining_total = loss_remaining.detach()
    remaining_mask_total = final_remaining_mask.sum().float()
    total_loss = transition_total + aux_remaining_weight * aux_remaining_total
    return {
        'loss': total_loss,
        'transition_dmd_loss': transition_total,
        'aux_remaining_loss': aux_remaining_total,
        'student_decoded_length': decoded_positions.sum().float() / B,
        'remaining_mask_length': remaining_mask_total / B,
        'transition_count': torch.tensor(transition_count, device=device),
        'backward_done': True,
    }


def compute_macro_remask_csm_loss(
    input_ids, denoiser, ema_lora, question_length, mask_id, block_size,
    enable_shift, eos_id, student_decode_steps=1, temperature=1.0, top_p=0.95,
    config=None, lengths=None, backward_callback=None,
):
    """Concrete score matching on a low-confidence remasked full draft.

    The detached rollout returns successor ``x_s`` and generation-time token
    confidences. Its low-confidence tokens are remasked to form predecessor
    ``x_t``. Teacher and EMA-fake scores are evaluated at ``x_s`` and update
    only student logits evaluated at ``x_t`` on the remasked support.
    """
    from utils.on_policy_rollout import student_blockwise_rollout_dmd

    if backward_callback is None:
        raise ValueError("macro_remask_csm requires a backward callback")
    B, L = input_ids.shape
    device = input_ids.device
    is_llada = config.get('training_mode', 'dream') == 'llada'
    shift = (not is_llada) and enable_shift
    full_draft, decoded_positions, _, rollout_confidence = student_blockwise_rollout_dmd(
        input_ids=input_ids, student_model=denoiser, question_length=question_length,
        block_size=block_size, num_decode_steps=student_decode_steps, mask_id=mask_id,
        eos_id=eos_id, temperature=temperature, top_p=top_p, device=device,
        is_llada=is_llada, shift=shift, lengths=lengths,
        return_rollout_confidence=True,
    )
    if lengths is None:
        valid_lengths = torch.full((B,), L, device=device, dtype=torch.long)
    else:
        valid_lengths = lengths.to(device=device, dtype=torch.long).clamp(0, L)
    positions = torch.arange(L, device=device).unsqueeze(0)
    valid_mask = positions < valid_lengths.unsqueeze(1)
    prompt_lengths = torch.minimum(
        question_length.to(device=device, dtype=torch.long).clamp(0, L), valid_lengths
    )
    answer_mask = (positions >= prompt_lengths.unsqueeze(1)) & valid_mask
    correction_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    remask_ratio = config.train.get('macro_remask_ratio', 0.25)
    for batch_idx in range(B):
        candidates = torch.nonzero(answer_mask[batch_idx] & full_draft[batch_idx].ne(mask_id), as_tuple=True)[0]
        if candidates.numel() == 0:
            continue
        remask_count = min(candidates.numel(), max(1, int(candidates.numel() * remask_ratio)))
        selected = torch.topk(rollout_confidence[batch_idx, candidates], remask_count, largest=False).indices
        correction_mask[batch_idx, candidates[selected]] = True

    predecessor_ids = full_draft.masked_fill(correction_mask, mask_id)
    successor_ids = full_draft
    successor_mode = config.train.get('macro_remask_successor', 'full_draft')
    teacher_mask = torch.zeros([B, 1, L, L], dtype=torch.float16, device=device).masked_fill(
        ~valid_mask[:, None, None, :], float('-inf')
    )
    teacher_kwargs = {'attention_bias': teacher_mask} if is_llada else {'attention_mask': teacher_mask}
    if successor_mode == 'teacher_corrected':
        with torch.no_grad():
            adapter_model = _unwrap_model(denoiser)
            with adapter_model.disable_adapter():
                correction_logits = denoiser(predecessor_ids, **teacher_kwargs).logits
                if shift:
                    correction_logits = shift_logits(correction_logits)
                successor_ids = predecessor_ids.clone()
                successor_ids[correction_mask] = correction_logits[correction_mask].argmax(dim=-1)
    elif successor_mode != 'full_draft':
        raise ValueError(f"Unsupported macro_remask_successor: {successor_mode}")

    predecessor_mask = build_custom_float_attention_mask(
        predecessor_ids, prompt_lengths, block_size, device=device
    ).to(torch.float16).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
    successor_mask = build_custom_float_attention_mask(
        successor_ids, prompt_lengths, block_size, device=device
    ).to(torch.float16).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
    student_kwargs = {'attention_bias': predecessor_mask} if is_llada else {'attention_mask': predecessor_mask}
    fake_kwargs = {'attention_bias': successor_mask} if is_llada else {'attention_mask': successor_mask}
    adapter_model = _unwrap_model(denoiser)
    with torch.no_grad():
        with adapter_model.disable_adapter():
            teacher_logits = denoiser(successor_ids, **teacher_kwargs).logits
            if shift:
                teacher_logits = shift_logits(teacher_logits)
            teacher_selected = teacher_logits[correction_mask].float()
        with ema_lora.swap(denoiser):
            fake_logits = denoiser(successor_ids, **fake_kwargs).logits
            if shift:
                fake_logits = shift_logits(fake_logits)
            fake_selected = fake_logits[correction_mask].float()

    student_logits_full = denoiser(predecessor_ids, **student_kwargs).logits
    if shift:
        student_logits_full = shift_logits(student_logits_full)
    if correction_mask.any():
        student_logits = student_logits_full[correction_mask]
        student_probs = F.softmax(student_logits.detach(), dim=-1, dtype=torch.float32)
        fake_score = fake_selected - (student_probs * fake_selected).sum(dim=-1, keepdim=True)
        teacher_score = teacher_selected - (student_probs * teacher_selected).sum(dim=-1, keepdim=True)
        grad_coeff = (2.0 * student_probs * (fake_score - teacher_score)).detach()
        csm_loss = (grad_coeff * student_logits.float()).sum(dim=-1).mean()
        ce_loss = F.cross_entropy(student_logits, input_ids[correction_mask])
    else:
        csm_loss = student_logits_full[0, 0, 0] * 0.0
        ce_loss = csm_loss
    ce_weight = config.train.get('macro_remask_gt_ce_weight', 0.0)
    backward_callback(csm_loss + ce_weight * ce_loss + sum(
        parameter.reshape(-1)[0] * 0.0 for parameter in denoiser.parameters()
        if parameter.requires_grad
    ))
    return {
        'loss': (csm_loss.detach() + ce_weight * ce_loss.detach()),
        'transition_dmd_loss': csm_loss.detach(),
        'aux_remaining_loss': ce_loss.detach(),
        'student_decoded_length': decoded_positions.sum().float() / B,
        'remaining_mask_length': correction_mask.sum().float() / B,
        'transition_count': torch.tensor(1, device=device),
        'backward_done': True,
    }


def compute_final_draft_remask_loss(
    input_ids, denoiser, question_length, mask_id, block_size, enable_shift,
    eos_id, student_decode_steps=1, temperature=1.0, top_p=0.95,
    config=None, lengths=None, backward_callback=None,
):
    """Train one confidence-guided teacher correction on a final hard draft."""
    from utils.on_policy_rollout import student_blockwise_rollout_dmd

    if backward_callback is None:
        raise ValueError("final_draft_remask requires a backward callback")
    B, L = input_ids.shape
    device = input_ids.device
    is_llada = config.get('training_mode', 'dream') == 'llada'
    shift = (not is_llada) and enable_shift
    if lengths is None:
        valid_lengths = torch.full((B,), L, dtype=torch.long, device=device)
    else:
        valid_lengths = lengths.to(device=device, dtype=torch.long).clamp(0, L)
    prompt_lengths = torch.minimum(
        question_length.to(device=device, dtype=torch.long).clamp(0, L), valid_lengths
    )
    token_positions = torch.arange(L, device=device).unsqueeze(0)
    answer_mask = (
        (token_positions >= prompt_lengths.unsqueeze(1))
        & (token_positions < valid_lengths.unsqueeze(1))
    )
    valid_mask = token_positions < valid_lengths.unsqueeze(1)

    final_draft, decoded_positions, _ = student_blockwise_rollout_dmd(
        input_ids=input_ids, student_model=denoiser, question_length=question_length,
        block_size=block_size, num_decode_steps=student_decode_steps, mask_id=mask_id,
        eos_id=eos_id, temperature=temperature, top_p=top_p, device=device,
        is_llada=is_llada, shift=shift, lengths=lengths,
    )
    draft_mask = build_custom_float_attention_mask(
        final_draft, prompt_lengths, block_size, device=device
    ).to(torch.float16).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
    draft_kwargs = {'attention_bias': draft_mask} if is_llada else {'attention_mask': draft_mask}
    with torch.no_grad():
        draft_logits = denoiser(final_draft, **draft_kwargs).logits
        if shift:
            draft_logits = shift_logits(draft_logits)
        draft_probs = F.softmax(draft_logits.float(), dim=-1)
        draft_confidence = draft_probs.gather(-1, final_draft.unsqueeze(-1)).squeeze(-1)

    remask_ratio = config.train.get('final_draft_remask_ratio', 0.25)
    remask_mask = final_draft.eq(mask_id) & answer_mask
    decoded_answer_mask = answer_mask & final_draft.ne(mask_id)
    for batch_idx in range(B):
        candidates = torch.nonzero(decoded_answer_mask[batch_idx], as_tuple=True)[0]
        if candidates.numel() == 0:
            continue
        remask_count = max(1, int(candidates.numel() * remask_ratio))
        lowest_confidence = torch.topk(
            draft_confidence[batch_idx, candidates], remask_count, largest=False
        ).indices
        remask_mask[batch_idx, candidates[lowest_confidence]] = True

    corrupted_draft = final_draft.masked_fill(remask_mask, mask_id)
    correction_mask = build_custom_float_attention_mask(
        corrupted_draft, prompt_lengths, block_size, device=device
    ).to(torch.float16).masked_fill(~valid_mask[:, None, None, :], float('-inf'))
    student_kwargs = {'attention_bias': correction_mask} if is_llada else {'attention_mask': correction_mask}
    teacher_mask = torch.zeros([B, 1, L, L], dtype=torch.float16, device=device).masked_fill(
        ~valid_mask[:, None, None, :], float('-inf')
    )
    teacher_kwargs = {'attention_bias': teacher_mask} if is_llada else {'attention_mask': teacher_mask}
    with torch.no_grad():
        with denoiser.disable_adapter():
            teacher_logits = denoiser(corrupted_draft, **teacher_kwargs).logits
            if shift:
                teacher_logits = shift_logits(teacher_logits)
            teacher_probs = F.softmax(teacher_logits[remask_mask].float(), dim=-1)

    student_logits = denoiser(corrupted_draft, **student_kwargs).logits
    if shift:
        student_logits = shift_logits(student_logits)
    if remask_mask.any():
        correction_loss = F.cross_entropy(student_logits[remask_mask], teacher_probs)
    else:
        correction_loss = student_logits[0, 0, 0] * 0.0
    correction_loss = correction_loss + sum(
        parameter.reshape(-1)[0] * 0.0 for parameter in denoiser.parameters()
        if parameter.requires_grad
    )
    backward_callback(correction_loss)
    return {
        'loss': correction_loss.detach(),
        'transition_dmd_loss': correction_loss.detach(),
        'aux_remaining_loss': torch.zeros((), device=device),
        'student_decoded_length': decoded_positions.sum().float() / B,
        'remaining_mask_length': remask_mask.sum().float() / B,
        'transition_count': torch.tensor(1, device=device),
        'backward_done': True,
    }


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
        ema_lora=None,
        backward_callback=None,
        gradient_probe_callback=None,
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    distillation_mode = config.train.get('distillation_mode', 'off-policy') if hasattr(config, 'train') else 'off-policy'
    dmd_loss = config.train.get('dmd_loss', False) if hasattr(config, 'train') else False
    transition_csm = config.train.get('transition_csm', False) if hasattr(config, 'train') else False
    final_draft_remask = config.train.get('final_draft_remask', False) if hasattr(config, 'train') else False
    macro_remask_csm = config.train.get('macro_remask_csm', False) if hasattr(config, 'train') else False

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

        # Per-step hard-rollout transition CSM uses the EMA fake even when the
        # legacy final-state dmd_loss mode is disabled.
        if transition_csm:
            if ema_lora is None:
                raise ValueError("transition_csm requires ema_lora to be passed to compute_loss_by_config")
            return compute_transition_csm_loss(
                input_ids, denoiser, ema_lora, question_length, mask_id, block_size,
                enable_shift, eos_id, student_decode_steps=student_decode_steps,
                temperature=temperature, top_p=top_p, config=config, lengths=lengths,
                backward_callback=backward_callback,
                gradient_probe_callback=gradient_probe_callback,
            )

        if macro_remask_csm:
            if ema_lora is None:
                raise ValueError("macro_remask_csm requires ema_lora to be passed to compute_loss_by_config")
            return compute_macro_remask_csm_loss(
                input_ids, denoiser, ema_lora, question_length, mask_id, block_size,
                enable_shift, eos_id, student_decode_steps=student_decode_steps,
                temperature=temperature, top_p=top_p, config=config, lengths=lengths,
                backward_callback=backward_callback,
            )

        if final_draft_remask:
            return compute_final_draft_remask_loss(
                input_ids, denoiser, question_length, mask_id, block_size,
                enable_shift, eos_id, student_decode_steps=student_decode_steps,
                temperature=temperature, top_p=top_p, config=config, lengths=lengths,
                backward_callback=backward_callback,
            )

        # Legacy DMD-style final-state loss.
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