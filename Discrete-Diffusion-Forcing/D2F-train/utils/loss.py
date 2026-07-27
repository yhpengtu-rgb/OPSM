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
        config=None
):
    """
    Compute on-policy distillation loss.
    
    In on-policy distillation:
    1. Student model decodes n steps within each block using block-wise causal attention
    2. Teacher model performs m-step rollout based on student's decoded sequence (using full bidirectional attention)
    3. Compute distillation loss between student and teacher outputs
    
    Args:
        input_ids: Original input tokens [B, L]
        denoiser: Student model (will also be used as teacher by disabling adapter)
        question_length: Length of prompt for each sample [B]
        mask_id: Token ID for mask token
        block_size: Size of each block
        enable_shift: Whether to shift logits
        share_steps: Number of shared steps
        self_align: Whether to use self-alignment
        feature_align: Whether to use feature alignment
        self_step: Whether to use self-step
        eos_id: End-of-sequence token ID
        student_decode_steps: Number of steps for student to decode in each block (n)
        teacher_rollout_steps: Number of steps for teacher to rollout (m)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        config: Configuration object
    
    Returns:
        Dictionary containing loss values
    """
    B, L = input_ids.shape
    device = input_ids.device
    
    # Import on_policy_distillation_step lazily to avoid circular imports
    from utils.on_policy_rollout import on_policy_distillation_step
    
    # Step 1: Perform on-policy rollout (student + teacher)
    rollout_results = on_policy_distillation_step(
        input_ids=input_ids,
        student_model=denoiser,
        teacher_model=denoiser,  # Teacher is the same model but with full attention
        question_length=question_length,
        block_size=block_size,
        student_decode_steps=student_decode_steps,
        teacher_rollout_steps=teacher_rollout_steps,
        mask_id=mask_id,
        eos_id=eos_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    
    student_decoded = rollout_results['student_decoded']
    teacher_decoded = rollout_results['teacher_decoded']
    student_log_probs = rollout_results['student_log_probs']
    teacher_log_probs = rollout_results['teacher_log_probs']
    decoded_positions = rollout_results['decoded_positions']
    
    # Step 2: Compute forward pass for student on its own decoded sequence
    # (to get gradients for training)
    attention_mask_student = build_custom_float_attention_mask(
        student_decoded, question_length, block_size, device=device
    )
    attention_mask_student = attention_mask_student.to(torch.float16)
    
    # Get student predictions (with gradients)
    logits_student = denoiser(student_decoded, attention_mask=attention_mask_student).logits
    if enable_shift:
        logits_student = shift_logits(logits_student)
    
    # Step 3: Compute forward pass for teacher on the student-decoded sequence
    # Teacher uses full bidirectional attention
    L = student_decoded.shape[1]
    attention_mask_teacher = torch.zeros(
        [L, L],
        dtype=torch.float32, device=device
    )
    
    with torch.no_grad():
        with denoiser.disable_adapter():
            logits_teacher = denoiser(student_decoded, attention_mask=attention_mask_teacher).logits
            if enable_shift:
                logits_teacher = shift_logits(logits_teacher)
            teacher_probs = F.softmax(logits_teacher, dim=-1)
    
    # Step 4: Compute distillation loss
    # Only compute loss on positions that were decoded by student
    token_loss = F.cross_entropy(
        logits_student[decoded_positions],
        teacher_probs[decoded_positions],
        reduction='none'
    )
    
    # Also compute loss on remaining masked positions (teacher's rollout targets)
    remaining_mask = (student_decoded == mask_id) & (~decoded_positions)
    if remaining_mask.any():
        # Use original input_ids as targets for remaining positions
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
        'teacher_rollout_length': remaining_mask.sum().float() / B,
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
        config
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    distillation_mode = config.train.get('distillation_mode', 'off-policy') if hasattr(config, 'train') else 'off-policy'
    
    # Check if on-policy distillation is requested
    if distillation_mode == 'on-policy':
        student_decode_steps = config.train.get('student_decode_steps', 1)
        teacher_rollout_steps = config.train.get('teacher_rollout_steps', 1)
        temperature = config.train.get('temperature', 1.0)
        top_p = config.train.get('top_p', 0.95)
        
        if training_mode == 'llada':
            return compute_on_policy_loss(
                input_ids, denoiser, question_length, mask_id, block_size,
                enable_shift, share_steps, self_align, feature_align, self_step, eos_id,
                student_decode_steps=student_decode_steps,
                teacher_rollout_steps=teacher_rollout_steps,
                temperature=temperature,
                top_p=top_p,
                config=config
            )
        elif training_mode == 'dream':
            return compute_on_policy_loss(
                input_ids, denoiser, question_length, mask_id, block_size,
                enable_shift, share_steps, self_align, feature_align, self_step, eos_id,
                student_decode_steps=student_decode_steps,
                teacher_rollout_steps=teacher_rollout_steps,
                temperature=temperature,
                top_p=top_p,
                config=config
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