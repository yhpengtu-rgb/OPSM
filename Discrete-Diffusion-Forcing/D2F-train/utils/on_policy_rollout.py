"""
On-policy rollout functions for D2F training.
This module implements the on-policy distillation strategy where:
1. Student model performs n-step decoding within each block
2. Teacher model performs m-step rollout based on student's outputs
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from utils.util import build_custom_float_attention_mask, shift_logits


def student_blockwise_rollout(
    input_ids: torch.Tensor,
    student_model: torch.nn.Module,
    question_length: torch.Tensor,
    block_size: int,
    num_decode_steps: int,
    mask_id: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    vocab_size: int = 128000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Student model performs on-policy decoding within each block.
    
    Args:
        input_ids: Original input tokens [B, L]
        student_model: Student model with block-wise causal attention
        question_length: Length of prompt for each sample [B]
        block_size: Size of each block
        num_decode_steps: Number of steps to decode in each block (n < block_size)
        mask_id: Token ID for mask token
        eos_id: Token ID for end-of-sequence
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to use
        vocab_size: Vocabulary size (default 128000 for D2F)
    
    Returns:
        student_decoded: Decoded sequence [B, L]
        decoded_positions: Positions that were decoded by student [B, L] boolean mask
        decode_log_probs: Log probabilities of decoded tokens [B, L, vocab_size]
    """
    B, L = input_ids.shape
    if device is None:
        device = input_ids.device
    
    # Initialize with fully masked sequence (except prompt)
    student_decoded = input_ids.clone()
    token_positions = torch.arange(L, device=device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    
    # Mask all non-prompt tokens initially
    non_prompt_mask = ~prompt_mask
    student_decoded[non_prompt_mask] = mask_id
    
    # Track which positions have been decoded
    decoded_positions = torch.zeros_like(input_ids, dtype=torch.bool)
    decode_log_probs = torch.zeros(B, L, vocab_size, device=device)
    
    # Calculate number of blocks
    non_prompt_lens = L - question_length
    full_blocks = non_prompt_lens // block_size
    remainders = non_prompt_lens % block_size
    total_blocks = full_blocks + (remainders > 0).long()
    max_blocks = total_blocks.max().item()
    
    # Decode each block
    for block_idx in range(max_blocks):
        for i in range(B):
            if block_idx >= total_blocks[i].item():
                continue
                
            prompt_len = question_length[i].item()
            start = prompt_len + block_idx * block_size
            end = min(start + block_size, L)
            
            # Decode n steps within this block
            for step in range(min(num_decode_steps, end - start)):
                pos = start + step
                
                # Build attention mask for current state
                attention_mask = build_custom_float_attention_mask(
                    student_decoded, question_length, block_size, device
                )
                
                # Get model predictions
                with torch.no_grad():
                    # Use student's forward pass with block-wise causal attention
                    if hasattr(student_model, 'module'):
                        outputs = student_model.module(
                            student_decoded,
                            attention_mask=attention_mask if 'llada' not in str(type(student_model)).lower() else None,
                            attention_bias=attention_mask if 'llada' in str(type(student_model)).lower() else None
                        )
                    else:
                        outputs = student_model(
                            student_decoded,
                            attention_mask=attention_mask if 'llada' not in str(type(student_model)).lower() else None,
                            attention_bias=attention_mask if 'llada' in str(type(student_model)).lower() else None
                        )
                    
                    logits = outputs.logits
                
                # Apply shift if needed (for Dream model)
                if hasattr(student_model, 'module'):
                    model_ref = student_model.module
                else:
                    model_ref = student_model
                
                # Shift logits to align with next token prediction
                if 'dream' in str(type(model_ref)).lower():
                    logits = shift_logits(logits)
                
                # Get logits for current position
                current_logits = logits[i:i+1, pos, :]  # [1, vocab_size]
                
                # Sample token
                probs = F.softmax(current_logits / temperature, dim=-1)
                if top_p < 1.0:
                    # Apply top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                
                sampled_token = torch.multinomial(probs, 1)
                
                # Update decoded sequence
                student_decoded[i, pos] = sampled_token.squeeze()
                decoded_positions[i, pos] = True
                
                # Store log probabilities
                decode_log_probs[i, pos] = current_logits.squeeze()
                
                # Stop if EOS token is generated
                if sampled_token.item() == eos_id:
                    break
    
    return student_decoded, decoded_positions, decode_log_probs


def teacher_rollout(
    student_decoded: torch.Tensor,
    teacher_model: torch.nn.Module,
    question_length: torch.Tensor,
    block_size: int,
    num_rollout_steps: int,
    decoded_positions: torch.Tensor,
    mask_id: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    vocab_size: int = 128000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher model performs rollout based on student's decoded sequence.
    Teacher uses full bidirectional attention.
    
    Args:
        student_decoded: Sequence decoded by student [B, L]
        teacher_model: Teacher model with bidirectional attention
        question_length: Length of prompt for each sample [B]
        block_size: Size of each block
        num_rollout_steps: Number of additional steps for teacher rollout (m)
        decoded_positions: Positions decoded by student [B, L]
        mask_id: Token ID for mask token
        eos_id: Token ID for end-of-sequence
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to use
        vocab_size: Vocabulary size (default 128000 for D2F)
    
    Returns:
        teacher_decoded: Final decoded sequence [B, L]
        teacher_log_probs: Teacher's log probabilities [B, L, vocab_size]
    """
    B, L = student_decoded.shape
    if device is None:
        device = student_decoded.device
    
    teacher_decoded = student_decoded.clone()
    teacher_log_probs = torch.zeros(B, L, vocab_size, device=device)
    
    # Calculate number of blocks
    non_prompt_lens = L - question_length
    full_blocks = non_prompt_lens // block_size
    remainders = non_prompt_lens % block_size
    total_blocks = full_blocks + (remainders > 0).long()
    max_blocks = total_blocks.max().item()
    
    # Teacher performs additional rollout
    for block_idx in range(max_blocks):
        for i in range(B):
            if block_idx >= total_blocks[i].item():
                continue
            
            prompt_len = question_length[i].item()
            start = prompt_len + block_idx * block_size
            end = min(start + block_size, L)
            
            # Find positions that are still masked (not decoded by student)
            block_positions = torch.arange(start, end, device=device)
            masked_in_block = (teacher_decoded[i, start:end] == mask_id) & (block_positions < L)
            
            # Decode m additional steps or until all tokens are decoded
            num_to_decode = min(num_rollout_steps, masked_in_block.sum().item())
            
            for step in range(num_to_decode):
                # Find next masked position
                masked_positions = (teacher_decoded[i] == mask_id) & (teacher_decoded[i] != eos_id)
                if not masked_positions.any():
                    break
                
                # Get first masked position in current block
                block_masked = masked_positions[start:end]
                if not block_masked.any():
                    break
                
                pos = start + block_masked.nonzero()[0].item()
                
                # Use bidirectional attention (no causal mask)
                # Full attention mask (all positions can attend to all)
                # Use 2-D mask [L, L] which will be broadcast to all batches and heads
                attention_mask = torch.zeros(L, L, dtype=torch.float32, device=device)
                
                # Get teacher predictions
                with torch.no_grad():
                    with teacher_model.disable_adapter():  # Disable adapter to use original weights
                        if hasattr(teacher_model, 'module'):
                            outputs = teacher_model.module(
                                teacher_decoded,
                                attention_mask=attention_mask if 'llada' not in str(type(teacher_model)).lower() else None,
                                attention_bias=attention_mask if 'llada' in str(type(teacher_model)).lower() else None
                            )
                        else:
                            outputs = teacher_model(
                                teacher_decoded,
                                attention_mask=attention_mask if 'llada' not in str(type(teacher_model)).lower() else None,
                                attention_bias=attention_mask if 'llada' in str(type(teacher_model)).lower() else None
                            )
                        
                        logits = outputs.logits
                
                # Apply shift if needed
                if 'dream' in str(type(teacher_model)).lower():
                    logits = shift_logits(logits)
                
                # Get logits for current position
                current_logits = logits[i:i+1, pos, :]
                
                # Sample token
                probs = F.softmax(current_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                
                sampled_token = torch.multinomial(probs, 1)
                
                # Update decoded sequence
                teacher_decoded[i, pos] = sampled_token.squeeze()
                teacher_log_probs[i, pos] = current_logits.squeeze()
                
                # Stop if EOS token is generated
                if sampled_token.item() == eos_id:
                    break
    
    return teacher_decoded, teacher_log_probs


def on_policy_distillation_step(
    input_ids: torch.Tensor,
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    question_length: torch.Tensor,
    block_size: int,
    student_decode_steps: int,
    teacher_rollout_steps: int,
    mask_id: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    vocab_size: int = 128000,
) -> Dict[str, torch.Tensor]:
    """
    Perform one step of on-policy distillation.
    
    Args:
        input_ids: Original input tokens [B, L]
        student_model: Student model
        teacher_model: Teacher model (same as student but with full attention)
        question_length: Length of prompt [B]
        block_size: Block size
        student_decode_steps: Number of steps for student to decode (n)
        teacher_rollout_steps: Number of steps for teacher rollout (m)
        mask_id: Mask token ID
        eos_id: EOS token ID
        temperature: Sampling temperature
        top_p: Top-p sampling
        device: Device
        vocab_size: Vocabulary size (default 128000 for D2F)
    
    Returns:
        Dictionary containing:
            - student_decoded: Student's decoded sequence
            - teacher_decoded: Teacher's decoded sequence
            - student_log_probs: Student's log probabilities
            - teacher_log_probs: Teacher's log probabilities
            - decoded_positions: Positions decoded by student
    """
    # Step 1: Student performs block-wise rollout
    student_decoded, decoded_positions, student_log_probs = student_blockwise_rollout(
        input_ids=input_ids,
        student_model=student_model,
        question_length=question_length,
        block_size=block_size,
        num_decode_steps=student_decode_steps,
        mask_id=mask_id,
        eos_id=eos_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        vocab_size=vocab_size,
    )
    
    # Step 2: Teacher performs rollout based on student's outputs
    teacher_decoded, teacher_log_probs = teacher_rollout(
        student_decoded=student_decoded,
        teacher_model=teacher_model,
        question_length=question_length,
        block_size=block_size,
        num_rollout_steps=teacher_rollout_steps,
        decoded_positions=decoded_positions,
        mask_id=mask_id,
        eos_id=eos_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        vocab_size=vocab_size,
    )
    
    return {
        'student_decoded': student_decoded,
        'teacher_decoded': teacher_decoded,
        'student_log_probs': student_log_probs,
        'teacher_log_probs': teacher_log_probs,
        'decoded_positions': decoded_positions,
    }