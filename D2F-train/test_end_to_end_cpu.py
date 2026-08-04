#!/usr/bin/env python3
"""
End-to-End On-Policy Distillation Training Test
Tests the complete training pipeline with a small test dataset
on CPU using a simulated model.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
from pathlib import Path

# Import project modules
from utils.on_policy_rollout import student_blockwise_rollout, teacher_rollout, on_policy_distillation_step
from utils.loss import compute_on_policy_loss


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # Handle attention mask for different dimensions
        num_heads = self.self_attn.num_heads
        
        if attn_mask is not None and attn_mask.dim() == 2:
            # [L, L] -> [batch_size * num_heads, L, L]
            L = attn_mask.shape[0]
            batch_size = x.shape[0]
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
        elif attn_mask is not None and attn_mask.dim() == 3:
            # Already [batch_size * num_heads, L, L]
            pass
        elif attn_mask is not None and attn_mask.dim() == 4:
            # [batch_size, 1, L, L] -> [batch_size * num_heads, L, L]
            batch_size, _, L, _ = attn_mask.shape
            attn_mask = attn_mask.squeeze(1)  # [batch_size, L, L]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [batch_size, num_heads, L, L]
            attn_mask = attn_mask.reshape(batch_size * num_heads, L, L)  # [batch_size * num_heads, L, L]
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward
        ff_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing"""
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, 
                 dim_feedforward=512, dropout=0.1, max_seq_length=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # For adapter simulation
        self._adapter_disabled = False
        
    def disable_adapter(self):
        """Context manager to disable adapter (for teacher mode)"""
        self._adapter_disabled = True
        return self
        
    def enable_adapter(self):
        self._adapter_disabled = False
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self._adapter_disabled = False
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Embedding
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)
        
        # Output projection
        logits = self.output_proj(x)
        
        return type('Output', (), {'logits': logits})()


class SimpleMathDataset(Dataset):
    """Simple math dataset for testing"""
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['qwen7b_answer']
        
        # Simple tokenization (character-based for testing)
        # Use simple encoding where each character is a token
        question_ids = [ord(c) % 10000 for c in question]
        answer_ids = [ord(c) % 10000 for c in answer]
        
        # Truncate to max_length
        max_q_len = self.max_length // 2
        max_a_len = self.max_length - max_q_len - 1
        
        if len(question_ids) > max_q_len:
            question_ids = question_ids[:max_q_len]
        if len(answer_ids) > max_a_len:
            answer_ids = answer_ids[:max_a_len]
        
        # Combine: question + [SEP] + answer
        input_ids = question_ids + [100] + answer_ids
        question_length = len(question_ids) + 1  # +1 for SEP token
        
        # Pad to max_length
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [0] * padding_len
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        question_length = torch.tensor(question_length, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'question_length': question_length,
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    question_length = torch.stack([item['question_length'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'question_length': question_length,
    }


def create_test_data():
    """Create test data"""
    return [
        {
            'question': 'What is 2 + 3?',
            'qwen7b_answer': '5'
        },
        {
            'question': 'A farmer has 10 apples. He gives 4 to his friend. How many apples does he have left?',
            'qwen7b_answer': '6'
        },
        {
            'question': 'If a train travels at 60 km/h for 2 hours, how far does it go?',
            'qwen7b_answer': '120 km'
        },
        {
            'question': 'What is the square root of 144?',
            'qwen7b_answer': '12'
        },
        {
            'question': 'A rectangle has length 8 and width 5. What is its area?',
            'qwen7b_answer': '40'
        },
    ]


def create_simple_attention_mask(input_ids, question_length, block_size, device=None):
    """Create a simple attention mask for block-wise attention"""
    batch_size, seq_len = input_ids.shape
    
    # Create mask: [batch_size, 1, seq_len, seq_len]
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)
    
    for b in range(batch_size):
        q_len = question_length[b].item()
        
        # Question part: causal attention
        for i in range(q_len):
            for j in range(i + 1):
                attention_mask[b, 0, i, j] = 1.0
        
        # Answer part: block-wise attention
        for block_start in range(q_len, seq_len, block_size):
            block_end = min(block_start + block_size, seq_len)
            
            # Within block: bidirectional attention
            for i in range(block_start, block_end):
                for j in range(q_len, block_end):
                    attention_mask[b, 0, i, j] = 1.0
            
            # Cross blocks: causal attention
            for i in range(block_start, block_end):
                for j in range(q_len, min(i + 1, block_start)):
                    attention_mask[b, 0, i, j] = 1.0
    
    return attention_mask


def run_training(config):
    """Run on-policy distillation training"""
    print("\n" + "=" * 60)
    print("D2F On-Policy Distillation Training")
    print("=" * 60)
    
    # Extract config
    experiment_name = config.get('experiment_name', 'on_policy_test')
    distill_mode = config.get('distillation', {}).get('mode', 'on_policy')
    student_decode_steps = config.get('distillation', {}).get('student_decode_steps', 1)
    teacher_rollout_steps = config.get('distillation', {}).get('teacher_rollout_steps', 1)
    
    max_seq_length = config.get('training', {}).get('max_seq_length', 2048)
    question_length_val = config.get('training', {}).get('question_length', 256)
    batch_size = config.get('training', {}).get('per_device_train_batch_size', 1)
    learning_rate = float(config.get('training', {}).get('learning_rate', 1e-4))
    max_steps = config.get('training', {}).get('max_steps', 5)
    
    block_size = config.get('D2F', {}).get('block_size', 32768)
    enable_shift = config.get('D2F', {}).get('enable_shift', True)
    
    print(f"\nExperiment: {experiment_name}")
    print(f"Distillation mode: {distill_mode}")
    print(f"Student decode steps: {student_decode_steps}")
    print(f"Teacher rollout steps: {teacher_rollout_steps}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Block size: {block_size}")
    print(f"Max training steps: {max_steps}")
    
    # Set device
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Create models
    vocab_size = 10000
    d_model = 256
    nhead = 4
    num_layers = 2
    
    print("\nCreating student model...")
    student_model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    ).to(device)
    
    print("Creating teacher model (same architecture for testing)...")
    teacher_model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    ).to(device)
    
    # Copy weights from student to teacher (simulating pretrained teacher)
    teacher_model.load_state_dict(student_model.state_dict())
    
    # Freeze teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Create optimizer for student
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_data = create_test_data()
    dataset = SimpleMathDataset(test_data, tokenizer=None, max_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting On-Policy Distillation Training")
    print("=" * 60)
    
    mask_id = 9999  # Mask token ID for denoising
    eos_id = 101  # EOS token ID
    
    student_model.train()
    training_losses = []
    
    global_step = 0
    
    for epoch in range(1):  # Single epoch for testing
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= max_steps:
                break
            
            input_ids = batch['input_ids'].to(device)
            question_length = batch['question_length'].to(device)
            
            print(f"\nStep {global_step + 1}/{max_steps}:")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Question lengths: {question_length.tolist()}")
            
            # Create masked input for denoising
            batch_size_val, seq_len = input_ids.shape
            masked_input_ids = input_ids.clone()
            
            # Mask some tokens in the answer part
            for b in range(batch_size_val):
                q_len = question_length[b].item()
                for pos in range(q_len, seq_len):
                    if masked_input_ids[b, pos] != 0:
                        if torch.rand(1).item() < 0.3:  # 30% masking ratio
                            masked_input_ids[b, pos] = mask_id
            
            print(f"  Creating attention mask...")
            attention_mask = create_simple_attention_mask(
                masked_input_ids, question_length, block_size=min(block_size, 256), device=device
            )
            
            print(f"  Running on-policy distillation step...")
            
            # Run on-policy distillation step
            rollout_results = on_policy_distillation_step(
                input_ids=masked_input_ids,
                student_model=student_model,
                teacher_model=teacher_model,
                question_length=question_length,
                block_size=min(block_size, 256),
                student_decode_steps=student_decode_steps,
                teacher_rollout_steps=teacher_rollout_steps,
                eos_id=eos_id,
                mask_id=mask_id,
                temperature=1.0,
                top_p=0.95,
                device=device,
                vocab_size=vocab_size
            )
            
            student_decoded = rollout_results['student_decoded']
            teacher_decoded = rollout_results['teacher_decoded']
            decoded_positions = rollout_results['decoded_positions']
            
            print(f"  Student decoded shape: {student_decoded.shape}")
            print(f"  Teacher decoded shape: {teacher_decoded.shape}")
            print(f"  Decoded positions: {decoded_positions.sum().item()} / {decoded_positions.numel()}")
            
            # Compute loss
            print(f"  Computing on-policy loss...")
            
            # Student forward pass
            logits_student = student_model(student_decoded, attention_mask=attention_mask).logits
            
            # Student teacher forward pass (for comparison)
            with torch.no_grad():
                with teacher_model.disable_adapter():
                    logits_teacher = teacher_model(student_decoded, attention_mask=attention_mask).logits
            
            # Distillation loss: MSE between student and teacher logits at decoded positions
            # Only compute loss at decoded positions
            distill_loss = F.mse_loss(
                logits_student[decoded_positions],
                logits_teacher[decoded_positions].detach()
            )
            
            # Cross-entropy loss with original tokens
            ce_loss = F.cross_entropy(
                logits_student[decoded_positions],
                input_ids[decoded_positions]
            )
            
            # Total loss
            total_loss = distill_loss + ce_loss
            training_losses.append(total_loss.item())
            
            print(f"  Distillation loss: {distill_loss.item():.4f}")
            print(f"  Cross-entropy loss: {ce_loss.item():.4f}")
            print(f"  Total loss: {total_loss.item():.4f}")
            
            # Backward pass
            print(f"  Running backward pass...")
            total_loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"  ✓ Step completed")
            
            global_step += 1
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    if len(training_losses) > 0:
        print(f"\nTraining loss progression:")
        for i, loss in enumerate(training_losses):
            print(f"  Step {i + 1}: {loss:.4f}")
        
        print(f"\nInitial loss: {training_losses[0]:.4f}")
        print(f"Final loss: {training_losses[-1]:.4f}")
        
        if len(training_losses) > 1 and training_losses[-1] < training_losses[0]:
            print(f"✓ Loss decreased by {training_losses[0] - training_losses[-1]:.4f}")
            print("✓ Training is working correctly!")
        else:
            print("⚠ Loss did not decrease (may need more training or hyperparameter tuning)")
    
    return training_losses


def main():
    parser = argparse.ArgumentParser(description='Run on-policy distillation training test')
    parser.add_argument('--config', type=str, default='config/on_policy_cpu_test.yaml',
                       help='Path to configuration file')
    parser.add_argument('--max-steps', type=int, default=5,
                       help='Maximum number of training steps')
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {config_path}")
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        config = {}
    
    # Override max_steps if specified
    config.setdefault('training', {})['max_steps'] = args.max_steps
    
    # Run training
    training_losses = run_training(config)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✓ On-Policy Distillation Training Test Complete!")
    print("=" * 60)
    print("\nTo run on actual GPU with real data:")
    print("  python train.py --config config/llada_on_policy.yaml")
    print("  python train.py --config config/dream_on_policy.yaml")


if __name__ == '__main__':
    main()