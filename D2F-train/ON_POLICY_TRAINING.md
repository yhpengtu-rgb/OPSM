# On-Policy Distillation Training for D2F

This document describes how to use the on-policy distillation training mode in D2F.

## Overview

The on-policy distillation training mode differs from the original off-policy mode in the following ways:

### Off-Policy (Original)
- Teacher and Student both receive the same noisy batch
- Noise is generated through a fixed masking strategy
- No actual decoding happens during training

### On-Policy (New)
- Student model performs actual decoding within each block using block-wise causal attention
- Teacher model performs further rollout based on student's decoded sequence using full bidirectional attention
- Distillation loss is computed on the on-policy generated data

## Key Differences

| Aspect | Off-Policy | On-Policy |
|--------|------------|-----------|
| Data Generation | Random masking | Student decoding + Teacher rollout |
| Student Behavior | Receives noisy input | Generates tokens autoregressively |
| Teacher Behavior | Receives noisy input | Receives student's decoded sequence |
| Loss Computation | On randomly masked positions | On student-decoded positions |

## Configuration

To use on-policy training, add the following parameters to your config file:

```yaml
train:
  # Enable on-policy distillation
  distillation_mode: 'on-policy'

  # On-policy specific parameters
  student_decode_steps: 2  # Number of steps student decodes in each block (n < block_size)
  teacher_rollout_steps: 4  # Number of steps teacher rolls out (m)
  temperature: 0.8  # Sampling temperature for on-policy rollout
  top_p: 0.95  # Top-p sampling for on-policy rollout
```

### Parameters Explained

1. **distillation_mode**: Set to `'on-policy'` to enable on-policy training. Default is `'off-policy'`.

2. **student_decode_steps** (n): Number of tokens the student model decodes in each block. Must be less than `block_size`. This controls how many steps the student model performs before passing to the teacher.

3. **teacher_rollout_steps** (m): Number of additional steps the teacher model performs. The teacher uses the student's decoded sequence as input and can decode more tokens using its full bidirectional attention.

4. **temperature**: Sampling temperature for token generation during rollout. Lower values make the model more deterministic.

5. **top_p**: Top-p (nucleus) sampling parameter. Controls the diversity of generated tokens.

## Example Configs

We provide example configurations for both LLaDA and Dream models:

- `config/llada_on_policy.yaml` - On-policy training for LLaDA-8B-Instruct
- `config/dream_on_policy.yaml` - On-policy training for Dream-v0-Base-7B

## Training

To train with on-policy distillation:

```bash
# For LLaDA
python train.py --config config/llada_on_policy.yaml

# For Dream
python train.py --config config/dream_on_policy.yaml
```

## Algorithm Details

### Student Block-wise Rollout

1. For each block in the sequence:
   - Initialize with mask tokens (except prompt)
   - Decode `n` tokens using student model with block-wise causal attention
   - Student can only attend to:
     - All tokens in the prompt
     - All tokens in previous blocks (fully)
     - Tokens in current block that have been decoded so far (within block, bidirectional)

2. This creates a partial decoding where the student has "explored" the first n tokens of each block.

### Teacher Rollout

1. Receive student's partially decoded sequence
2. For each block:
   - Identify positions still masked
   - Decode `m` additional tokens using teacher model with full bidirectional attention
   - Teacher can attend to all tokens in the sequence (no causal restriction)

3. This provides better predictions because the teacher has access to more context.

### Loss Computation

The loss is computed as:

1. **Student-decoded positions**: Knowledge distillation loss (student's logits vs teacher's soft labels)
2. **Remaining masked positions**: Cross-entropy loss with ground truth labels

This combination ensures:
- Student learns to match teacher's predictions on its own decoded tokens
- Student still learns to predict the remaining tokens correctly

## Advantages of On-Policy Training

1. **Better Teacher-Student Alignment**: Student learns on data it actually generates, not random masks.

2. **Curriculum Learning Effect**: Student starts with simpler predictions (few tokens decoded) and gradually learns harder predictions.

3. **More Realistic Training**: Matches the actual inference scenario where the model generates tokens sequentially.

4. **Potential for Better Generalization**: Training on self-generated data can improve the model's ability to handle its own prediction errors.

## Hyperparameter Tuning Tips

1. **student_decode_steps (n)**:
   - Start with small values (1-2)
   - Larger values mean more student exploration but slower training
   - Should be less than `block_size`

2. **teacher_rollout_steps (m)**:
   - Should be `block_size - n` or larger
   - Larger values give teacher more time to correct student's mistakes
   - Too large may make training slow

3. **temperature**:
   - Start with 0.8-1.0
   - Lower temperature (0.5-0.7) for more deterministic rollout
   - Higher temperature (1.0-1.2) for more diverse training data

4. **top_p**:
   - Standard values: 0.9-0.95
   - Lower values (0.8) for more focused sampling
   - Higher values (0.98) for more diverse sampling

## Monitoring Training

When using on-policy training, you'll see additional metrics logged:

- `student_decoded_length`: Average number of tokens decoded by student per batch
- `teacher_rollout_length`: Average number of remaining masked tokens per batch

These metrics help you understand:
- How many tokens the student is actually decoding
- How much work the teacher is doing

## Implementation Files

The on-policy training implementation consists of:

1. `utils/on_policy_rollout.py`: Core rollout functions
   - `student_blockwise_rollout()`: Student decoding logic
   - `teacher_rollout()`: Teacher decoding logic
   - `on_policy_distillation_step()`: Combined rollout step

2. `utils/loss.py`: Loss computation
   - `compute_on_policy_loss()`: On-policy distillation loss
   - Modified `compute_loss_by_config()`: Dispatches to on-policy or off-policy based on config

3. Config files in `config/`:
   - `llada_on_policy.yaml`
   - `dream_on_policy.yaml`

## Troubleshooting

### Issue: Training is very slow
- Reduce `student_decode_steps` and `teacher_rollout_steps`
- Increase `gradient_accumulation_steps`
- Use mixed precision training (already enabled in configs)

### Issue: Loss doesn't decrease
- Check that `distillation_mode` is set correctly in config
- Try lower temperature (0.7-0.8)
- Increase learning rate slightly (e.g., from 1e-5 to 2e-5)

### Issue: CUDA out of memory
- Reduce batch size
- Reduce sequence length
- Reduce block size

## Comparison with Off-Policy

To compare on-policy vs off-policy training:

1. Run both training modes with same hyperparameters (except distillation-related ones)
2. Compare:
   - Final loss values
   - Evaluation metrics (GSM8K, HumanEval, etc.)
   - Inference speed (should be similar)
   - Training time (on-policy may be slightly slower)

## Future Improvements

Potential enhancements for on-policy training:

1. **Adaptive n and m**: Dynamically adjust decode steps based on loss or confidence
2. **Importance Sampling**: Weight different tokens differently based on teacher's confidence
3. **Multi-step Rollout**: Perform multiple rollouts per training step
4. **Parallel Rollout**: Parallelize student and teacher rollouts across GPUs

## References

- Original D2F paper: "Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing"
- Knowledge Distillation: Hinton, G., et al. "Distilling the Knowledge in a Neural Network"
- On-Policy RL: Schulman, J., et al. "Proximal Policy Optimization Algorithms"