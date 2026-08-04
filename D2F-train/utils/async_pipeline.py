"""
Async rollout pipeline for on-policy distillation.

Uses a dedicated GPU (rollout_device) for the student blockwise rollout
(inference-only, no gradients), overlapping it with the loss computation
+ backward pass on the training GPU (train_device).

Architecture:
    GPU A (train):   [loss_t + backward_t] [loss_t+1 + backward_t+1] ...
    GPU B (rollout):    [rollout_t+1      ] [rollout_t+2           ] ...

Only LoRA adapter weights are synced between GPUs (33.5M params = 67MB),
which takes ~3ms over PCIe.

Usage:
    pipeline = AsyncRolloutPipeline(
        train_model=denoiser,
        rollout_device=torch.device('cuda:1'),
        config=config,
        tokenizer=tokenizer,
    )
    # In training loop (prime + prefetch + drain):
    pipeline.prime_submit(batch_0)                      # start first rollout
    pending = batch_0
    for batch in dataloader[1:]:
        result = pipeline.submit_and_get(batch, dev)    # overlap: get prev, start next
        train_step(pending, result)                     # trains on prev while next rolls out
        pipeline.sync_lora_weights()
        pending = batch
    result = pipeline.get_last_result(dev)              # drain final in-flight rollout
    train_step(pending, result)
    pipeline.sync_lora_weights()
"""

import torch
import threading
import queue
import copy
from typing import Optional, Dict, Any
from utils.on_policy_rollout import student_blockwise_rollout
from utils.util import build_custom_float_attention_mask, shift_logits


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Peel off DeepSpeed / DDP / Peft wrappers to get the base model."""
    while hasattr(model, 'module'):
        model = model.module
    return model


def _get_lora_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA parameter names and their data from a model."""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.clone()
    return state


class AsyncRolloutPipeline:
    """Async rollout on a dedicated GPU, pipelined with training on another.

    Args:
        train_model: The student model on the training GPU (DeepSpeed-wrapped).
        rollout_device: Device for the rollout model (e.g. cuda:1).
        block_size: Block size for block-causal attention.
        mask_id: Mask token ID.
        eos_id: EOS token ID.
        is_llada: Whether the model is LLaDA (controls attention kwarg).
        shift: Whether to shift logits (Dream only).
        student_decode_steps: Student decode steps per block.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
    """

    def __init__(
        self,
        train_model: torch.nn.Module,
        rollout_device: torch.device,
        block_size: int = 16,
        mask_id: int = 126336,
        eos_id: int = 126348,
        is_llada: bool = True,
        shift: bool = False,
        student_decode_steps: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.rollout_device = rollout_device
        self.block_size = block_size
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.is_llada = is_llada
        self.shift = shift
        self.student_decode_steps = student_decode_steps
        self.temperature = temperature
        self.top_p = top_p

        # Get the unwrapped base model from the training side
        train_base = _unwrap(train_model)
        self._train_param_names = set(
            n for n, p in train_base.named_parameters() if p.requires_grad
        )

        # Create a lightweight rollout model: copy base weights, freeze all,
        # keep only LoRA params (will be synced from train_model).
        self.rollout_model = copy.deepcopy(train_base).to(rollout_device)
        for p in self.rollout_model.parameters():
            p.requires_grad = False
        self.rollout_model.eval()

        # Threading infrastructure
        self._input_queue: queue.Queue = queue.Queue(maxsize=2)
        self._output_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_flag = threading.Event()
        self._worker = threading.Thread(target=self._rollout_worker, daemon=True)
        self._worker.start()

        # Track train model reference for weight sync
        self._train_model = train_model

    def _rollout_worker(self):
        """Worker thread: continuously processes batches from the input queue."""
        while not self._stop_flag.is_set():
            try:
                batch = self._input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if batch is None:
                break

            input_ids = batch['data'].to(self.rollout_device)
            question_length = batch['question_length'].to(self.rollout_device)

            with torch.no_grad():
                student_decoded, decoded_positions = student_blockwise_rollout(
                    input_ids=input_ids,
                    student_model=self.rollout_model,
                    question_length=question_length,
                    block_size=self.block_size,
                    num_decode_steps=self.student_decode_steps,
                    mask_id=self.mask_id,
                    eos_id=self.eos_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    device=self.rollout_device,
                    is_llada=self.is_llada,
                    shift=self.shift,
                )

            # Transfer results to CPU (avoid keeping rollout GPU memory busy)
            result = {
                'student_decoded': student_decoded.cpu(),
                'decoded_positions': decoded_positions.cpu(),
                'question_length': question_length.cpu(),
            }
            self._output_queue.put(result)

    def _move_to(self, result: Dict[str, torch.Tensor],
                 train_device: torch.device) -> Dict[str, torch.Tensor]:
        """Move a rollout result dict onto the training device."""
        return {
            'student_decoded': result['student_decoded'].to(train_device),
            'decoded_positions': result['decoded_positions'].to(train_device),
            'question_length': result['question_length'].to(train_device),
        }

    def prime_submit(self, batch: Dict[str, torch.Tensor]):
        """Submit the first batch without fetching a result.

        Starts the rollout pipeline so that the first ``submit_and_get`` call
        has a result waiting.  Must be called once before the training loop
        (and again after ``get_last_result`` if the pipeline is reused).
        """
        self._input_queue.put(batch)

    def submit_and_get(self, batch: Dict[str, torch.Tensor],
                       train_device: torch.device) -> Dict[str, torch.Tensor]:
        """Submit ``batch`` for rollout and return the *previous* batch's result.

        This is the overlap primitive: the caller submits batch *t+1* here and
        immediately receives batch *t*'s rollout result.  While the caller
        trains on batch *t*, batch *t+1*'s rollout runs concurrently on the
        rollout GPU.

        Blocks until the previous batch's rollout finishes.  Requires a prior
        ``prime_submit`` so a result is available.

        Note (weight staleness): batch *t+1*'s rollout is enqueued *before*
        batch *t*'s optimizer step, so it runs with LoRA weights from after
        step *t-1* (one step stale).  With LoRA (lr=1e-5) this staleness is
        negligible and is the standard cost of pipelining.
        """
        self._input_queue.put(batch)
        result = self._output_queue.get()
        return self._move_to(result, train_device)

    def get_last_result(self, train_device: torch.device) -> Dict[str, torch.Tensor]:
        """Drain the final in-flight rollout result after the loop ends."""
        result = self._output_queue.get()
        return self._move_to(result, train_device)


    def sync_lora_weights(self):
        """Sync LoRA weights from the training model to the rollout model.

        Only trainable parameters (LoRA adapters, ~33.5M = 67MB) are
        transferred, which takes ~3ms over PCIe.
        """
        train_base = _unwrap(self._train_model)
        with torch.no_grad():
            for name, param in self.rollout_model.named_parameters():
                if name in self._train_param_names:
                    train_param = dict(train_base.named_parameters())[name]
                    param.data.copy_(
                        train_param.data.to(self.rollout_device)
                    )

    def stop(self):
        """Stop the worker thread and release rollout-GPU resources cleanly.

        The rollout model lives on ``rollout_device`` (a different GPU). If its
        CUDA tensors are left to be garbage-collected during interpreter
        shutdown — after the CUDA context on that device has torn down — the
        C++ runtime aborts with "terminate called without an active exception"
        (SIGABRT).  We therefore join the worker first (so no thread is mid-op
        on the rollout GPU), then explicitly free the model + cache while the
        context is still valid.
        """
        if self._stop_flag.is_set():
            return
        self._stop_flag.set()
        # Wake the worker if it is blocked on _input_queue.get()
        self._input_queue.put(None)
        self._worker.join(timeout=60.0)
        # Explicitly release the rollout model on the rollout GPU. The
        # synchronize() ensures all pending cuda ops on the rollout device
        # finish before we free the tensors; without it the C++ runtime can
        # abort at interpreter shutdown ("terminate called without an active
        # exception" / SIGABRT) when the daemon thread's CUDA context is torn
        # down mid-op.
        try:
            import gc
            with torch.cuda.device(self.rollout_device):
                torch.cuda.synchronize()
                self.rollout_model = None
                torch.cuda.empty_cache()
            gc.collect()
            with torch.cuda.device(self.rollout_device):
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def has_pending(self) -> bool:
        """Check if there are results ready in the output queue."""
        return not self._output_queue.empty()
