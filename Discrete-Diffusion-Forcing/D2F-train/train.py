# from peft import PeftModel, PeftConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig, get_peft_model
from utils.util import flatten_dict,shift_logits
from utils.data import get_bs17k_dataloader,get_llada_bs17k_dataloader,get_dataloader_by_config
from utils.model import get_model,get_llada,get_model_by_config
from utils.loss import compute_loss,compute_llada_loss,compute_normal_loss,compute_loss_by_config
from utils.generation import sample_tokens
# import dataloader

import os
import torch
import argparse
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_accelerator(config, global_config):
    # Select experiment path based on config
    if hasattr(global_config, 'paths') and hasattr(global_config.paths, 'experiment'):
        root_path = global_config.paths.experiment
    else:
        root_path = config.root if hasattr(config, 'root') else '/tmp/experiment'
    
    output_dir = os.path.join(root_path, config.exp_name, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=None if config.report_to == 'no' else config.report_to,
        mixed_precision=config.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    return accelerator, output_dir

def main(args):
    config = OmegaConf.load(args.config)
    accelerator, output_dir = get_accelerator(config.train, config)
    
    # Use unified model and data loading functions
    denoiser, tokenizer = get_model_by_config(config)
    # Pass ``max_length`` from the config so sequences are capped.  Without
    # this, ``get_dataloader_by_config`` defaults to max_length=1024 and
    # ignores ``config.data.max_length``, letting sequences reach ~600 tokens.
    # That forces the differentiable DMD rollout to do ~36-64 block-wise
    # grad forward passes (vs ~8 at max_length=128) — the dominant cost.
    data_max_length = config.data.get('max_length', 1024) if hasattr(config, 'data') else 1024
    dataloader = get_dataloader_by_config(tokenizer, config.data, config, max_length=data_max_length)
    
    if config.train.decoder_resume_path is not None:
        ckpt = torch.load(config.train.decoder_resume_path, map_location='cpu', weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = denoiser.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            print(f'model ckpt loaded from {config.train.decoder_resume_path}')

        # ckpt = torch.load(config.train.head_resume_path, map_location='cpu', weights_only=True)
        # if config.train.skipped_keys:
        #     ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        # m, u = denoiser.lm_head.load_state_dict(ckpt, strict=False)
        # if accelerator.is_main_process:
        #     print(f'model ckpt loaded from {config.train.head_resume_path}')

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(param for param in denoiser.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )
    
    denoiser, dataloader, optimizer = accelerator.prepare(denoiser, dataloader, optimizer)

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))

    training_done = False
    epoch = 0

    # Epoch-based training: iterate over epochs, each epoch covers the full
    # dataloader once.  ``num_iters`` (if set) is an iteration cap that stops
    # training mid-epoch; ``num_epochs`` (if set) stops after N full epochs.
    # At least one of the two should be set in config.
    num_iters_cap = config.train.get('num_iters', None)
    num_epochs = config.train.get('num_epochs', None)

    # ``global_step`` counts optimizer updates, while ``len(dataloader)``
    # counts micro-batches.  Keep the progress bar in optimizer-step units so
    # gradient accumulation does not overstate training progress.
    accumulation_steps = config.train.gradient_accumulation_steps
    updates_per_epoch = (len(dataloader) + accumulation_steps - 1) // accumulation_steps
    if num_iters_cap is not None:
        total_steps = num_iters_cap
    elif num_epochs is not None:
        total_steps = global_step + num_epochs * updates_per_epoch
    else:
        total_steps = None

    progress_bar = tqdm(
        total   = total_steps,
        initial = global_step,
        desc    = 'Optimizer steps',
        disable = not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        print(f'Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e9} B')

    # --- Async rollout pipeline (optional) -------------------------------
    # If train.async_rollout_device is set (e.g. 'cuda:1'), create a
    # dedicated rollout model on that device. Rollout for batch t+1 runs
    # on the rollout GPU while loss + backward for batch t runs on the
    # training GPU.
    async_pipeline = None
    use_async = (
        hasattr(config.train, 'async_rollout_device')
        and config.train.async_rollout_device
    )
    if use_async and accelerator.is_main_process:
        from utils.async_pipeline import AsyncRolloutPipeline
        rollout_device = torch.device(config.train.async_rollout_device)
        training_mode = config.get('training_mode', 'dream')
        is_llada = (training_mode == 'llada')
        async_pipeline = AsyncRolloutPipeline(
            train_model=denoiser,
            rollout_device=rollout_device,
            block_size=config.train.block_size,
            mask_id=126336 if is_llada else config.denoiser.encoder.mask_id,
            eos_id=tokenizer.eos_token_id,
            is_llada=is_llada,
            shift=(not is_llada) and config.train.enable_shift,
            student_decode_steps=config.train.get('student_decode_steps', 1),
            temperature=config.train.get('temperature', 0.8),
            top_p=config.train.get('top_p', 0.95),
        )
        print(f'[async] Rollout pipeline on {rollout_device}')

    # --- EMA LoRA for DMD fake model (optional) --------------------------
    # When dmd_loss is enabled, create an EMA copy of the LoRA adapter
    # weights.  The EMA model serves as the "fake" distribution in the
    # DMD density-ratio score: c = log p_teacher - log p_fake.
    # On-policy losses construct their rollout synchronously so the active
    # student weights and attention masks remain aligned with the loss.
    ema_lora = None
    dmd_loss = getattr(config.train, 'dmd_loss', False)
    transition_csm = getattr(config.train, 'transition_csm', False)
    final_draft_remask = getattr(config.train, 'final_draft_remask', False)
    if dmd_loss or transition_csm:
        if async_pipeline is not None:
            raise ValueError("dmd_loss/transition_csm and async_rollout_device are mutually exclusive.")
    if final_draft_remask and async_pipeline is not None:
        raise ValueError("final_draft_remask and async_rollout_device are mutually exclusive.")
    if dmd_loss or transition_csm:
        from utils.ema_lora import EMALoRA
        ema_decay = getattr(config.train, 'dmd_ema_decay', 0.999)
        ema_lora = EMALoRA(denoiser, decay=ema_decay)
        print(f'[dmd] EMA LoRA fake model (decay={ema_decay})')

    def save_checkpoint(name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            denoiser.eval()
            accelerator.unwrap_model(denoiser).save_pretrained(
                os.path.join(output_dir, name)
            )
        accelerator.wait_for_everyone()

    adaptive_aux_weight = config.train.get('aux_remaining_weight', 0.0)
    adaptive_csm_ema = None
    adaptive_ce_ema = None
    adaptive_last_csm_norm = torch.zeros((), device=accelerator.device)
    adaptive_last_ce_norm = torch.zeros((), device=accelerator.device)

    # --- Per-step training closure ---------------------------------------
    # Extracted so the async-pipeline loop can train on the *previously*
    # fetched batch (whose rollout result is in hand) while the *next* batch
    # is rolling out on the rollout GPU.  ``rollout_results`` is None for the
    # synchronous path (loss fn runs the rollout inline).
    def train_one_step(batch, rollout_results):
        nonlocal global_step, training_done, adaptive_aux_weight, adaptive_csm_ema, adaptive_ce_ema, adaptive_last_csm_norm, adaptive_last_ce_norm
        with accelerator.accumulate([denoiser]):
            denoiser.train()
            input_ids = batch['data']
            question_length = batch['question_length']
            # ``length`` = question_length + answer_length (real tokens only);
            # positions >= length are pure padding added by dynamic padding.
            # The on-policy loss uses this to exclude padding positions so
            # the reported loss is consistent across batch sizes / pad len.
            lengths = batch.get('length', None)

            # Use unified loss function selection. When the async pipeline
            # supplies ``rollout_results``, the loss fn skips its internal
            # rollout and consumes these directly.
            probe_every = config.train.get('adaptive_weight_probe_every', 0)
            adaptive_enabled = config.train.get('adaptive_weighting', False)
            probe = None
            if transition_csm and adaptive_enabled and probe_every and global_step % probe_every == 0:
                def probe(csm_norm, ce_norm):
                    nonlocal adaptive_csm_ema, adaptive_ce_ema, adaptive_last_csm_norm, adaptive_last_ce_norm, adaptive_aux_weight
                    adaptive_last_csm_norm = accelerator.gather(csm_norm.detach()).mean()
                    adaptive_last_ce_norm = accelerator.gather(ce_norm.detach()).mean()
                    if not (torch.isfinite(adaptive_last_csm_norm) and torch.isfinite(adaptive_last_ce_norm)):
                        return
                    if adaptive_csm_ema is None:
                        adaptive_csm_ema = adaptive_last_csm_norm
                        adaptive_ce_ema = adaptive_last_ce_norm
                    else:
                        ema_decay = config.train.get('adaptive_weight_ema_decay', 0.9)
                        adaptive_csm_ema = ema_decay * adaptive_csm_ema + (1 - ema_decay) * adaptive_last_csm_norm
                        adaptive_ce_ema = ema_decay * adaptive_ce_ema + (1 - ema_decay) * adaptive_last_ce_norm
                    target_ratio = config.train.get('adaptive_weight_target_ce_ratio', 0.3)
                    update_rate = config.train.get('adaptive_weight_update_rate', 0.1)
                    target = target_ratio * adaptive_csm_ema / adaptive_ce_ema.clamp_min(1e-8)
                    target = torch.nan_to_num(target, nan=adaptive_aux_weight, posinf=config.train.get('adaptive_weight_max', 10.0), neginf=config.train.get('adaptive_weight_min', 0.01))
                    adaptive_aux_weight = float((adaptive_aux_weight * (target / max(adaptive_aux_weight, 1e-8)).pow(update_rate)).clamp(
                        config.train.get('adaptive_weight_min', 0.01), config.train.get('adaptive_weight_max', 10.0)
                    ).item())
            if adaptive_enabled:
                config.train.aux_remaining_weight = adaptive_aux_weight
            losses = compute_loss_by_config(
                input_ids,
                denoiser,
                question_length,
                block_size    = config.train.block_size,
                mask_id       = config.denoiser.encoder.mask_id,
                enable_shift  = config.train.enable_shift,
                share_steps   = config.train.share_steps,
                self_align    = config.train.self_align,
                feature_align = config.train.feature_align,
                self_step     = config.train.self_step,
                eos_id        = tokenizer.eos_token_id,
                config        = config,
                rollout_results = rollout_results,
                lengths       = lengths,
                ema_lora      = ema_lora,
                backward_callback = accelerator.backward if (transition_csm or final_draft_remask) else None,
                gradient_probe_callback = probe,
            )

            if config.train.share_steps > 1:
                loss_tgt = losses['loss']
                # loss_1 = losses['loss_1']
                # loss_2 = losses['loss_2']
            else:
                raise NotImplementedError
            torch.cuda.empty_cache()
            if not losses.get('backward_done', False):
                accelerator.backward(loss_tgt)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_learn, 1.0)

            optimizer.step()
            optimizer.zero_grad()

            # Update EMA fake model after optimizer step (DMD only)
            if ema_lora is not None and accelerator.sync_gradients:
                ema_lora.update(denoiser)

            # Sync updated LoRA weights to the rollout GPU so the next rollout
            # sees the latest adapter (1-step stale under prefetching — see
            # AsyncRolloutPipeline.submit_and_get docstring).
            if async_pipeline is not None and accelerator.sync_gradients:
                async_pipeline.sync_lora_weights()

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

            def gather_loss(name):
                value = losses.get(name)
                if value is None:
                    value = torch.zeros((), device=loss_tgt.device)
                return accelerator.gather(value.detach()).mean().item()

            logs = {
                'total_loss': gather_loss('loss'),
                'dmd_loss': gather_loss('transition_dmd_loss'),
                'ce_loss': gather_loss('aux_remaining_loss'),
                'adaptive_aux_weight': adaptive_aux_weight,
                'csm_grad_norm': adaptive_last_csm_norm.item(),
                'ce_grad_norm': adaptive_last_ce_norm.item(),
                'adaptive_csm_ema': adaptive_csm_ema.item() if adaptive_csm_ema is not None else 0.0,
                'adaptive_ce_ema': adaptive_ce_ema.item() if adaptive_ce_ema is not None else 0.0,
            }
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)

        if global_step > 0 and global_step % config.train.eval_every == 0 and accelerator.is_main_process:
            denoiser.eval();
            question = 'Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?'
            # prompt = tokenizer(question)['input_ids']
            # prompt = torch.tensor(prompt).to(accelerator.device).unsqueeze(0)
            messages = [
                {"role": "user", "content": question}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
            ).input_ids
            prompt = prompt.to(accelerator.device)

            mask_id = 151666
            gen_len = 512 - prompt.shape[1]
            temperature = 0.2
            top_p = 0.95

            x_t = torch.cat([prompt, torch.tensor([[mask_id]*gen_len]).to(accelerator.device)], dim=1)
            with torch.inference_mode():
                for i in range(gen_len):
                    mask_index = (x_t == mask_id)
                    if i % 2 == 0:
                        z_t = denoiser.module.encoder(x_t, output_hidden_states=True).hidden_states[-1]
                        hidden_state = denoiser.module.decoder(x_t, z_t)
                        logits = denoiser.module.encoder.lm_head(hidden_state)
                    else:
                        hidden_state = denoiser.module.decoder(x_t, z_t)
                        logits = denoiser.module.lm_head(hidden_state)

                    if config.train.enable_shift:
                        logits = shift_logits(logits)

                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=None, neg_entropy=True)

                    number_transfer_tokens = 1
                    _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=accelerator.device, dtype=torch.long) + mask_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x_t[mask_index] = x0_

            answer = tokenizer.batch_decode(x_t[:, prompt.shape[1]:], skip_special_tokens=True)[0]
            print(answer)

        if (
            accelerator.sync_gradients
            and global_step > 0
            and global_step % config.train.save_every == 0
        ):
            save_checkpoint(f"Decoder-{config.train.exp_name}-{global_step // 1000}k")
        # Stop if we've reached the iteration cap (if set)
        if num_iters_cap is not None and global_step >= num_iters_cap:
            training_done = True

    def _payload(batch):
        return {'data': batch['data'].cpu(), 'question_length': batch['question_length'].cpu()}

    # --- Main epoch loop --------------------------------------------------
    # Async prefetching: prime the pipeline with batch 0, then for each
    # subsequent batch call submit_and_get (returns batch t's result while
    # starting batch t+1's rollout), and train on batch t.  When the
    # dataloader is exhausted, drain the final in-flight rollout.  The
    # synchronous path (no async_pipeline) just trains each batch inline.
    while not training_done:
        if accelerator.is_main_process:
            print(f'Epoch: {epoch}')
        data_iter = iter(dataloader)
        first_batch = next(data_iter, None)
        if first_batch is None:
            break  # empty dataloader

        if async_pipeline is not None:
            async_pipeline.prime_submit(_payload(first_batch))
        pending_batch = first_batch
        broke_early = False
        for batch in data_iter:
            if async_pipeline is not None:
                # Overlap: enqueue this batch's rollout, fetch previous result
                rollout_results = async_pipeline.submit_and_get(
                    _payload(batch), accelerator.device
                )
            else:
                rollout_results = None
            train_one_step(pending_batch, rollout_results)
            if training_done:
                broke_early = True
                break
            pending_batch = batch

        if not broke_early and not training_done:
            # Dataloader exhausted (end of epoch): drain the last rollout
            if async_pipeline is not None:
                rollout_results = async_pipeline.get_last_result(accelerator.device)
            else:
                rollout_results = None
            train_one_step(pending_batch, rollout_results)

        completed_epoch = not broke_early
        epoch += 1
        if completed_epoch:
            save_checkpoint(
                f"Decoder-{config.train.exp_name}-epoch-{epoch}-step-{global_step}"
            )
        # Stop if we've completed all epochs (if num_epochs is set)
        if num_epochs is not None and epoch >= num_epochs:
            training_done = True
        if training_done:
            break
    if async_pipeline is not None:
        async_pipeline.stop()
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()

    # The async rollout model lives on a second GPU via a daemon worker
    # thread. Even after stop() joins the thread and frees the model, the
    # C++ runtime can abort at interpreter shutdown ("terminate called
    # without an active exception" / SIGABRT) from a joinable CUDA thread
    # tied to that device.  All training work and checkpoint saves are
    # already complete by this point, so force-exit to skip the problematic
    # C++ teardown.  (Standard practice in distributed PyTorch to avoid
    # cleanup-time aborts.)
    if async_pipeline is not None:
        os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dream.yaml')
    args = parser.parse_args()
    main(args)    