import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig

def get_model_by_config(config):
    """Select different models based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return get_llada(config)
    elif training_mode == 'dream':
        return get_model(config)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def get_model(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/home/wx/data/model/Dream-org/Dream-v0-Base-7B"
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    # print(model.named_modules())
    # print(model,"model
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "o_proj"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

def get_llada(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/data1/xck/models/llada-8b-instruct"
    
    config_obj=LLaDAConfig.from_pretrained(model_path)
    # Load weights in fp16 so the 8B model occupies ~16 GB (vs ~32 GB in
    # fp32).  This is essential for the differentiable DMD rollout, which
    # keeps the autograd graph across multiple block-wise forward passes and
    # needs the freed headroom for activations.  Mixed-precision autocast
    # (config.train.mixed_precision="fp16") still wraps the forward, so
    # compute stays in fp16; LoRA adapters are kept in fp32 below for stable
    # gradients (standard mixed-precision master-weights pattern).
    model = LLaDAModelLM.from_pretrained(model_path, config=config_obj, torch_dtype=torch.float16)
    # print(model.named_modules())
    # print(model,"model"
    # print(model)
    # exit()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "attn_out"],)
    model = get_peft_model(model, peft_config)
    # Keep trainable LoRA adapters in fp32 (master weights) for gradient
    # stability while the frozen 8B base stays in fp16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()
    return model, tokenizer
# def create_attention_mask(input_ids, mask_id):
#     """
#     Create an attention mask based on the input_ids and mask_id.

#     Args:
#         input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
#         mask_id (int): The ID of the mask token.

#     Returns:
#         torch.Tensor: The attention mask of shape (batch_size, sequence_length, sequence_length).
