import collections

import torch
from transformers import AutoModelForCausalLM
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    TaskType,
    LoraConfig,
)

from scaleoututil.helpers.helpers import get_helper
from scaleoututil.utils.model import ScaleoutModel

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def get_device_config():
    """Detect device capabilities and return (device, bnb_config, torch_dtype).
    Necessary to find optimal settings for QLoRA depending on device.

    Devices:
      - CUDA, new:                         QLoRA — 4-bit NF4 base, bfloat16 compute/LoRA
      - CUDA, old:                         QLoRA — 4-bit NF4 base, float16 compute/LoRA
      - CUDA, no bitsandbytes:             LoRA  — bfloat16 or float16 (matches GPU capability)
      - MPS  (Apple Silicon):              LoRA  — bfloat16
      - CPU  (any platform):               LoRA  — float32

    Returns:
        device    (str):                       "cuda", "mps", or "cpu"
        bnb_config (BitsAndBytesConfig|None):  quantization config for CUDA, None otherwise
        torch_dtype (torch.dtype):             bfloat16/float16 for accelerators, float32 for CPU
    """
    if torch.cuda.is_available():
        # use bfloat16 for training if available, has wider dynamic range
        # float16 as fallback for older GPUs
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            print(f"[Device config] CUDA — QLoRA (4-bit NF4), compute dtype: {compute_dtype}")
            return "cuda", bnb_config, compute_dtype
        except (ImportError, Exception):
            # bitsandbytes not available — fall back to plain LoRA on CUDA
            print(f"[Device config] CUDA — LoRA (bitsandbytes unavailable), dtype: {compute_dtype}")
            return "cuda", None, compute_dtype
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon supports bfloat16
        print("[Device config] MPS — LoRA, dtype: torch.bfloat16")
        return "mps", None, torch.bfloat16
    else:
        print("[Device config] CPU — LoRA, dtype: torch.float32")
        return "cpu", None, torch.float32


def compile_model():
    model_name = "HuggingFaceTB/SmolLM2-135M"
    device, bnb_config, dtype = get_device_config()

    if bnb_config is not None:
        # QLoRA path — 4-bit quantized base model; LoRA adapters train in compute dtype (bfloat16 or float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # LoRA for CUDA (no bitsandbytes), MPS, or CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        if device != "cpu":
            model = model.to(device)

    model.config.pad_token_id = model.config.eos_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def save_parameters(model):
    """Save LoRA adapter parameters to a ScaleoutModel.

    Always serialized as float32 numpy arrays so FedAvg aggregation is
    numerically consistent regardless of the client's training dtype.
    """
    lora_state_dict = get_peft_model_state_dict(model)
    parameters_np = [val.detach().cpu().float().numpy() for _, val in lora_state_dict.items()]
    return ScaleoutModel.from_model_params(parameters_np, helper)


def load_parameters(scaleout_model: ScaleoutModel):
    """Load LoRA parameters from a ScaleoutModel into a fresh compiled model."""
    peft_model = compile_model()
    parameters_np = scaleout_model.get_model_params(helper)

    peft_model_statedict = get_peft_model_state_dict(peft_model)
    params_dict = zip(peft_model_statedict.keys(), parameters_np)

    # Match device and dtype of the LoRA adapters (not the quantized base)
    lora_param = next(p for n, p in peft_model.named_parameters() if "lora_" in n)
    lora_device = lora_param.device
    lora_dtype = lora_param.dtype

    lora_state_dict = collections.OrderedDict(
        {key: torch.tensor(x, dtype=lora_dtype, device=lora_device) for key, x in params_dict}
    )
    set_peft_model_state_dict(peft_model, lora_state_dict)
    return peft_model


def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save LoRA parameters to file.
    Only LoRA adapter weights (~460K) are uploaded to the server.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    model = compile_model()
    lora_state_dict = get_peft_model_state_dict(model)
    parameters_np = [val.detach().cpu().float().numpy() for _, val in lora_state_dict.items()]
    helper.save(parameters_np, out_path)


def build():
    init_seed("seed.npz")


if __name__ == "__main__":
    init_seed("../seed.npz")
