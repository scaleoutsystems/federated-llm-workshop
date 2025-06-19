import collections
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, TaskType, LoraConfig

from fedn.utils.helpers.helpers import get_helper


HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model():
    model_name = "HuggingFaceTB/SmolLM2-135M"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.config.pad_token_id = model.config.eos_token_id  # Tell the model which token to use for padding

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def save_lora_parameters(model, out_path):
    """Saves the LoRA adaperts, not the full model"""
    lora_state_dict = get_peft_model_state_dict(model)
    parameters_np = [val.cpu().numpy() for _, val in lora_state_dict.items()]
    helper.save(parameters_np, out_path)


def load_lora_parameters(model_path):
    """Loads the LoRA adaperts, not the full model"""
    peft_model = compile_model()
    parameters_np = helper.load(model_path)

    peft_model_statedict = get_peft_model_state_dict(peft_model)
    params_dict = zip(peft_model_statedict.keys(), parameters_np)
    lora_state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in params_dict}
    )

    set_peft_model_state_dict(peft_model, lora_state_dict)

    return peft_model


def init_seed(out_path="seed.npz"):
    """Initialize seed model and save it to file. 
    Only lora parameters are uploaded to server

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_lora_parameters(model, out_path)


if __name__=="__main__":
    init_seed("../seed.npz")