import json
import collections

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, TaskType


from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

model = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model)


def generate_instruction_format(example):
    question = example["question"]
    answer = example["answer"]

    instruction = "You are a knowledgeable assistant. Answer this question truthfully!"

    prompt = (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Input:\n"
        f"{question.strip()}\n\n"
        "### Response:\n"
        f"{answer.strip()}" + tokenizer.eos_token
    )
    return {"text": prompt}


def save_list(list, path):
    with open(path, 'w') as f:
        json.dump(list, f, indent=4)

def load_list(path):
    with open(path, 'r') as f:
        ls = json.load(f)
    return ls

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        scores.append(score)
    
    # Average scores
    avg_scores = {
        'rouge1': sum(s['rouge1'].fmeasure for s in scores) / len(scores),
        'rouge2': sum(s['rouge2'].fmeasure for s in scores) / len(scores),
        'rougeL': sum(s['rougeL'].fmeasure for s in scores) / len(scores),
    }
    return avg_scores


def compile_model():
    model_name = "HuggingFaceTB/SmolLM2-135M"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.config.pad_token_id = model.config.eos_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model


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