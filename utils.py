import json

from transformers import AutoTokenizer
from rouge_score import rouge_scorer


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