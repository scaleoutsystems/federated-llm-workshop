import torch
from fedn.utils.helpers.helpers import save_metadata
import sys
import json
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from data import load_data
from model import save_lora_parameters, load_lora_parameters


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as padding token

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


def train(
    in_model_path, out_model_path, data_path=None, batch_size=4, epochs=1, lr=2e-4
):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """

    lora_model = load_lora_parameters(in_model_path)

    train_dataset = load_data(data_path)

    train_dataset = train_dataset.map(
        generate_instruction_format,
        remove_columns=train_dataset.column_names,
        batched=False,
    )

    use_cuda = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir="qa-finetuned",
        per_device_train_batch_size=batch_size, 
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=5e-4,
        logging_steps=20,
        save_total_limit=2,
        use_cpu=not(use_cuda)
    )

    trainer = SFTTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
    )

    training_output = trainer.train()

    metrics = {
        "training_loss": training_output.training_loss
    }

    # save metrics file
    with open('./metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)


    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(train_dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save lora statedict (to be sent to server)
    save_lora_parameters(trainer.model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])