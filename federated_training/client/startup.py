import json

import torch
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from scaleout import EdgeClient
from scaleoututil.utils.model import ScaleoutModel

from model import load_parameters, save_parameters
from data import load_data, prepare_data


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer.pad_token = tokenizer.eos_token


def startup(client: EdgeClient):
    prepare_data()
    MyClient(client)


def _get_device():
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client
        self.device = torch.device(_get_device())

        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)

    def train(
        self,
        scaleout_model: ScaleoutModel,
        settings,
        data_path=None,
        batch_size=4,
        epochs=1,
        lr=2e-4,
    ):
        lora_model = load_parameters(scaleout_model)

        train_dataset = load_data(data_path)

        train_dataset = train_dataset.map(
            generate_instruction_format,
            remove_columns=train_dataset.column_names,
            batched=False,
        )

        device = _get_device()

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            learning_rate=5e-4,
            logging_steps=20,
            save_total_limit=2,
            use_cpu=(device == "cpu"),
            dataloader_pin_memory=(device == "cuda"),
        )

        trainer = SFTTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=train_dataset,
        )

        training_output = trainer.train()

        self.client.log_metric({"training_loss": training_output.training_loss})

        metadata = {
            "num_examples": len(train_dataset),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        }

        result_model = save_parameters(trainer.model)
        return result_model, {"training_metadata": metadata}

    def validate(self, scaleout_model: ScaleoutModel, data_path=None):
        """Load metrics from the training step and return them."""
        try:
            with open("./metrics.json", "r") as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = {}

        return metrics
