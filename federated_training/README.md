# Federated LLM Fine-Tuning with QLoRA

Federated fine-tuning of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) across 5 simulated hospitals using the [CARDBiomedBench](https://huggingface.co/datasets/NIH-CARD/CARDBiomedBench) dataset. Each hospital trains locally on a private biomedical category; only LoRA adapter weights are sent to the server for aggregation — raw data never leaves the client.


### Data partitioning

Each of the 5 clients is assigned one biomedical category from CARDBiomedBench:

| Client | Category |
|--------|----------|
| 1 | Drug Gene Relations |
| 2 | Pharmacology |
| 3 | Drug Meta |
| 4 | SNP Disease Relations |
| 5 | SMR Gene Disease Relations |

Each client trains on a subset of the data samples. Data is stored locally at `client/data/clients/{client_id}/hf_dataset.pt` and never leaves the client.

### Model

- **Base model**: `HuggingFaceTB/SmolLM2-135M` (135M parameters)
- **Adapter**: LoRA — rank=8, alpha=16, targets `q_proj` + `v_proj`, dropout=0.1
- **Trainable params**: ~460K (0.34% of total) — the only weights exchanged with the server

### Device-adaptive training

The client auto-detects hardware and applies the best available training strategy:

| Hardware | Strategy | Base model dtype | LoRA dtype |
|----------|----------|------------------|------------|
| CUDA, new hardware | QLoRA (4-bit NF4) | 4-bit NF4 | bfloat16 |
| CUDA, older hardware | QLoRA (4-bit NF4) | 4-bit NF4 | float16 |
| CUDA, no bitsandbytes | LoRA | bfloat16 or float16 | bfloat16 or float16 |
| Apple Silicon (MPS) | LoRA | bfloat16 | bfloat16 |
| CPU | LoRA | float32 | float32 |

Regardless of local dtype, LoRA adapter weights are always serialized as **float32** before being sent to the server, so FedAvg aggregation is numerically consistent across all client types.

## Setup

```bash
pip install scaleout
```

### Running client steps manually

From `federated_training/`:

```bash
# Create package:
scaleout package create --path  client/

# Create seed model: 
scaleout run build --path client/

# start the client
scaleout client start --api-url http://localhost:8092
```

Then start a session via `api.ipynb`.


### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCALEOUT_DATA_PATH` | `client/data/clients/1/hf_dataset.pt` | Path to local training data |

Change the environment variable so the client loads data from a different data partition.