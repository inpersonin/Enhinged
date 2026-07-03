# Enhinged

Enhinged is a GPT-2 style conversational language model trained from scratch on Hinglish conversational data. The repository is backend-only and is organized for training, checkpointing, preprocessing, and FastAPI-based inference.

## Overview

The model uses a causal transformer architecture with learned token and position embeddings, masked multi-head attention, feed-forward blocks, residual connections, and tied input/output embeddings. The codebase keeps the model definition, training loop, preprocessing pipeline, and inference backend separate so it can be deployed cleanly or imported from another project.

## Features

- GPT-2 style architecture implemented in PyTorch
- conversational generation
- Memory-mapped dataset loading for efficient training
- Checkpoint saving and resuming
- Greedy and sampling-based decoding with top-k and top-p controls
- FastAPI backend for production-style inference
- Optional loading of official GPT-2 pretrained weights for fine-tuning

## Model Architecture

The default configuration uses a compact transformer suitable for local training and deployment:

- `block_size = 256`
- `n_layer = 6`
- `n_head = 6`
- `n_embd = 384`
- `vocab_size = 50257`

Pipeline:

1. Token embeddings and position embeddings are added.
2. A stack of transformer blocks applies masked self-attention and feed-forward layers.
3. Final layer normalization produces hidden states.
4. The language modeling head projects hidden states to vocabulary logits.
5. The token embedding matrix is tied to the output head.

## Training Overview

Training uses a memory-mapped dataset of token IDs and a standard GPT-2 style optimization loop with gradient clipping, warmup, and cosine learning-rate decay. Checkpoints include model weights, optimizer state, configuration, and iteration metadata so training can be resumed safely.

## Dataset

The preprocessing script converts raw Hinglish text into token IDs using the GPT-2 BPE tokenizer from `tiktoken`.

Outputs:

- `train.bin`
- `val.bin`
- `meta.json`

The binary files are stored as memory-mapped arrays so the trainer can sample batches without loading the full corpus into RAM.

## Project Structure

```text
.
├── api.py
├── config.py
├── inference.py
├── model.py
├── prepare_data.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Prepare Data

Single file:

```bash
python prepare_data.py \
  --input_file data/enhinged_corpus.txt \
  --output_dir data/ \
  --val_ratio 0.1
```

Directory of text files:

```bash
python prepare_data.py \
  --input_dir data/scripts/ \
  --output_dir data/ \
  --val_ratio 0.1
```

## Run Training

Train from scratch:

```bash
python train.py \
  --mode train \
  --data_dir data/ \
  --out_dir checkpoints/ \
  --block_size 256 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --max_iters 5000 \
  --batch_size 32
```

Fine-tune from pretrained GPT-2 weights:

```bash
python train.py \
  --mode train \
  --init_from gpt2 \
  --data_dir data/ \
  --out_dir checkpoints/ \
  --max_iters 3000
```

Resume from a checkpoint:

```bash
python train.py \
  --mode train \
  --ckpt_path checkpoints/latest.pt \
  --data_dir data/ \
  --out_dir checkpoints/
```

## Run Inference

Load a checkpoint from Python:

```python
from inference import load_model, generate_response

load_model("checkpoints/best.pt")
reply = generate_response("Hello, how are you?")
print(reply)
```

Start the FastAPI service:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, how are you?"}'
```

## Loading Checkpoints

The inference layer uses `load_model()` and `unload_model()` for lifecycle management. By default it loads `checkpoints/best.pt`, but you can point it to any compatible checkpoint created by the training script.

## FastAPI Backend

The backend exposes a small HTTP surface intended for integration with another application:

- `GET /health`
- `POST /generate`
- `POST /load`
- `POST /unload`

This makes it straightforward to connect the model to a custom React, Next.js, or other client later without changing the inference code.

### Deployment Notes

For Hugging Face Spaces with the Docker SDK:

- keep the checkpoint at `checkpoints/best.pt`
- set `ENHINGED_CKPT_PATH` if you want to point at a different checkpoint
- set `ENHINGED_CORS_ORIGINS` to your GitHub Pages origin, or leave it as `*` for open cross-origin access
- the container listens on port `7860`

The intended split is:

- backend: Hugging Face Space running this FastAPI app
- frontend: static GitHub Pages site calling the HF Space `/generate` endpoint

## Frontend

The repository now also includes a Next.js frontend scaffold designed for GitHub Pages export. It uses a brutalist void layout with a restrained glass chat panel, and the message input lives in the hero itself.

Frontend build settings:

- set `NEXT_PUBLIC_API_URL` to your HF Space URL if it differs from `https://inpersonin-enhinged.hf.space`
- set `NEXT_PUBLIC_BASE_PATH` during build if you publish under a project pages path
- the GitHub Actions workflow in `.github/workflows/deploy-pages.yml` builds the static export and deploys it to Pages

## Future Deployment

Recommended next steps for production deployment:

- Package the service behind a reverse proxy
- Add request authentication if the API is public
- Introduce structured logging and tracing
- Add request batching if traffic increases
- Pin the checkpoint path through environment variables for deployment
- Add automated tests for API and sampling behavior

## Notes

- The repository now includes both the FastAPI backend and the static Next.js frontend scaffold.
- The project is designed to be published on GitHub Pages for the frontend and Hugging Face Spaces for the backend.
