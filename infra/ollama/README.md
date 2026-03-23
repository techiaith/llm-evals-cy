# Ollama GPU Server

Run Ollama with GPU acceleration on a remote Linux server for the Welsh LLM evaluation pipeline.

## Quick Start

### 1. Setup the server (one-time)

Prerequisites: Docker + NVIDIA GPU drivers on a Linux server.

Install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Start Ollama and pull a model

Copy this directory to the server and run:

```bash
make up           # Start Ollama container with GPU support
make gpu-check    # Verify GPUs are visible
make pull         # Download llama3 (8B) + llama3.2:1b
make test         # Quick sanity check — asks llama3 to say hello in Welsh
```

### 3. Run evals from your Mac

Add the server address to `openai.env` in the repo root:

```
OLLAMA_API_BASE=http://YOUR_SERVER_IP:11434
```

Then run a small smoke test to verify the pipeline works end-to-end:

```bash
pip install -r requirements.txt
python -m deepeval_evals.run_all --model ollama/llama3 --eval welsh-obscenities --max-samples 10
```

If that works, run a full eval:

```bash
python -m deepeval_evals.run_all --model ollama/llama3 --eval welsh-obscenities
```

Or all evals:

```bash
python -m deepeval_evals.run_all --model ollama/llama3
```

## Server Makefile targets

```bash
make up          # Start Ollama server
make pull        # Download llama3 (8B) + llama3.2:1b
make list        # Show downloaded models
make gpu-check   # Verify GPU access inside container
make test        # Quick test (asks llama3 to say hello in Welsh)
make logs        # Follow container logs
make down        # Stop server
make clean       # Stop + delete downloaded models
```

### Pull additional models

```bash
make pull-model MODEL=mistral
make pull-model MODEL=gemma2:27b
```

## Model recommendations for Welsh evals

| Model | VRAM needed | Notes |
|-------|-------------|-------|
| `llama3.2:1b` | ~1.3 GB | Very fast, useful for checking the pipeline works |
| `llama3.2:3b` | ~2.5 GB | Better quality, still small |
| `llama3` (8B) | ~5 GB | Good balance of speed and quality |
| `mistral` (7B) | ~5 GB | Good multilingual support |
| `gemma2:27b` | ~18 GB | Strong multilingual |
