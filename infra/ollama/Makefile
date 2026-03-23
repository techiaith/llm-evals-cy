SHELL = /bin/sh

# Default models to pull
MODELS = llama3 llama3.2:1b

default: up

# Start Ollama server with GPU support
up:
	docker compose up -d
	@echo "Ollama running at http://localhost:11434"
	@echo "Run 'make pull' to download models"

down:
	docker compose down

logs:
	docker compose logs -f

# Pull default models
pull:
	@for model in $(MODELS); do \
		echo "Pulling $$model..."; \
		docker exec ollama ollama pull $$model; \
	done

# Pull a specific model: make pull-model MODEL=llama3:70b
pull-model:
	docker exec ollama ollama pull $(MODEL)

# List downloaded models
list:
	docker exec ollama ollama list

# Check GPU visibility inside the container
gpu-check:
	docker exec ollama nvidia-smi

# Test that a model responds
test:
	curl -s http://localhost:11434/api/generate \
		-d '{"model":"llama3","prompt":"Say hello in Welsh","stream":false}' \
		| python3 -m json.tool

# Remove everything (including downloaded models)
clean:
	docker compose down -v
