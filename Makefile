SHELL = /bin/sh
IMAGE = techiaith/welsh-evals
default: build

build:
	docker build --rm -t $(IMAGE) .

# Run all evals: make eval MODEL=gpt-4o
# Run one eval:  make eval MODEL=gpt-4o EVAL=welsh-lexicon
# Limit samples: make eval MODEL=gpt-4o MAX_SAMPLES=50
eval: build
	docker run --rm -it --name techiaith-welsh-evals \
		--network llm-evals \
		--env-file=openai.env \
		-v ${PWD}/results:/app/results \
		$(IMAGE) python -m deepeval_evals.run_all \
		--model $(MODEL) \
		$(if $(EVAL),--eval $(EVAL)) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES))

# Run pytest-style: make test MODEL=gpt-4o
test: build
	docker run --rm -it --name techiaith-welsh-evals \
		--network llm-evals \
		--env-file=openai.env \
		-v ${PWD}/results:/app/results \
		$(IMAGE) pytest deepeval_evals/tests/ \
		--model $(MODEL) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES)) \
		-v

# Interactive shell for creating eval data
create: build
	docker run --rm -it --name techiaith-create-evals \
		--network llm-evals \
		--env-file=openai.env \
		-v ${PWD}/evals-cymraeg:/app/evals-cymraeg \
		-v ${PWD}/src:/app/src \
		-v ${PWD}/results:/app/results \
		$(IMAGE) bash

clean:
	-docker rmi $(IMAGE)
