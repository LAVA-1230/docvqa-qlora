# Reproduce every step. On Colab the paths are overridden, e.g.
#   make train DATA=/content/data/processed CKPT=/content/drive/MyDrive/docvqa-qlora/outputs/qlora
# (Windows has no `make`; copy the commands, or run them from the Colab notebook.)
PY      ?= python
DATA    ?= data/processed
RESULTS ?= results
CKPT    ?= outputs/qlora
ADAPTER ?= $(CKPT)/best_adapter
MERGED  ?= models/merged
LIMIT   ?=
LIMIT_ARG = $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: test data baseline train eval compare merge serve serve-mock docker docker-cpu

test:
	$(PY) -m pytest

data:
	$(PY) -m src.data.prepare_dataset --config configs/data.yaml --output_dir $(DATA) $(LIMIT_ARG)

baseline:
	$(PY) -m src.evaluate --config configs/eval.yaml --split test --data_dir $(DATA) --output $(RESULTS)/baseline.json $(LIMIT_ARG)

train:
	$(PY) -m src.train --config configs/train.yaml --data_dir $(DATA) --output_dir $(CKPT) --results_dir $(RESULTS) $(LIMIT_ARG)

eval:
	$(PY) -m src.evaluate --config configs/eval.yaml --split test --data_dir $(DATA) --adapter $(ADAPTER) --load_in_4bit --output $(RESULTS)/finetuned.json $(LIMIT_ARG)

compare:
	$(PY) -m src.compare --baseline $(RESULTS)/baseline.json --finetuned $(RESULTS)/finetuned.json --results_dir $(RESULTS)

merge:
	$(PY) -m src.merge_adapter --config configs/eval.yaml --adapter $(ADAPTER) --output $(MERGED) --data_dir $(DATA) --results_dir $(RESULTS)

serve:
	MERGED_MODEL_PATH=$(MERGED) uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

serve-mock:
	MODEL_MODE=mock uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

docker:
	docker build -t docvqa-api .
	docker run --gpus all -p 8000:8000 -v ./models:/models docvqa-api

docker-cpu:
	docker build -f Dockerfile.cpu -t docvqa-api-cpu .
	docker run -p 8000:8000 -v ./models:/models docvqa-api-cpu
