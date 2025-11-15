# Makefile for lab_project_fresh_guard
# Targets:
#  - install: create virtualenv and install requirements
#  - train: run the training pipeline (produces artifacts)
#  - serve: start the Flask server (no retrain)
#  - clean: remove generated artifacts
#  - status: show git short status

.PHONY: install train serve clean status

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

train:
	@echo "Running training pipeline (this will create model artifacts)..."
	.venv/bin/python app.py

serve:
	@echo "Starting Flask server on port 8080 (no training)..."
	.venv/bin/python app.py --serve

clean:
	@echo "Removing generated artifacts (*.pkl, *.png, metadata)"
	rm -f *.pkl metadata.pkl prediction_pipeline.pkl
	rm -f *.png
	rm -rf __pycache__

status:
	@git status --short
