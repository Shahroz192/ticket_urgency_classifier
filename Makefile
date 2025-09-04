#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ticket_urgency_classifier
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\.venv\Scripts\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Validate raw data schema
.PHONY: validate-raw
validate-raw: requirements
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier validate raw


## Validate processed data schema
.PHONY: validate-processed
validate-processed: requirements
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier validate processed


## Make dataset
.PHONY: data
data: validate-raw
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier dataset


## Make features
.PHONY: features
features: data
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier features


## Make model
.PHONY: model
model: features validate-processed
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier model


## Run batch predictions on input data
.PHONY: predict
predict: model
	@echo "Usage: make predict INPUT_PATH=path/to/input.csv"
	@if [ -z "$(INPUT_PATH)" ]; then echo "Error: INPUT_PATH is required"; exit 1; fi
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier predict batch --input $(INPUT_PATH)


## Serve the prediction API
.PHONY: serve-api
serve-api: model
	uvicorn ticket_urgency_classifier.api.main:app --host 0.0.0.0 --port 8000


#################################################################################
# PIPELINES                                                                     #
#################################################################################

## Run the feature engineering pipeline
.PHONY: feature-pipeline
feature-pipeline: requirements validate-raw
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier feature-pipeline


## Run the model training pipeline
.PHONY: model-pipeline
model-pipeline: requirements validate-raw
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier model-pipeline


## Run the full model training pipeline
.PHONY: full-pipeline
full-pipeline: requirements validate-raw
	$(PYTHON_INTERPRETER) -m ticket_urgency_classifier full-pipeline


#################################################################################
# DOCKER RULES                                                                  #
#################################################################################

## Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(PROJECT_NAME) .

## Run the Docker container
.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 $(PROJECT_NAME)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
