.PHONY: clean dirs virtualenv lint requirements push pull reproduce

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Activate virtualenv.
activate:
	$(PYTHON_INTERPRETER) -m pipenv shell

## Install Python Dependencies.
dependencies: 
	$(PYTHON_INTERPRETER) -m pip install -U pipenv setuptools wheel
	$(PYTHON_INTERPRETER) -m pipenv shell
	$(PYTHON_INTERPRETER) -m pipenv install

## Import Python Dependencies from requirements.txt.
import: 
	$(PYTHON_INTERPRETER) -m pip install -U pipenv setuptools wheel
	$(PYTHON_INTERPRETER) -m pipenv shell
	$(PYTHON_INTERPRETER) -m pipenv import -r requirements.txt
	$(PYTHON_INTERPRETER) -m pipenv install

## Install dependencies defined on Pipenv.lock
install: 
	$(PYTHON_INTERPRETER) -m pipenv install --ignore-pipfile

## Check if all the dependencies are installed correctly
check: 
	$(PYTHON_INTERPRETER) -m pipenv check

## Dependencies graph
graph: 
	$(PYTHON_INTERPRETER) -m pipenv graph

## Update Pipfile.lock dependencies
lock: 
	$(PYTHON_INTERPRETER) -m pipenv lock

## Create directories that are ignored by git but required for the project
dirs:
	mkdir -p data/raw data/processed models

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to default DVC remote
push:
	dvc push

## Download Data from default DVC remote
pull:
	dvc pull

## Reproduce the DVC pipeline - recompute any modified outputs such as processed data or trained models
reproduce:
	dvc repro eval.dvc

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / Missing" $Missing \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'Missing \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')