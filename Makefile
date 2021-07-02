PYTHON = python3

.PHONY:

setup:
	bash scripts/env_setup.sh

data:
	${PYTHON} scrape.py

train:
	${PYTHON} train.py

local:
	${PYTHON} app.py