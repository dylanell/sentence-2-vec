PYTHON = python3

.PHONY:

setup:
	pip install -r requirements.txt

data:
	${PYTHON} scrape.py

train:
	${PYTHON} train.py

local:
	${PYTHON} app.py

docker:
	docker-compose up