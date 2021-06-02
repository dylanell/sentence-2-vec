PYTHON = python3

.PHONY:

setup:
	pip install -r requirements.txt

data:
	${PYTHON} scrape.py

train:
	${PYTHON} train.py

app:
	${PYTHON} app.py

#test:
#	${PYTHON} -m pytest -v