PYTHON = python3

.PHONY:

setup:
	pip install -r requirements.txt

data:
	${PYTHON} scrape.py

train:
	${PYTHON} train.py

#test:
#	${PYTHON} -m pytest -v

#dashboard:
#	${PYTHON} deploy/dashboard.py