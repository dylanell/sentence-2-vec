# Sentence-2-Vec

Learning sentence vectors from Question-Answer pairs and Triplet Networks.

### Makefile Rules

The Makefile contains several rules for environment setup, dataset configuration, training the model, serving the model, etc. Descriptions of each rule and the command to run are given below.

1. `$ make setup`: 

    Use `venv` to create a virtual environment under a `virtualenv/` directory located at the root of this project. Initialize this virtual environment with the command `$ source virtualenv/bin.activate`.

2. `$ make data`:

    Scrape Question-Answer (QA) pairs from a topic on [answers.com](answers.com). Both raw QA pairs and their preprocessed versions are written to a `sqlite` DB. Edit `scrape_cfg.yml` configuration file to change the topic to scrape, the path which to write dataset artifacts, and the maximum number of samples to collect.

3. `$ make train`:

    Train the triplet model to learn sentence vectors from scraped QA data. 