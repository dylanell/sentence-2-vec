# sentence-2-vec
Learning sentence representations from domain-specific QA pairs scraped from the web.

### Environment:

- Python 3.7.4

### Python Packages:

- pytorch
- torchtext
- pyyaml

### QA Pairs Text Dataset Format:

This project assumes you have a dataset of question-answer pairs pre
 -configured locally on your machine and saved as a `csv` file. My [dataset
  -helpers](https://github.com/dylanell/dataset-helpers) Github project also
   contains tools that perform this local configuration automatically for data
    scraped from [answers.com](https://www.answers.com/). The `csv` file should
     be in the following format:

```
question, answer
question one ?, answer one .
question two ?, answer two .
...
```

The dataset processing in this project assumes the text data in each row of
 the `csv` datafile is ready to be tokenized by whitespace alone. For example
 , row one of the example `csv` file above would yield a tokenized question
  example as `['question', 'one', '?']`. It is important to note that if the
   question of row one was `question one?`, the tokenization in this project
    would yield `['question', 'one?']`, therefore punctuation must be
     pre-processed as separate words and separated by a 'space' character
      prior to generating the `csv` datafile for this project.

 ### Training:

 Training options and hyperparameters are pulled from the `config.yaml` file
  and can be changed by editing the file contents. The `train.py` script
   accepts only several specific values for the `model_type` variable in
    `config.yaml` corresponding to the type of NLP model you would like to
     train. Train a model by running the command:

```
$ python train.py
```

### Jupyter Notebook:

This project is accompanied by a Jupyter notebook that explores the learned
 sentence representations by performing data visualizations and cluster
  analysis. Run the following command to start the Jupyter notebook server in
   your browser:

```
$ jupyter-notebook notebook.ipynb
```

### References:

1. [A Comprehensive Introduction to Torchtext](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/)
2. [Load Datasets With Torchtext](https://dzlab.github.io/dltips/en/pytorch/torchtext-datasets/)
3. [CNN for Text](https://arxiv.org/pdf/1408.5882.pdf)
4. [Dynamic CNN (DCNN) for Text](https://arxiv.org/pdf/1404.2188.pdf)
5. [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
6. https://medium.com/@nainaakash012/simclr-contrastive-learning-of-visual-representations-52ecf1ac11fa
7. http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
8. https://arxiv.org/pdf/2010.05113.pdf
9. https://gombru.github.io/2019/04/03/ranking_loss/
10. [On the Surprising Behavior of Distance Metrics in High Dimensional Space](https://bib.dbvis.de/uploadedFiles/155.pdf)
11. [When is Nearest Neighbor Meaningful?](https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/beyer.pdf)
12. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
