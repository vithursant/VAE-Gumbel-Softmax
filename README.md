# VAE with Gumbel-Softmax

TensorFlow implementation of a Variational Autoencoder with Gumbel-Softmax Distribution. Refer to the following paper:

* [Categorical Reparametrization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf)
* [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/pdf/1611.00712.pdf)
* [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/pdf/1703.07370.pdf) 

Also, included is a jupyter notebook which shows how the Gumbel-Max trick for sampling discrete variables relates to Concrete distributions. 

## Table of Contents
* [Dataset](#dataset)
* [Installation](#installation)
* [Ananconda](#anaconda)
* [Docker](#docker)
* [IPython Notebook](#ipython-notebook)
* [Future Work](#future-work)


## Installation 

The program requires the following dependencies (easy to install using pip, Ananconda or Docker):

* python 2.7
* tensorflow (tested with r1.1.0)
* numpy 
* holoviews
* jupyter
* pandas
* matplotlib
* seaborn
* tqdm

## Anaconda

### Anaconda: Installation

To install DLFractalSequences in an Anaconda environment:

```python
conda env create -f environment.yml
```

To activate Anaconda environment:

```python
source activate dlfractals-env
```

### Anaconda: Train

Train VAE-Gumbel-Softmax model on the local machine using MNIST dataset:

```python
python vae_gumbel_softmax.py
```
