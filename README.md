# Mitigating Data Sparsity via Neuro-Symbolic Knowledge Transfer

This repository contains the implementation for the paper "Mitigating Data Sparsity via Neuro-Symbolic Knowledge Transfer" published at the 46th European Conference on Information Retrieval (ECIR'24). The entire code has been well documented to allow reviewers and researchers to understand the logic if needed.

### Directory structure

- `datasets`: this folder contains the two raw datasets used in the experiments, namely `MindReader-100k` and `MindReader-200k`. 
They are available at this [link](https://mindreader.tech/dataset/releases).
- `nesy`: it contains the implementation of the paper. This Python package is organized as follows:
  - `configs`: it contains the configurations to perform the Bayesian optimizations with [Weights and Biases](https://wandb.ai/site).
  It also contains the best hyper-parameter configurations found in our experiments for the all models and datasets.
  They have been included in the repository to allow reviewers to reproduce the code without waiting for the
  best hyper-parameters to be generated;
  - `data`: it contains functions to process and prepare the datasets for the training;
  - `loaders`: it contains data loaders used to train, evaluate, and test our models;
  - `metrics`: it contains the implementation of the evaluation metrics used in the paper. In particular, we used the `F-measure`;
  - `models`: it contains the implementation of our models, namely the LFM baselines, LTN-based baselines, and Neuro-Symbolic knowledge transfer baselines;
  - `training`: it contains some utility functions used to train the models;
  - `tuning`: it contains some utility functions used to perform the hyper-parameter tuning of the models;
  - `utils`: it contains some utility function used by the remainder of the code.
- `uploaded-exp`: this folder contains the results obtained from our experiments. We included them in case reviewers 
do not have time to execute the experiments and want to give a look at the results. If the experiments are run, they will lead 
to the same results.

### How to reproduce the experiments?

First of all, make sure to install all the packages included in `requirements.txt`. Then, leave the directory structure as given in this repository.

To reproduce our experiments, you just need to run the `training_script.py` script. 

Notice this script uses parallel computing
to make the experiments faster. If you want to avoid parallelism, you just need to change parameter
`n_jobs` of function `run_experiment` to 1. Note that if parallelism is used, the output on the console will be difficult to
read due to different threads logging at the same time. 

`training_script.py` does not make any hyper-parameter search for the models.
It uses the best hyper-parameter configurations uploaded to the repository. This allows to reproduce the results faster. 

This script
runs the complete experiment for both datasets and then saves the results in two separated folders, one for `MindReader-100k` and one for `MindReader-200k`, respectively. 
The results are saved as a JSON file and a txt file. The JSON file reports every test metric averaged across the 30 runs of the experiment. Results are subdivided by model and dataset fold.
The txt file contains the LaTeX report table included in the paper.

If one wants to perform some hyper-parameter tuning, it can run the `tuning_script.py` script. Notice you will have to
configure Weights and Biases properly first. You can follow [this](https://docs.wandb.ai/quickstart) step-by-step guide. 
The script will run the hyper-parameter tuning for all the models on the full datasets (i.e., 100% folds) and the first seed of execution (i.e., seed 0). 
In some cases, it is possible the hyper-parameter tuning gets interrupted by Weights and Biases. For this reason, we allow the user to 
set the sweep ID to restore the tuning from the point it got blocked. We apologize for that but the error is due to Weights and Biases servers.

All the information that is not written in this file can be found in the documentation of the implementation.
