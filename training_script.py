import os
from nesy.data import process_mindreader, increase_sparsity
import pickle
from nesy.loaders import DataLoader
from nesy.training import mf_training, nesy_training
from nesy.utils import generate_report_dict, generate_report_table
import json
from joblib import Parallel, delayed
from nesy.configs import best_config_nesy_100k, best_config_nesy_200k, best_config_mf_100k, best_config_mf_200k, \
    best_config_genres_100k, best_config_genres_200k


def create_dataset(seed, percentages, data_path, version):
    """
    It creates the datasets for performing the experiment of the paper.

    :param seed: seed for the creation of the datasets
    :param percentages: list of percentages of training ratings to include in the folds. A fold will be created for
    each percentage included in this list. In the paper, 1.00, 0.5, 0.2, 0.1, and 0.05 percentages have been used.
    :param data_path: path where the dataset has to be saved
    :param version: version of the dataset ('mindreader-100k' or 'mindreader-200k')
    :return:
    """
    # create dataset only if it does not already exist in folder
    if not os.path.exists(os.path.join(data_path, str(seed))):
        print("Creating dataset with seed %d" % (seed,))
        data = process_mindreader(seed, movie_user_level_split=True, genre_user_level_split=True, movie_val_size=0.1,
                                  movie_test_size=0.2, genre_val_size=0.2, version=version)
        data["p"] = {}
        # create the folds with the given percentages
        for p in percentages:
            data["p"][p] = increase_sparsity(data["i_tr_small"], p, seed, user_level=False)
        # save dataset to disk
        with open(os.path.join(data_path, str(seed)), 'wb') as dataset_file:
            pickle.dump(data, dataset_file)


def train_model(seed, p, model, config_mf, config_nesy, source_config, exp_name):
    """
    It trains the given model with the correct hyper-parameter configuration on the given fold, denoted as p.
    If model == "nesy", the source config is used to pre-train an MF model on the source domain.
    After the training, the model is tested and a result file is produced and saved to disk.

    :param seed: seed for reproducibility
    :param p: it specifies the fold on which the model has to be trained
    :param model: str specifying the model that has to be trained
    :param config_mf: hyper-parameter configuration for the training of the baseline MF model
    :param config_nesy: hyper-parameter configuration for the training of the Neuro-Symbolic model
    :param source_config: hyper-parameter configuration for pre-training the MF model on the source domain when
    model == "nesy"
    :param exp_name: name of the experiment
    :return:
    """
    # prepare dataset for training
    file_name = "%s-movies-seed-%d-p-%.2f" % (model, seed, p)
    # train model only if a result file for the the model does not exist for the specified dataset percentage and seed
    if not os.path.exists("%s/results/%s.json" % (exp_name, file_name)):
        # pick correct dataset from dick
        with open("%s/datasets/%d" % (exp_name, seed), 'rb') as dataset_file:
            data = pickle.load(dataset_file)
        test_loader = DataLoader(data["i_test"], 256)

        if model == 'MF':
            # train the standard MF model
            _, test_result = mf_training(seed, data["p"][p], data["i_val"], data["n_users"], data["n_items"],
                                         config_mf, metric="fbeta-1.0", test_loader=test_loader,
                                         path="%s/models/%s.pth" % (exp_name, file_name))
        else:
            # pre-train an MF model on the source domain
            user_genres_matrix, _ = mf_training(seed, data["g_tr"], data["g_val"], data["n_users"], data["n_genres"],
                                                source_config, metric="fbeta-0.5", get_genre_matrix=True,
                                                path="%s/models/%s.pth" % (exp_name, "source-seed-%d" %
                                                                           (seed,)))
            # train the Neuro-Symbolic model
            test_result = nesy_training(seed, data["p"][p], data["i_val"], data["n_users"], data["n_items"],
                                        data["i_g_matrix"], user_genres_matrix, config_nesy,
                                        metric="fbeta-1.0", test_loader=test_loader,
                                        path="%s/models/%s.pth" % (exp_name, file_name))
        # save result to disk
        with open("%s/results/%s.json" % (exp_name, file_name), "w") as outfile:
            json.dump(test_result, outfile, indent=4)


def run_experiment(exp_name, config_dict, percentages=(1.00,), models=('MF', 'nesy'), starting_seed=0, n_runs=1,
                   dataset_version="mindreader-200k", n_jobs=os.cpu_count()):
    """
    This function runs the entire experiment for the paper. The experiment consists of the following steps:
    1. dataset creation: it includes the creation of the datasets with different sparsity levels (specified
    through `percentages` parameter)
    2. model training: this is done for both the baseline MF and the Neuro-Symbolic approach
    3. model testing

    This procedure is repeated for 'n_runs' runs with different seeds.
    At the end of the experiment, a LaTeX table containing the results is produced. The test metrics are averaged across
    the runs. The table reports precision, recall, and F1-measure for each model and dataset.

    :param exp_name: name of the experiment. A folder to save all the data associated with the experiment is created
    with this name on the root folder
    :param config_dict: best hyper-parameter configuration dictionary. It contains the best hyper-parameter
    configurations for all the models and for both the version of the dataset
    :param percentages: percentages of training ratings on the different datasets that have to be generated by
    the procedure
    :param models: list of model names. The procedure will be applied to these models
    :param starting_seed: the seed from which the experiments begin
    :param n_runs: number of times that the experiment has to be repeated
    :param dataset_version: version of the dataset on which the experiment has to be performed ('mindreader-100k' or
    'mindreader-200k')
    :param n_jobs: number of processors to be used for running the experiment
    :return:
    """
    # creating folders for the experiments
    if not os.path.exists(exp_name):
        os.mkdir(exp_name)
        os.mkdir(os.path.join(exp_name, "datasets"))
        os.mkdir(os.path.join(exp_name, "models"))
        os.mkdir(os.path.join(exp_name, "results"))

    # create datasets for the experiment
    Parallel(n_jobs=n_jobs)(
        delayed(create_dataset)(seed, percentages, os.path.join(exp_name, "datasets"), dataset_version)
        for seed in range(starting_seed, starting_seed + n_runs))

    # perform training of the models and get results on the test set
    Parallel(n_jobs=n_jobs)(
        delayed(train_model)(seed, p, m, config_dict[dataset_version]["mf"], config_dict[dataset_version]["nesy"],
                             config_dict[dataset_version]["source"], exp_name)
        for seed in range(starting_seed, starting_seed + n_runs)
        for p in percentages
        for m in models)

    # generate report table
    generate_report_dict(os.path.join(exp_name, "results"), ("neg_prec", "neg_rec", "neg_f"))
    table = generate_report_table(os.path.join(exp_name, "results.json"),
                                  {"neg_prec": "\\texttt{Precision}",
                                   "neg_rec": "\\texttt{Recall}",
                                   "neg_f": "\\texttt{F1-measure}"},
                                  {"MF": "$\\operatorname{MF}$",
                                   "nesy": "$\\operatorname{NESY_{MF}}$"}, models=models)
    # save table to disk
    with open(os.path.join(exp_name, "report-table.txt"), "w") as table_file:
        table_file.write(table)


if __name__ == "__main__":
    # this is a dictionary containing all the best hyper-parameter configurations for the models
    # they have been obtained by performing a Bayesian optimization on the first seed (i.e., 0) and just for the
    # dataset with the 100% of the ratings
    # note that a grid search has been performed also for the MF model pre-trained on the source domain
    # these configurations are used to train the models and get the results and final report table
    configs = {
        'mindreader-100k': {
            "source": best_config_genres_100k,
            "mf": best_config_mf_100k,
            "nesy": best_config_nesy_100k
        },
        'mindreader-200k': {
            "source": best_config_genres_200k,
            "mf": best_config_mf_200k,
            "nesy": best_config_nesy_200k
        }
    }
    # perform the experiments on MindReader-200k
    run_experiment("exp-mindreader-200k", config_dict=configs, percentages=(1.0, 0.5, 0.2, 0.1, 0.05),
                   models=('MF', 'nesy'), starting_seed=0, n_runs=30, dataset_version="mindreader-200k",
                   n_jobs=os.cpu_count())
    # perform the experiments on MindReader-100k
    run_experiment("exp-mindreader-100k", config_dict=configs, percentages=(1.0, 0.5, 0.2, 0.1, 0.05),
                   models=('MF', 'nesy'), starting_seed=0, n_runs=30, dataset_version="mindreader-100k",
                   n_jobs=os.cpu_count())
