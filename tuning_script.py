from nesy.tuning import mf_tuning, nesy_tuning, fm_tuning
from nesy.configs import SWEEP_CONFIG_NESY, SWEEP_CONFIG_MF, SWEEP_CONFIG_MF_SOURCE
from nesy.data import process_mindreader
from nesy.configs import best_config_genres_100k, best_config_genres_200k
import numpy as np
import os
from nesy.training import mf_training


def tune_model(model, seed, data, exp_name, folder_name, sweep_id=None):
    """
    It performs the hyper-parameter tuning of the given model on the given data. Based on the model, the function
    uses the right hyper-parameter search configuration.
    It creates a WandB project with name `exp_name` and logs the sweep data there.

    :param model: str indicating the name of the model ('source', 'mf', 'nesy')
    :param seed: seed for reproducibility
    :param data: dataset on which the hyper-parameter tuning has to be performed
    :param exp_name: name of the WandB project where to log the sweep data
    :param folder_name: folder where to save temporary files for tuning
    :param sweep_id: sweep identifier. It has to be used if the gets interrupted and one wants to recover it
    :return:
    """
    if model == "source":
        mf_tuning(seed, SWEEP_CONFIG_MF_SOURCE, data["g_tr"], data["g_val"], data["n_users"], data["n_genres"],
                  'fbeta-0.5', exp_name, sweep_id)
    if model == "mf":
        mf_tuning(seed, SWEEP_CONFIG_MF, data["i_tr_small"], data["i_val"], data["n_users"], data["n_items"],
                  'fbeta-1.0', exp_name, sweep_id)
    if model == "mf-genres":
        mf_tuning(seed, SWEEP_CONFIG_MF, np.append(data["i_tr_small"], data["mf_g_ratings"], axis=0), data["i_val"],
                  data["n_users"], data["n_items"] + data["n_genres"], 'fbeta-1.0', exp_name, sweep_id)
    if model == "fm":
        fm_tuning(seed, SWEEP_CONFIG_MF, data["i_tr_small"], data["genres_tr"], data["i_val"], data["genres_val"],
                  data["n_users"], data["n_items"], data["n_genres"], data["genres_tr"].shape[1], 'fbeta-1.0',
                  exp_name, sweep_id)
    if model == "nesy":
        # for simplicity, here we do not perform hyper-parameter tuning in the source domain
        # instead, we directly train an MF model on the source domain with best hyper-parameters saved in this repo
        # select parameters based on dataset
        if exp_name.split("-")[-1] == "100k":
            config = best_config_genres_100k
        else:
            config = best_config_genres_200k
        # train MF model on the source domain
        user_genres_matrix, _ = mf_training(seed, data["g_tr"], data["g_val"], data["n_users"], data["n_genres"],
                                            config, metric="fbeta-0.5", get_genre_matrix=True,
                                            path="%s/%s.pth" % (folder_name, "tuning_nesy-source-seed-%d" %
                                                                (seed,)))
        nesy_tuning(seed, SWEEP_CONFIG_NESY, data["i_tr_small"], data["i_val"], data["n_users"], data["n_items"],
                    data["i_g_matrix"], user_genres_matrix, 'fbeta-1.0', sweep_id, exp_name)
    if model == "ltn":
        nesy_tuning(seed, SWEEP_CONFIG_NESY, data["i_tr_small"], data["i_val"], data["n_users"], data["n_items"],
                    data["i_g_matrix"], data["u_g_matrix"], 'fbeta-1.0', sweep_id, exp_name)


if __name__ == "__main__":
    # create folders for tuning
    if not os.path.exists("tuning"):
        os.mkdir("tuning")
    # create MindReader-100k and MindReader-200k
    data_100k = process_mindreader(0, version='mindreader-100k')
    data_200k = process_mindreader(0, version='mindreader-200k')
    models = ('source', 'mf', 'mf-genres', 'fm', 'ltn', 'nesy')
    # hyper-parameter tuning for each model on both MindReader-100k and MindReader-200k
    for m in models:
        tune_model(m, 0, data_100k, 'tuning-mindreader-100k', "tuning")
        tune_model(m, 0, data_200k, 'tuning-mindreader-200k', "tuning")
