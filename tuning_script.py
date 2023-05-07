from nesy.tuning import mf_tuning, nesy_tuning
from nesy.configs import SWEEP_CONFIG_NESY, SWEEP_CONFIG_MF, SWEEP_CONFIG_MF_SOURCE
from nesy.data import process_mindreader


def tune_model(model, seed, data, exp_name, sweep_id=None):
    """
    It performs the hyper-parameter tuning of the given model on the given data. Based on the model, the function
    uses the right hyper-parameter search configuration.
    It creates a WandB project with name `exp_name` and logs the sweep data there.

    :param model: str indicating the name of the model ('source', 'mf', 'nesy')
    :param seed: seed for reproducibility
    :param data: dataset on which the hyper-parameter tuning has to be performed
    :param exp_name: name of the WandB project where to log the sweep data
    :param sweep_id: sweep identifier. It has to be used if the gets interrupted and one wants to recover it
    :return:
    """
    if model == "source":
        mf_tuning(seed, SWEEP_CONFIG_MF_SOURCE, data["g_tr"], data["g_val"], data["n_users"], data["n_genres"],
                  'fbeta-0.5', exp_name, sweep_id)
    if model == "mf":
        mf_tuning(seed, SWEEP_CONFIG_MF, data["i_tr_small"], data["i_val"], data["n_users"], data["n_items"],
                  'fbeta-1.0', exp_name, sweep_id)
    if model == "nesy":
        nesy_tuning(seed, SWEEP_CONFIG_NESY, data["i_tr_small"], data["i_val"], data["n_users"], data["n_items"],
                    'fbeta-1.0', exp_name, sweep_id)


if __name__ == "__main__":
    # create MindReader-100k and MindReader-200k
    data_100k = process_mindreader(0, version='mindreader-100k')
    data_200k = process_mindreader(0, version='mindreader-200k')
    models = ('source', 'mf', 'nesy')
    # hyper-parameter tuning for each model on both MindReader-100k and MindReader-200k
    for m in models:
        tune_model(m, 0, data_100k, 'tuning-mindreader-100k')
        tune_model(m, 0, data_200k, 'tuning-mindreader-200k')
