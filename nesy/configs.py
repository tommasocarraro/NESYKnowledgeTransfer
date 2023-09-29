p_values = list(range(2, 12, 2))
k_values = [5, 10, 25, 50]
wd_values = [0.00005, 0.0001, 0.001, 0.01]
bs_values = [64, 128, 256]
lr_values = [0.0001, 0.001, 0.01]


# CONFIGURATIONS FOR HYPER-PARAMETER TUNING
# hyper-parameter search configuration for the baseline MF model
SWEEP_CONFIG_MF = {
    'name': "mf",
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'fbeta-1.0'},
    'parameters': {
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values},
        'threshold': {"value": 0.5},
        'alpha': {"values": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]},
        'gamma': {"values": [0, 1, 2, 3]}
    }
}
# hyper-parameter search configuration for the MF model trained in the source domain
SWEEP_CONFIG_MF_SOURCE = {
    'name': 'source',
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'fbeta-0.5'},
    'parameters': {
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values},
        'threshold': {"values": [0.3, 0.4, 0.5, 0.6, 0.7]},
        'alpha': {"values": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]},
        'gamma': {"values": [0, 1, 2, 3]}
    }
}
# hyper-parameter search configuration for the Neuro-Symbolic approach
SWEEP_CONFIG_NESY = {
    'name': 'nesy',
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'fbeta-1.0'},
    'parameters': {
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values},
        'p_forall': {"values": p_values},
        'p_exists': {"values": p_values},
        'p_pos': {"values": p_values},
        'p_neg': {"values": p_values},
        'p_sat_agg': {"values": p_values}
    }
}

# BEST CONFIGURATION FOUND USING BAYESIAN OPTIMIZATION
# Configurations for MindReader-200k
# Best configuration for the MF model trained on the source domain
best_config_genres_200k = {
    "alpha": 0.3,
    "gamma": 1,
    "k": 25,
    "lr": 0.01,
    "tr_batch_size": 64,
    "wd": 0.00005,
    "threshold": 0.4
}
# Best configuration for the Neuro-Symbolic approach
best_config_nesy_200k = {
    "k": 50,
    "lr": 0.0001,
    "wd": 0.00005,
    "p_exists": 10,
    "p_forall": 10,
    "p_neg": 8,
    "p_pos": 6,
    "p_sat_agg": 2,
    "tr_batch_size": 128
}
# Best configuration for the baseline MF model
best_config_mf_200k = {
    "alpha": 0.3,
    "gamma": 0,
    "k": 50,
    "lr": 0.01,
    "mse_loss": 0,
    "threshold": 0.5,
    "tr_batch_size": 256,
    "wd": 0.00005
}
# Best configuration for the baseline MF model with genre latent factors
best_config_mf_genres_200k = {
    "alpha": 0.3,
    "gamma": 0,
    "k": 25,
    "lr": 0.01,
    "tr_batch_size": 256,
    "wd": 0.00005,
    "threshold": 0.5
}
# Best configuration for the baseline FM model with movie genres as side info
best_config_fm_200k = {
    "alpha": 0.3,
    "gamma": 2,
    "k": 25,
    "lr": 0.001,
    "tr_batch_size": 128,
    "wd": 0.00005,
    "threshold": 0.5
}
# Best configuration for the baseline LTN model from "Logic Tensor Networks for Top-N Recommendation
best_config_ltn_200k = {
    "k": 25,
    "lr": 0.001,
    "wd": 0.00005,
    "p_exists": 8,
    "p_forall": 6,
    "p_neg": 4,
    "p_pos": 4,
    "p_sat_agg": 8,
    "tr_batch_size": 128
}
# Configurations for MindReader-100k
# Best configuration for the baseline MF model
best_config_mf_100k = {
    "alpha": 0.3,
    "gamma": 2,
    "k": 50,
    "lr": 0.01,
    "tr_batch_size": 256,
    "wd": 0.00005,
    "threshold": 0.5
}
# Best configuration for the baseline MF model with genre latent factors
best_config_mf_genres_100k = {
    "alpha": 0.3,
    "gamma": 0,
    "k": 25,
    "lr": 0.001,
    "tr_batch_size": 64,
    "wd": 0.00005,
    "threshold": 0.5
}
# Best configuration for the Neuro-Symbolic approach
best_config_nesy_100k = {
    "k": 50,
    "lr": 0.001,
    "wd": 0.0001,
    "p_exists": 10,
    "p_forall": 10,
    "p_neg": 10,
    "p_pos": 8,
    "p_sat_agg": 2,
    "tr_batch_size": 256
}
# Best configuration for the MF model trained on the source domain
best_config_genres_100k = {
    "alpha": 0.4,
    "gamma": 1,
    "k": 50,
    "lr": 0.01,
    "tr_batch_size": 128,
    "wd": 0.0001,
    "threshold": 0.5
}
# Best configuration for the baseline FM model with movie genres as side info
best_config_fm_100k = {
    "alpha": 0.3,
    "gamma": 3,
    "k": 50,
    "lr": 0.001,
    "tr_batch_size": 256,
    "wd": 0.0001,
    "threshold": 0.5
}
# Best configuration for the baseline LTN model from "Logic Tensor Networks for Top-N Recommendation
best_config_ltn_100k = {
    "k": 25,
    "lr": 0.001,
    "wd": 0.0001,
    "p_exists": 10,
    "p_forall": 8,
    "p_neg": 2,
    "p_pos": 2,
    "p_sat_agg": 2,
    "tr_batch_size": 128
}
