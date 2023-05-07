from nesy.loaders import DataLoader, LTNDataLoader
from nesy.models import MatrixFactorization, MFTrainerClassifier, LTNTrainer, FocalLoss
from torch.optim import Adam
import wandb
from nesy.utils import set_seed


def mf_tuning(seed, tune_config, train_set, val_set, n_users, n_items, metric, exp_name=None, sweep_id=None):
    """
    It performs the hyper-parameter tuning of the MF model using the given hyper-parameter search configuration,
    training and validation set. It can be used for both the MF model trained on the source domain and the baseline MF.

    :param seed: seed for reproducibility
    :param tune_config: configuration for the tuning of hyper-parameters
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param metric: validation metric that has to be used
    :param exp_name: name of experiment. It is used to log data to the corresponding WandB project
    :param sweep_id: sweep id if ones wants to continue a WandB that got blocked
    """
    # create loader for validation
    val_loader = DataLoader(val_set, 256)

    # define function to call for performing one run of the hyper-parameter search

    def tune():
        with wandb.init():
            # get one random configuration
            k = wandb.config.k
            lr = wandb.config.lr
            wd = wandb.config.wd
            tr_batch_size = wandb.config.tr_batch_size
            threshold = wandb.config.threshold
            alpha = wandb.config.alpha
            gamma = wandb.config.gamma
            # define loader, model, optimizer and trainer
            train_loader = DataLoader(train_set, tr_batch_size)
            mf = MatrixFactorization(n_users, n_items, k)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = MFTrainerClassifier(mf, optimizer, loss=FocalLoss(alpha, gamma), threshold=threshold,
                                          wandb_train=True)
            # perform training
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep for 150 runs
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=tune_config, project=exp_name)
    set_seed(seed)
    wandb.agent(sweep_id, function=tune, count=150, project=exp_name)


def nesy_tuning(seed, tune_config, train_set, val_set, n_users, n_items, item_genres_matrix, user_genres_matrix,
                metric, sweep_id=None, exp_name=None):
    """
    It performs the hyper-parameter tuning of the Neuro-Symbolic model.

    :param seed: seed for reproducibility
    :param tune_config: configuration for the hyper-parameter tuning
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param item_genres_matrix: item X genre matrix containing 1 if movie i belongs to genre j
    :param user_genres_matrix: user X genre matrix containing the predictions of the MF model trained on the source
    domain (it is matrix G)
    :param sweep_id: id of WandB sweep, to be used if a sweep has been interrupted and it is needed to continue it
    :param
    """
    # create loader for validation
    val_loader = DataLoader(val_set, 256)

    # define function to call for performing one run of the hyper-parameter search
    def tune():
        with wandb.init():
            # sample one random configuration
            tr_batch_size = wandb.config.tr_batch_size
            k = wandb.config.k
            lr = wandb.config.lr
            wd = wandb.config.wd
            p_pos = wandb.config.p_pos
            p_neg = wandb.config.p_neg
            p_sat_agg = wandb.config.p_sat_agg
            p_forall = wandb.config.p_forall
            p_exists = wandb.config.p_exists
            # define loader, model, optimizer and trainer
            train_loader = LTNDataLoader(train_set, n_users=n_users, n_items=n_items, batch_size=tr_batch_size)
            mf = MatrixFactorization(n_users, n_items, k, normalize=True)
            optimizer = Adam(mf.parameters(), lr=lr, weight_decay=wd)
            trainer = LTNTrainer(mf, optimizer, user_genres_matrix, item_genres_matrix, p_pos=p_pos, p_neg=p_neg,
                                 p_sat_agg=p_sat_agg, p_forall=p_forall, p_exists=p_exists, wandb_train=True,
                                 threshold=0.5)
            trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1)

    # launch the WandB sweep
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=tune_config, project=exp_name)
    set_seed(seed)
    wandb.agent(sweep_id, function=tune, count=150, project=exp_name)
