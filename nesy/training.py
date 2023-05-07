from nesy.loaders import LTNDataLoader, DataLoader
from nesy.models import MatrixFactorization, MFTrainerClassifier, LTNTrainer, FocalLoss
from torch.optim import Adam
from nesy.utils import set_seed, compute_u_g_matrix


def nesy_training(seed, train_set, val_set, n_users, n_items, item_genres_matrix, user_genres_matrix, config, metric,
                  test_loader=None, path=None):
    """
    It performs the training of the Neuro-Symbolic approach on the given dataset with the given configuration of
    hyper-parameters.

    If test_loader is not None, it tests the best model found on the validation set on the test set.

    :param seed: seed for reproducibility
    :param train_set: train set on which the training is performed
    :param val_set: validation set on which the model is evaluated
    :param user_genres_matrix: matrix containing the pre-trained preferences of users for the genres (matrix G)
    :param config: dictionary containing the configuration of hyper-parameters for training the model
    :param metric: metric used to validate and test the model
    :param test_loader: test loader to test the performance of the model on the test set. Defaults to None
    :param path: path where to save the best model during training
    """
    # set seed for reproducibility
    set_seed(seed)
    # create loaders for training and validating
    train_loader = LTNDataLoader(train_set, n_users=n_users, n_items=n_items, batch_size=config["tr_batch_size"])
    val_loader = DataLoader(val_set, 256)
    # create model, optimizer, and trainer
    mf = MatrixFactorization(n_users, n_items, config["k"], normalize=True)
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
    trainer = LTNTrainer(mf, optimizer, u_g_matrix=user_genres_matrix, i_g_matrix=item_genres_matrix,
                         p_sat_agg=config["p_sat_agg"], p_forall=config["p_forall"], p_exists=config["p_exists"],
                         wandb_train=False, p_pos=config["p_pos"], p_neg=config["p_neg"], threshold=0.5)
    # train the model
    trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1, save_path=path)
    # test the model
    if test_loader is not None:
        trainer.load_model(path)
        return trainer.test(test_loader)


def mf_training(seed, train_set, val_set, n_users, n_items, config, metric, path, get_genre_matrix=False,
                test_loader=None):
    """
    It performs the training of the MF model. It could be used for both the MF model trained on the source domain and
    the baseline MF model.

    If test_loader is not None, it tests the best model found on the validation set on the test set.

    If get_genre_matrix is True, the function will compute the user-genres matrix G using the predictions of the model.

    :param seed: seed for reproducibility
    :param train_set: train set on which the tuning is performed
    :param val_set: validation set on which the tuning is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param config: configuration dictionary containing the hyper-parameter values to train the model
    :param metric: metric used to validate and test the model
    :param get_genre_matrix: whether a user x genres pre-trained matrix has to be returned or not
    :param test_loader: test loader to test the performance of the model on the test set of the dataset. Defaults to
    None, meaning that the test phase is not performed
    :param path: path where to save the model every time a new best validation score is reached
    :param
    """
    # set seed for reproducibility
    set_seed(seed)
    # create train and validation loaders
    train_loader = DataLoader(train_set, config["tr_batch_size"])
    val_loader = DataLoader(val_set, 256)
    # create model, optimizer, and trainer
    mf = MatrixFactorization(n_users, n_items, config["k"])
    optimizer = Adam(mf.parameters(), lr=config["lr"], weight_decay=config["wd"])
    trainer = MFTrainerClassifier(mf, optimizer, FocalLoss(alpha=config["alpha"], gamma=config["gamma"]),
                                  threshold=config["threshold"])
    # train the model
    trainer.train(train_loader, val_loader, metric, n_epochs=1000, early=10, verbose=1, save_path=path)
    trainer.load_model(path)
    u_g_matrix, test_metric = None, None
    # compute and return matrix G, if requested
    if get_genre_matrix:
        u_g_matrix = compute_u_g_matrix(mf, config["threshold"])
    # get test performance, if test loader is given
    if test_loader is not None:
        test_metric = trainer.test(test_loader)
    return u_g_matrix, test_metric