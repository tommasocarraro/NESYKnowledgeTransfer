from nesy.loaders import LTNDataLoader, DataLoader, DataLoaderFM, LTNDataLoaderFM
from nesy.models import MatrixFactorization, MFTrainerClassifier, LTNTrainer, FocalLoss, FactorizationMachineModel, \
    FMTrainerClassifier, LTNTrainerFM
from torch.optim import Adam
import time
from nesy.utils import set_seed, compute_u_g_matrix


def nesy_training(seed, train_set, val_set, n_users, n_items, item_genres_matrix, user_genres_matrix, config, metric,
                  test_loader=None, path=None, time_exp=None):
    """
    It performs the training of NESY_MF on the given dataset with the given configuration of hyper-parameters. It can
    also be used for the training of LTN_MF as the loss is the same.

    If test_loader is not None, it tests the best model found on the validation set on the test set.

    :param seed: seed for reproducibility
    :param train_set: train set on which the training is performed
    :param val_set: validation set on which the model is evaluated
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param item_genres_matrix: item X genres matrix to understand at which genres each movie belongs to
    :param user_genres_matrix: matrix containing the pre-trained preferences of users for the genres (matrix G)
    :param config: dictionary containing the configuration of hyper-parameters for training the model
    :param metric: metric used to validate and test the model
    :param test_loader: test loader to test the performance of the model on the test set. Defaults to None
    :param path: path where to save the best model during training
    :param time_exp: dictionary containing hyper-parameter setup for time experiment
    """
    # set seed for reproducibility
    set_seed(seed)
    # create loaders for training and validating
    train_loader = LTNDataLoader(train_set, n_users=n_users, n_items=n_items,
                                 batch_size=time_exp["tr_batch_size"]
                                 if time_exp is not None else config["tr_batch_size"])
    val_loader = DataLoader(val_set, 256)
    # create model, optimizer, and trainer
    mf = MatrixFactorization(n_users, n_items, time_exp["k"] if time_exp is not None else config["k"], normalize=True)
    optimizer = Adam(mf.parameters(), lr=time_exp["lr"] if time_exp is not None else config["lr"],
                     weight_decay=config["wd"])
    trainer = LTNTrainer(mf, optimizer, u_g_matrix=user_genres_matrix, i_g_matrix=item_genres_matrix,
                         p_sat_agg=config["p_sat_agg"], p_forall=config["p_forall"], p_exists=config["p_exists"],
                         wandb_train=False, p_pos=config["p_pos"], p_neg=config["p_neg"], threshold=0.5)
    # train the model
    b = time.time()
    trainer.train(train_loader, val_loader, metric, n_epochs=time_exp["n_epochs"] if time_exp is not None else 1000,
                  early=time_exp["early"] if time_exp is not None else 10, verbose=1, save_path=path)
    e = time.time()
    t = e - b

    # test the model
    if test_loader is not None:
        trainer.load_model(path)
        return trainer.test(test_loader), t


def nesy_training_fm(seed, train_set, genres_tr, val_set, genres_val, genres_all, n_users, n_items, n_genres,
                     n_side_feat, item_genres_matrix, user_genres_matrix, config, metric, test_loader=None,
                     path=None, time_exp=None):
    """
    It performs the training of the NESY_FM approach on the given dataset with the given configuration of
    hyper-parameters. It has to be used for the training of LTN_FM too.

    If test_loader is not None, it tests the best model found on the validation set on the test set.

    :param seed: seed for reproducibility
    :param train_set: train set on which the training is performed
    :param genres_tr: tensor containing an ordinal encoding vector for each user-item interaction in the training set.
    This ordinal encoding contains the indexes of the movie genres related to the movie of the user-item interaction.
    This vector along with the user-item pair, form an input example for the FM model.
    :param val_set: validation set on which the model is evaluated
    :param genres_val: tensor containing an ordinal encoding vector for each user-item interaction in the validation
    set.
    :param genres_all: tensor containing an ordinal encoding vector for each item in the dataset
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param n_genres: number of movie genres in the dataset. Remind to pass n_genres + 1 as one additional genre index is
    used for padding tensors for efficiency
    :param n_side_feat: number of side information features in input. In our case, as each movie belongs to a maximum of
    12 movie genres, we have 12 side information features for each example (ordinal encoding of the movie genres). For
    the movies with a smaller number of movie genres, a fake genre is used to pad the side information feature vector.
    This is done to make everything efficient in batch computation. For this reason, one has to pass n_genres + 1 in the
    n_genres parameter.
    :param item_genres_matrix: item X genres matrix to understand at which genres each movie belongs to
    :param user_genres_matrix: matrix containing the pre-trained preferences of users for the genres (matrix G)
    :param config: dictionary containing the configuration of hyper-parameters for training the model
    :param metric: metric used to validate and test the model
    :param test_loader: test loader to test the performance of the model on the test set. Defaults to None
    :param path: path where to save the best model during training
    :param time_exp: dictionary containing hyper-parameter setup for time experiment
    """
    # set seed for reproducibility
    set_seed(seed)
    # create loaders for training and validating
    train_loader = LTNDataLoaderFM(train_set, genres_tr, genres_all, n_users=n_users, n_items=n_items,
                                   batch_size=time_exp["tr_batch_size"]
                                   if time_exp is not None else config["tr_batch_size"])
    val_loader = DataLoaderFM(val_set, genres_val, 256)
    # create model, optimizer, and trainer
    fm = FactorizationMachineModel([n_users, n_items, n_genres + 1],
                                   time_exp["k"] if time_exp is not None else config["k"],
                                   n_side_feat, normalize=True)
    optimizer = Adam(fm.parameters(), lr=time_exp["lr"] if time_exp is not None else config["lr"],
                     weight_decay=config["wd"])
    trainer = LTNTrainerFM(fm, optimizer, u_g_matrix=user_genres_matrix, i_g_matrix=item_genres_matrix,
                           p_sat_agg=config["p_sat_agg"], p_forall=config["p_forall"], p_exists=config["p_exists"],
                           wandb_train=False, p_pos=config["p_pos"], p_neg=config["p_neg"], threshold=0.5)
    # train the model
    b = time.time()
    trainer.train(train_loader, val_loader, metric, n_epochs=time_exp["n_epochs"] if time_exp is not None else 1000,
                  early=time_exp["early"] if time_exp is not None else 10, verbose=1, save_path=path)
    e = time.time()
    t = e - b

    # test the model
    if test_loader is not None:
        trainer.load_model(path)
        return trainer.test(test_loader), t


def mf_training(seed, train_set, val_set, n_users, n_items, config, metric, path, get_genre_matrix=False,
                test_loader=None, time_exp=None):
    """
    It performs the training of the MF model. It could be used for both the MF model trained on the source domain and
    the baseline MF model. This can also be used for the training of MF_genres.

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
    :param time_exp: dictionary containing hyper-parameter setup for time experiment
    """
    # set seed for reproducibility
    set_seed(seed)
    # create train and validation loaders
    train_loader = DataLoader(train_set, time_exp["tr_batch_size"] if time_exp is not None else config["tr_batch_size"])
    val_loader = DataLoader(val_set, 256)
    # create model, optimizer, and trainer
    mf = MatrixFactorization(n_users, n_items, time_exp["k"] if time_exp is not None else config["k"])
    optimizer = Adam(mf.parameters(), lr=time_exp["lr"] if time_exp is not None else config["lr"],
                     weight_decay=config["wd"])
    trainer = MFTrainerClassifier(mf, optimizer, FocalLoss(alpha=config["alpha"], gamma=config["gamma"]),
                                  threshold=config["threshold"])
    # train the model
    b = time.time()
    trainer.train(train_loader, val_loader, metric, n_epochs=time_exp["n_epochs"] if time_exp is not None else 1000,
                  early=time_exp["early"] if time_exp is not None else 10, verbose=1, save_path=path)
    e = time.time()
    t = e - b
    trainer.load_model(path)
    u_g_matrix, test_metric = None, None
    # compute and return matrix G, if requested
    if get_genre_matrix:
        u_g_matrix = compute_u_g_matrix(mf, config["threshold"])
    # get test performance, if test loader is given
    if test_loader is not None:
        test_metric = trainer.test(test_loader)
    return u_g_matrix, test_metric, t


def fm_training(seed, train_set, genres_tr, val_set, genres_val, n_users, n_items, n_genres, n_side_feat, config,
                metric, path, test_loader=None, time_exp=None):
    """
    It performs the training of the Factorization Machine model. This model uses movie genres as side content
    information. Note this model does not use user-genre ratings for training.

    If test_loader is not None, it tests the best model found on the validation set on the test set.

    :param seed: seed for reproducibility
    :param train_set: train set on which the training is performed
    :param genres_tr: tensor containing an ordinal encoding vector for each user-item interaction in the training set.
    This ordinal encoding contains the indexes of the movie genres related to the movie of the user-item interaction.
    This vector along with the user-item pair, form an input example for the FM model.
    :param val_set: validation set on which the model is evaluated
    :param genres_val: tensor containing an ordinal encoding vector for each user-item interaction in the validation
    set.
    :param n_users: number of users in the dataset
    :param n_items: number of items in the dataset
    :param n_genres: number of movie genres in the dataset. Remind to pass n_genres + 1 as one additional genre index is
    used for padding tensors for efficiency
    :param n_side_feat: number of side information features in input. In our case, as each movie belongs to a maximum of
    12 movie genres, we have 12 side information features for each example (ordinal encoding of the movie genres). For
    the movies with a smaller number of movie genres, a fake genre is used to pad the side information feature vector.
    This is done to make everything efficient in batch computation. For this reason, one has to pass n_genres + 1 in the
    n_genres parameter.
    :param config: configuration dictionary containing the hyper-parameter values to train the model
    :param metric: metric used to validate and test the model
    :param test_loader: test loader to test the performance of the model on the test set of the dataset. Defaults to
    None, meaning that the test phase is not performed
    :param path: path where to save the model every time a new best validation score is reached
    :param time_exp: dictionary containing hyper-parameter setup for time experiment
    """
    # set seed for reproducibility
    set_seed(seed)
    # create train and validation loaders
    train_loader = DataLoaderFM(train_set, genres_tr,
                                time_exp["tr_batch_size"] if time_exp is not None else config["tr_batch_size"])
    val_loader = DataLoaderFM(val_set, genres_val, 256)
    # create model, optimizer, and trainer
    fm = FactorizationMachineModel([n_users, n_items, n_genres + 1],
                                   time_exp["k"] if time_exp is not None else config["k"], n_side_feat)
    optimizer = Adam(fm.parameters(), lr=time_exp["lr"] if time_exp is not None else config["lr"],
                     weight_decay=config["wd"])
    trainer = FMTrainerClassifier(fm, optimizer, FocalLoss(alpha=config["alpha"], gamma=config["gamma"]),
                                  threshold=config["threshold"])
    # train the model
    b = time.time()
    trainer.train(train_loader, val_loader, metric, n_epochs=time_exp["n_epochs"] if time_exp is not None else 1000,
                  early=time_exp["early"] if time_exp is not None else 10, verbose=1, save_path=path)
    e = time.time()
    t = e - b
    trainer.load_model(path)
    test_metric = None
    # get test performance, if test loader is given
    if test_loader is not None:
        test_metric = trainer.test(test_loader)
    return test_metric, t
