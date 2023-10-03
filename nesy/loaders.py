import numpy as np
import torch
import ltn


class LTNDataLoader:
    """
    Data loader for the training of the LTN model with Matrix Factorization as recommendation model.

    It prepares the batches of positive, negative, and unknown user-item pairs to learn the LTN model.
    """
    def __init__(self,
                 data,
                 n_users,
                 n_items,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the data loader.

        :param data: list of training triples (user, item, rating)
        :param n_users: number of users in the dataset. It is used to randomly sample unknown user-item pairs
        :param n_items: number of items in the dataset. It is used to randomly sample unknown user-item pairs
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = np.array(data)
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            pos_ex = data[data[:, -1] == 1]
            neg_ex = data[data[:, -1] != 1]
            # get unknown user-item pairs - the probability of sampling real user-item pairs is very low due to the
            # sparsity of a recommendation dataset
            ukn_u_idx = np.random.choice(self.n_users, data.shape[0])
            ukn_i_idx = np.random.choice(self.n_items, data.shape[0])

            if len(pos_ex) and len(neg_ex):
                yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False), \
                      ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False), \
                      ltn.Variable('users_ukn', torch.tensor(ukn_u_idx), add_batch_dim=False), \
                      ltn.Variable('items_ukn', torch.tensor(ukn_i_idx), add_batch_dim=False)
            elif len(pos_ex):
                yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False), None, None, \
                      ltn.Variable('users_ukn', torch.tensor(ukn_u_idx), add_batch_dim=False), \
                      ltn.Variable('items_ukn', torch.tensor(ukn_i_idx), add_batch_dim=False)
            else:
                yield None, None, ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False), \
                      ltn.Variable('users_ukn', torch.tensor(ukn_u_idx), add_batch_dim=False), \
                      ltn.Variable('items_ukn', torch.tensor(ukn_i_idx), add_batch_dim=False)


class LTNDataLoaderFM:
    """
    Data loader for the training of the LTN model with Factorization Machine as recommendation model.

    It prepares the batches of positive, negative, and unknown user-item pairs to learn the LTN model.
    """
    def __init__(self,
                 data,
                 side_info,
                 movie_genres,
                 n_users,
                 n_items,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the data loader.

        :param data: list of training triples (user, item, rating)
        :param side_info: tensor of pre-computed side information for related user-item pairs (movie genre indexes in
        our scenario)
        :param movie_genres: tensor containing the movie genre indexes associated with each movie in the dataset
        :param n_users: number of users in the dataset. It is used to randomly sample unknown user-item pairs
        :param n_items: number of items in the dataset. It is used to randomly sample unknown user-item pairs
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = np.array(data)
        self.side_info = side_info
        self.movie_genres = movie_genres
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            side_info = self.side_info[idxlist[start_idx:end_idx]]
            pos_ex = data[data[:, -1] == 1]
            neg_ex = data[data[:, -1] != 1]
            pos_g = side_info[data[:, -1] == 1]
            neg_g = side_info[data[:, -1] != 1]
            # get unknown user-item pairs - the probability of sampling real user-item pairs is very low due to the
            # sparsity of a recommendation dataset
            ukn_u_idx = np.random.choice(self.n_users, data.shape[0])
            ukn_i_idx = np.random.choice(self.n_items, data.shape[0])
            ukn_g_side = self.movie_genres[ukn_i_idx]

            if len(pos_ex) and len(neg_ex):
                yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False), \
                      ltn.Variable('genres_pos', pos_g, add_batch_dim=False), \
                      ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False), \
                      ltn.Variable('genres_neg', neg_g, add_batch_dim=False), \
                      ltn.Variable('users_ukn', torch.tensor(ukn_u_idx), add_batch_dim=False), \
                      ltn.Variable('items_ukn', torch.tensor(ukn_i_idx), add_batch_dim=False), \
                      ltn.Variable('genres_ukn', ukn_g_side, add_batch_dim=False)
            elif len(pos_ex):
                yield ltn.Variable('users_pos', torch.tensor(pos_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_pos', torch.tensor(pos_ex[:, 1]), add_batch_dim=False), \
                      ltn.Variable('genres_pos', pos_g, add_batch_dim=False), None, None, None, \
                      ltn.Variable('users_ukn', torch.tensor(ukn_u_idx), add_batch_dim=False), \
                      ltn.Variable('items_ukn', torch.tensor(ukn_i_idx), add_batch_dim=False), \
                      ltn.Variable('genres_ukn', ukn_g_side, add_batch_dim=False)
            else:
                yield None, None, None, ltn.Variable('users_neg', torch.tensor(neg_ex[:, 0]), add_batch_dim=False), \
                      ltn.Variable('items_neg', torch.tensor(neg_ex[:, 1]), add_batch_dim=False), \
                      ltn.Variable('genres_neg', neg_g, add_batch_dim=False), \
                      ltn.Variable('users_ukn', torch.tensor(ukn_u_idx), add_batch_dim=False), \
                      ltn.Variable('items_ukn', torch.tensor(ukn_i_idx), add_batch_dim=False), \
                      ltn.Variable('genres_ukn', ukn_g_side, add_batch_dim=False)


class DataLoader:
    """
    Data loader for the training of the MF model and the evaluation of all the models based on MF.

    It prepares the batches of user-item pairs to learn the MF model or evaluate all the models based on MF.
    """
    def __init__(self,
                 data,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the data loader.

        :param data: list of training/evaluation triples (user, item, rating)
        :param batch_size: batch size for the training/evaluation of the model
        :param shuffle: whether to shuffle data during training/evaluation or not
        """
        self.data = np.array(data)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]

            yield torch.tensor(u_i_pairs), torch.tensor(ratings).float()


class DataLoaderFM:
    """
    Data loader for the training of the FM model. Note this model uses movie genres as side information. It has to be
    used also for the evaluation of all models based on FM.

    It prepares the batches of user-item pairs + side features to learn the FM model. Each example returned by this
    loader consists of a user index, an item index, and a set of movie genres indexes associated with the movie in
    the user-item pair.
    """
    def __init__(self,
                 data,
                 side_info,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the data loader.

        :param data: list of training/evaluation triples (user, item, rating)
        :param side_info: tensor of pre-computed side information for related user-item pairs (movie genre indexes in
        our scenario)
        :param batch_size: batch size for the training/evaluation of the model
        :param shuffle: whether to shuffle data during training/evaluation or not
        """
        self.data = np.array(data)
        self.side_info = side_info
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            side_info = self.side_info[idxlist[start_idx:end_idx]]
            u_i_pairs = data[:, :2]
            ratings = data[:, -1]

            yield torch.hstack([torch.tensor(u_i_pairs), side_info]), torch.tensor(ratings).float()
