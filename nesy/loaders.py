import numpy as np
import torch
import ltn


class LTNDataLoader:
    """
    Data loader for the training of the LTN model.

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


class DataLoader:
    """
    Data loader for the training of the MF model and the evaluation of all the models.

    It prepares the batches of user-item pairs to learn the MF model or evaluate all the models.
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
