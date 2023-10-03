import ltn
import torch
import numpy as np
import torch.nn.functional as F
import wandb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from nesy.metrics import compute_metric, check_metrics


class Trainer:
    """
    Abstract base class that manages the training of the model.

    Each model implementation must inherit from this class and implement the train_epoch() method. It is also possible
    to use overloading to redefine the behavior of some methods.
    """
    def __init__(self, model, optimizer, wandb_train=False):
        """
        Constructor of the trainer.

        :param model: neural model for which the training has to be performed
        :param optimizer: optimizer that has to be used for the training of the model
        :param wandb_train: whether to log data on Weights and Biases servers. This is used when using hyper-parameter
        optimization
        """
        self.model = model
        self.optimizer = optimizer
        self.wandb_train = wandb_train

    def train(self, train_loader, val_loader, val_metric=None, n_epochs=500, early=None, verbose=10, save_path=None):
        """
        Method for the train of the model.

        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 500
        :param early: patience for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details
        :param save_path: path where to save the best model, default to None
        """
        if val_metric is not None:
            check_metrics(val_metric)
        best_val_score = 0.0
        early_counter = 0
        if self.wandb_train:
            # log gradients and parameters with Weights and Biases
            wandb.watch(self.model, log="all")

        for epoch in range(n_epochs):
            # training step
            train_loss, log_dict = self.train_epoch(train_loader, epoch + 1)
            # validation step
            val_score = self.validate(val_loader, val_metric)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                log_record = "Epoch %d - Train loss %.3f - Val %s %.3f" % (epoch + 1, train_loss, val_metric, val_score)
                # add log_dict to log_record
                log_record += " - " + " - ".join(["%s %.3f" % (k, v) for k, v in log_dict.items() if k != "train_loss"])
                # print epoch report
                print(log_record)
                if self.wandb_train:
                    # log validation metric value
                    wandb.log({"smooth_%s" % (val_metric,): val_score})
                    # log training information
                    wandb.log(log_dict)
            # save best model and update early stop counter, if necessary
            if val_score > best_val_score:
                best_val_score = val_score
                if self.wandb_train:
                    # the metric is logged only when a new best value is achieved for it
                    wandb.log({"%s" % (val_metric,): val_score})
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    if save_path:
                        self.load_model(save_path)
                    break

    def train_epoch(self, train_loader, epoch=None):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :param epoch: index of epoch
        :return: training loss value averaged across training batches and a dictionary containing useful information
        to log, such as other metrics computed by this model
        """
        raise NotImplementedError()

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: the prediction of the model for the given user-item pair
        """
        u_idx, i_idx = x[:, 0], x[:, 1]
        with torch.no_grad():
            return self.model(u_idx, i_idx, dim)

    def prepare_for_evaluation(self, loader):
        """
        It prepares an array of predictions and targets for computing classification metrics.

        :param loader: loader containing evaluation data
        :return: predictions and targets
        """
        preds, targets = [], []
        for batch_idx, (X, y) in enumerate(loader):
            preds_ = self.predict(X)
            preds.append(preds_)
            targets.append(y.numpy())
        return np.concatenate(preds), np.concatenate(targets)

    def validate(self, val_loader, val_metric):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name (it is F0.5-score for source domain and F1-score for target domain)
        :return: validation score based on the given metric averaged across all validation examples
        """
        # prepare predictions and targets for evaluation
        preds, targets = self.prepare_for_evaluation(val_loader)
        # compute F-score
        val_score = compute_metric(val_metric, preds, targets)
        # compute precision and recall
        p, r, f, _ = precision_recall_fscore_support(targets, preds,
                                                     beta=float(val_metric.split("-")[1])
                                                     if "fbeta" in val_metric else 1.0, average=None)
        # compute other useful metrics used in classification tasks
        tn, fp, fn, tp = tuple(confusion_matrix(targets, preds).ravel())
        sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
        # log metrics to WandB servers
        if self.wandb_train:
            wandb.log({
                "neg_prec": p[0],
                "pos_prec": p[1],
                "neg_rec": r[0],
                "pos_rec": r[1],
                "neg_f": f[0],
                "pos_f": f[1],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "sensitivity": sensitivity,
                "specificity": specificity
            })

        return val_score

    def save_model(self, path):
        """
        Method for saving the model.

        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def test(self, test_loader):
        """
        Method for performing the test of the model based on the given test loader.

        The method computes precision, recall, F1-score, and other useful classification metrics.

        :param test_loader: data loader for test data
        :return: a dictionary containing the value of each metric average across the test examples
        """
        # create dictionary where the results have to be stored
        results = {}
        # prepare predictions and targets for evaluation
        preds, targets = self.prepare_for_evaluation(test_loader)
        # compute metrics
        results["fbeta-1.0"] = compute_metric("fbeta-1.0", preds, targets)
        p, r, f, _ = precision_recall_fscore_support(targets, preds, beta=1., average=None)
        results["neg_prec"] = p[0]
        results["pos_prec"] = p[1]
        results["neg_rec"] = r[0]
        results["pos_rec"] = r[1]
        results["neg_f"] = f[0]
        results["pos_f"] = f[1]
        results["tn"], results["fp"], results["fn"], results["tp"] = \
            (int(i) for i in tuple(confusion_matrix(targets, preds).ravel()))
        results["sensitivity"] = results["tp"] / (results["tp"] + results["fn"])
        results["specificity"] = results["tn"] / (results["tn"] + results["fp"])
        results["acc"] = accuracy_score(targets, preds)

        return results


class FocalLoss(torch.nn.Module):
    """
    Implementation of focal loss as in PyTorch this loss is not directly provided.
    """
    def __init__(self, alpha=0.2, gamma=2, reduction="mean"):
        """
        Constructor of the loss.

        :param alpha: hyper-parameter to give weights to the to classes (i.e., balance them)
        :param gamma: hyper-parameter to give a penalty to example that are hard to classify
        :param reduction: what type of reduction to use
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCELoss(reduction="none")
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: "
                f"'none', 'mean', 'sum'")
        return loss

# ---------- here, it starts the implementation of Factorization machine taken from pytorch-fm ----------------------


class FeaturesLinear(torch.nn.Module):
    """
    This layer computes the first-order interactions of the Factorization Machine model. It contains the biases of the
    model. There is one bias for each user, item, and movie genre.
    """
    def __init__(self, field_dims, n_side_feat, output_dim=1):
        """
        Constructor of the layer.

        :param field_dims: list of dimensions for the features used by the model. In our scenario, this list contains
        the number of users, items, and movie genres. The model constructs an embedding for each one of these features.
        :param n_side_feat: number of side information features. In our scenario, it is the maximum number of genres
        that a movie can belong to. The side information features are just the indexes of the movie genres. Each
        example is composed of a user index, an item index, and the indexes of movie genres associated with the item.
        :param output_dim: dimension of the output of the model.
        """
        super().__init__()
        self.field_dims = field_dims
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.offsets = np.append(self.offsets, [self.offsets[-1]] * (n_side_feat - 1))
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, dim=1):
        """
        In the retrieval of biases, a mask is used to mask the biases associated with the fake index added
        to the movie genre indexes to make operations more efficient.

        :param x: Long tensor of size ``(batch_size, num_fields)``. It represents the input examples composed of
        [u, i, gs], where u is a user index, i and item index, and gs a sequence of movie genre indexes.
        :param dim: dimension across which the operations have to be computed.
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        mask = torch.where(x == self.offsets[-1] + self.field_dims[-1] - 1, 0., 1.).unsqueeze_(2)
        return torch.sum(self.fc(x) * mask, dim=dim) + self.bias


class FeaturesEmbedding(torch.nn.Module):
    """
    This layer contains the embeddings to compute second-order interactions of the Factorization Machine model. There is
    an embedding for each user, item, and movie genre.
    """
    def __init__(self, field_dims, embed_dim, n_side_feat):
        """
        Constructor of the layer.

        :param field_dims: list of dimensions for the features used by the model. In our scenario, this list contains
        the number of users, items, and movie genres. The model constructs an embedding for each one of these features.
        :param n_side_feat: number of side information features. In our scenario, it is the maximum number of genres
        that a movie can belong to. The side information features are just the indexes of the movie genres. Each
        example is composed of a user index, an item index, and the indexes of movie genres associated with the item.
        :param embed_dim: dimension of embeddings in the model.
        :param n_side_feat: number of side information features in input at the model. This corresponds to the maximum
        number of genres a movie can belong to in out dataset.
        """
        super().__init__()
        self.field_dims = field_dims
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.offsets = np.append(self.offsets, [self.offsets[-1]] * (n_side_feat - 1))
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        In the retrieval of embeddings, a mask is used to mask the embeddings associated with the fake index added
        to the movie genre indexes to make operations more efficient.

        :param x: Long tensor of size ``(batch_size, num_fields)``. It represents the input examples composed of
        [u, i, gs], where u is a user index, i and item index, and gs a sequence of movie genre indexes.
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        mask = torch.where(x == self.offsets[-1] + self.field_dims[-1] - 1, 0., 1.).unsqueeze_(2)
        return self.embedding(x) * mask


class FactorizationMachine(torch.nn.Module):
    """
    This layer computes the second-order interactions of the Factorization Machine model.
    """
    def __init__(self, reduce_sum=True):
        """
        Constructor of the layer.

        :param reduce_sum: whether the output has to be a summation over examples. Defaults to True as it happens
        for Factorization Machines.
        """
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x, dim=1):
        """
        This implements the closed formula presented in the original FM paper to model second-order interactions.

        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``. It contains the embeddings of input
        users, items, and movie genres.
        :param dim: dimension across which the operations have to be computed.
        """
        square_of_sum = torch.sum(x, dim=dim) ** 2
        sum_of_square = torch.sum(x ** 2, dim=dim)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=dim, keepdim=True)
        return 0.5 * ix


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine inspired by the pytorch-fm github repository.
    This implementation uses ordinal encoding to make operations very efficient. However, it does not support
    multi-hot side information features. We adapted the implementation in such a way to account for this.

    Link to the original implementation: https://github.com/rixwew/pytorch-fm.

    Reference:
        S. Rendle, Factorization Machines, 2010.
    """
    def __init__(self, field_dims, embed_dim, n_side_feat, normalize=False):
        """
        Constructor of the Factorization Machine model.

        :param field_dims: list of dimensions for the features used by the model. In our scenario, this list contains
        the number of users, items, and movie genres. The model constructs an embedding for each one of these features.
        :param n_side_feat: number of side information features. In our scenario, it is the maximum number of genres
        that a movie can belong to. The side information features are just the indexes of the movie genres. Each
        example is composed of a user index, an item index, and the indexes of movie genres associated with the item.
        :param embed_dim: dimension of embeddings in the model.
        :param normalize: whether the predictions of the model have to be normalize between 0 and 1. Default to False.
        """
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, n_side_feat)
        self.linear = FeaturesLinear(field_dims, n_side_feat)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.normalize = normalize

    def forward(self, x, x2=None, x3=None, dim=1):
        """
        This function computes a prediction of the Factorization Machine model. It computes first-order and
        second-order interactions between features.

        :param x: Long tensor of size ``(batch_size, num_fields)``. It represents the input examples composed of
        [u, i, gs], where u is a user index, i and item index, and gs a sequence of movie genre indexes.
        :param x2: it is used when training the model with LTN. It contains item indexes
        :param x3: it is used when training the model with LTN. It contains genre indexes
        :param dim: dimension across which the operations have to be computed
        """
        if x2 is not None:
            x = torch.hstack([x.unsqueeze(1), x2.unsqueeze(1), x3])
        x = self.linear(x, dim=dim) + self.fm(self.embedding(x), dim=dim)
        if self.normalize:
            x = torch.sigmoid(x.squeeze(1))
        else:
            x = x.squeeze(1)
        return x


class FMTrainer(Trainer):
    """
    Basic trainer for training a Factorization Machine model using gradient descent.
    """
    def __init__(self, fm_model, optimizer, loss, wandb_train=False):
        """
        Constructor of the trainer for the MF model.

        :param fm_model: Factorization Machine model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param loss: loss that is used for training the Matrix Factorization model. It could be MSE or Focal Loss
        """
        super(FMTrainer, self).__init__(fm_model, optimizer, wandb_train)
        self.loss = loss

    def train_epoch(self, train_loader, epoch=None):
        train_loss = 0.0
        for batch_idx, (examples, ratings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            loss = self.loss(self.model(examples), ratings)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(train_loader), {"train_loss": train_loss / len(train_loader)}

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair + side info features for which the prediction has to be computed.
        The first position is the user index, the second position is the item index, while the last positions are
        dedicated to the indexes of movie genres associated with the movie in the user-item interaction.
        :param dim: dimension across which computations of the FM model have to be computed
        :return: the prediction of the model for the given example
        """
        with torch.no_grad():
            return self.model(x, dim=dim)


class FMTrainerClassifier(FMTrainer):
    """
    Trainer for training a Factorization Machine model for the binary classification task.

    This version of the trainer uses the focal loss as loss function. It implements a classification task, where the
    objective is to minimize the focal loss. The ratings are binary, hence they can be interpreted as binary classes.
    The objective is to discriminate between class 1 ("likes") and class 0 ("dislikes"). Note the focal loss is a
    generalization of the binary cross-entropy to deal with imbalance data.
    """
    def __init__(self, fm_model, optimizer, loss, wandb_train=False, threshold=0.5):
        """
        Constructor of the trainer for the MF model for binary classification.

        :param fm_model: Factorization Machine model
        :param optimizer: optimizer used for the training of the model
        :param loss: loss function used to train the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param threshold: threshold used to determine whether an example is negative or positive (decision boundary)
        """
        super(FMTrainerClassifier, self).__init__(fm_model, optimizer, loss, wandb_train)
        self.threshold = threshold

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair + side info features for which the prediction has to be computed.
        The first position is the user index, the second position is the item index, while the last positions are
        dedicated to the indexes of movie genres associated with the movie in the user-item interaction.
        :param dim: dimension across which computations of the FM model have to be computed
        :return: the prediction of the model for the given example
        """
        # apply sigmoid because during training the losses with logits are used for numerical stability
        preds = super().predict(x, dim)
        preds = torch.sigmoid(preds)
        preds = preds >= self.threshold
        return preds


# ------------------------ here, it ends the implementation of Factorization Machine ----------------------------


class MatrixFactorization(torch.nn.Module):
    """
    Vanilla Matrix factorization model.

    The model uses the inner product between a user and an item latent factor to get the prediction.

    The model has inside two matrices: one containing the embeddings of the users of the system, one containing the
    embeddings of the items of the system.
    The model has inside two vectors: one containing the biases of the users of the system, one containing the biases
    of the items of the system.
    """
    def __init__(self, n_users, n_items, n_factors, normalize=False):
        """
        Constructor of the matrix factorization model.

        :param n_users: number of users in the dataset
        :param n_items: number of items in the dataset
        :param n_factors: size of embeddings for users and items
        :param normalize: whether the output has to be normalized in [0.,1.] using sigmoid. This is used for the LTN
        model.
        """
        super(MatrixFactorization, self).__init__()
        self.u_emb = torch.nn.Embedding(n_users, n_factors)
        self.i_emb = torch.nn.Embedding(n_items, n_factors)
        self.u_bias = torch.nn.Embedding(n_users, 1)
        self.i_bias = torch.nn.Embedding(n_items, 1)
        self.normalize = normalize
        # initialization with Glorot
        torch.nn.init.xavier_normal_(self.u_emb.weight)
        torch.nn.init.xavier_normal_(self.i_emb.weight)
        torch.nn.init.xavier_normal_(self.u_bias.weight)
        torch.nn.init.xavier_normal_(self.i_bias.weight)

    def forward(self, u_idx, i_idx, dim=1):
        """
        It computes the scores for the given user-item pairs using inner product (dot product).

        :param u_idx: users for which the score has to be computed
        :param i_idx: items for which the score has to be computed
        :param dim: dimension along which the dot product has to be computed
        :return: predicted scores for given user-item pairs
        """
        pred = torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=dim, keepdim=True)
        # add user and item biases to the prediction
        pred += self.u_bias(u_idx) + self.i_bias(i_idx)
        pred = pred.squeeze()
        if self.normalize:
            pred = torch.sigmoid(pred)
        return pred


class MFTrainer(Trainer):
    """
    Basic trainer for training a Matrix Factorization model using gradient descent.
    """
    def __init__(self, mf_model, optimizer, loss, wandb_train=False):
        """
        Constructor of the trainer for the MF model.

        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param loss: loss that is used for training the Matrix Factorization model. It could be MSE or Focal Loss
        """
        super(MFTrainer, self).__init__(mf_model, optimizer, wandb_train)
        self.loss = loss

    def train_epoch(self, train_loader, epoch=None):
        train_loss = 0.0
        for batch_idx, (u_i_pairs, ratings) in enumerate(train_loader):
            self.optimizer.zero_grad()
            loss = self.loss(self.model(u_i_pairs[:, 0], u_i_pairs[:, 1]), ratings)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return train_loss / len(train_loader), {"train_loss": train_loss / len(train_loader)}


class MFTrainerClassifier(MFTrainer):
    """
    Trainer for training a Matrix Factorization model for the binary classification task.

    This version of the trainer uses the focal loss as loss function. It implements a classification task, where the
    objective is to minimize the focal loss. The ratings are binary, hence they can be interpreted as binary classes.
    The objective is to discriminate between class 1 ("likes") and class 0 ("dislikes"). Note the focal loss is a
    generalization of the binary cross-entropy to deal with imbalance data.
    """
    def __init__(self, mf_model, optimizer, loss, wandb_train=False, threshold=0.5):
        """
        Constructor of the trainer for the MF model for binary classification.

        :param mf_model: Matrix Factorization model
        :param optimizer: optimizer used for the training of the model
        :param loss: loss function used to train the model
        :param wandb_train: whether the data has to be logged to WandB or not
        :param threshold: threshold used to determine whether an example is negative or positive (decision boundary)
        """
        super(MFTrainerClassifier, self).__init__(mf_model, optimizer, loss, wandb_train)
        self.threshold = threshold

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: the prediction of the model for the given user-item pair
        """
        # apply sigmoid because during training the losses with logits are used for numerical stability
        preds = super().predict(x, dim)
        preds = torch.sigmoid(preds)
        preds = preds >= self.threshold
        return preds


class LTNTrainer(Trainer):
    """
    Trainer for the training of the Neuro-Symbolic approach with MF as the recommendation model. This approach uses
    axiomatic knowledge to transfer pre-trained information from the source domain to the target domain. Note that the
    axiom that transfer knowledge is added to the loss from epoch five. Notice also the axioms is applied to unknown
    user-item pairs.
    """
    def __init__(self, mf_model, optimizer, u_g_matrix, i_g_matrix, p_pos=2, p_neg=2, p_sat_agg=2, p_forall=2,
                 p_exists=2, wandb_train=False, threshold=0.5):
        """
        Constructor of the trainer for the Neuro-Symbolic model.

        :param mf_model: Matrix Factorization model to implement the Likes predicate
        :param optimizer: optimizer used for the training of the model
        :param u_g_matrix: users X genres matrix, containing the ratings predicted for each user-genre pair in the
        source domain. This is the pre-trained model. It is the G matrix in the paper
        :param i_g_matrix: items X genres matrix, containing a 1 if the item i belongs to genre j, 0 otherwise. It is
        used to implement the HasGenre predicate
        :param p_pos: hyper-parameter p for universal quantifier aggregator used on positive user-item pairs (Axiom 1)
        :param p_neg: hyper-parameter p for universal quantifier aggregator used on negative user-item pairs (Axiom 2)
        :param p_sat_agg: hyper-parameter p for universal quantifier aggregator used in the sat agg operator
        :param p_forall: hyper-parameter p for universal quantifier aggregator used on the formula that enables
        domain adaptation (Axiom 3)
        :param p_exists: hyper-parameter p for existential quantifier aggregator used on on the formula that enables
        domain adaptation (Axiom 3)
        :param wandb_train: whether the training information has to be logged on WandB or not
        :param threshold: threshold used for the decision boundary
        """
        super(LTNTrainer, self).__init__(mf_model, optimizer, wandb_train)
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.Likes = ltn.Predicate(mf_model)
        self.LikesGenre = ltn.Predicate(func=lambda u_idx, g_idx: u_g_matrix[u_idx, g_idx])
        self.genres = ltn.Variable("genres", torch.tensor(range(u_g_matrix.shape[1])), add_batch_dim=False)
        self.HasGenre = ltn.Predicate(func=lambda i_idx, g_idx: i_g_matrix[i_idx, g_idx])
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(), quantifier='e')
        self.SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=p_sat_agg))
        self.p_forall = p_forall
        self.p_exists = p_exists
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.threshold = threshold

    def train_epoch(self, train_loader, epoch=None):
        train_loss, train_sat_agg, f1_sat, f2_sat, f3_sat, f3 = 0.0, 0.0, [], [], [], None
        for batch_idx, (u_pos, i_pos, u_neg, i_neg, u_ukn, i_ukn) in enumerate(train_loader):
            self.optimizer.zero_grad()
            if epoch > 5:
                # compute SAT level of third formula
                f3 = self.Forall(ltn.diag(u_ukn, i_ukn),
                                 self.Implies(
                                     self.Exists(self.genres,
                                                 self.And(
                                                     self.Not(self.LikesGenre(u_ukn, self.genres)),
                                                     self.HasGenre(i_ukn, self.genres)
                                                 ), p=self.p_exists),
                                     self.Not(self.Likes(u_ukn, i_ukn))),
                                 p=self.p_forall)
                f3_sat.append(f3.value.item())
            if u_pos is not None and u_neg is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos), self.Likes(u_pos, i_pos), p=self.p_pos)
                f2 = self.Forall(ltn.diag(u_neg, i_neg), self.Not(self.Likes(u_neg, i_neg)), p=self.p_neg)
                f1_sat.append(f1.value.item())
                f2_sat.append(f2.value.item())
                train_sat = self.SatAgg(f1, f2, f3) if epoch > 5 else self.SatAgg(f1, f2)
            elif u_pos is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos), self.Likes(u_pos, i_pos), p=self.p_pos).value
                f1_sat.append(f1.item())
                train_sat = self.SatAgg(f1, f3) if epoch > 5 else f1
            else:
                f2 = self.Forall(ltn.diag(u_neg, i_neg), self.Not(self.Likes(u_neg, i_neg)), p=self.p_neg).value
                f2_sat.append(f2.item())
                train_sat = self.SatAgg(f2, f3) if epoch > 5 else f2
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader),
                                                "pos_sat": np.mean(f1_sat),
                                                "neg_sat": np.mean(f2_sat),
                                                "f3_sat": np.mean(f3_sat) if f3_sat else -1.}

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: the prediction of the model for the given user-item pair
        """
        preds = super().predict(x, dim)
        preds = preds >= self.threshold
        return preds


class LTNTrainerFM(Trainer):
    """
    Trainer for the training of the Neuro-Symbolic approach with the FM as the recommendation model. This approach uses
    axiomatic knowledge to transfer pre-trained information from the source domain to the target domain. Note that the
    axiom that transfer knowledge is added to the loss from epoch five. Notice also the axioms is applied to unknown user-item pairs.
    """
    def __init__(self, fm_model, optimizer, u_g_matrix, i_g_matrix, p_pos=2, p_neg=2, p_sat_agg=2, p_forall=2,
                 p_exists=2, wandb_train=False, threshold=0.5):
        """
        Constructor of the trainer for the Neuro-Symbolic model.

        :param fm_model: Factorization Machine model to implement the Likes predicate
        :param optimizer: optimizer used for the training of the model
        :param u_g_matrix: users X genres matrix, containing the ratings predicted for each user-genre pair in the
        source domain. This is the pre-trained model. It is the G matrix in the paper
        :param i_g_matrix: items X genres matrix, containing a 1 if the item i belongs to genre j, 0 otherwise. It is
        used to implement the HasGenre predicate
        :param p_pos: hyper-parameter p for universal quantifier aggregator used on positive user-item pairs (Axiom 1)
        :param p_neg: hyper-parameter p for universal quantifier aggregator used on negative user-item pairs (Axiom 2)
        :param p_sat_agg: hyper-parameter p for universal quantifier aggregator used in the sat agg operator
        :param p_forall: hyper-parameter p for universal quantifier aggregator used on the formula that enables
        domain adaptation (Axiom 3)
        :param p_exists: hyper-parameter p for existential quantifier aggregator used on on the formula that enables
        domain adaptation (Axiom 3)
        :param wandb_train: whether the training information has to be logged on WandB or not
        :param threshold: threshold used for the decision boundary
        """
        super(LTNTrainerFM, self).__init__(fm_model, optimizer, wandb_train)
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.Likes = ltn.Predicate(fm_model)
        self.LikesGenre = ltn.Predicate(func=lambda u_idx, g_idx: u_g_matrix[u_idx, g_idx])
        self.genres = ltn.Variable("genres", torch.tensor(range(u_g_matrix.shape[1])), add_batch_dim=False)
        self.HasGenre = ltn.Predicate(func=lambda i_idx, g_idx: i_g_matrix[i_idx, g_idx])
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(), quantifier='e')
        self.SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=p_sat_agg))
        self.p_forall = p_forall
        self.p_exists = p_exists
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.threshold = threshold

    def train_epoch(self, train_loader, epoch=None):
        train_loss, train_sat_agg, f1_sat, f2_sat, f3_sat, f3 = 0.0, 0.0, [], [], [], None
        for batch_idx, (u_pos, i_pos, g_pos, u_neg, i_neg, g_neg, u_ukn, i_ukn, g_ukn) in enumerate(train_loader):
            self.optimizer.zero_grad()
            if epoch > 5:
                # compute SAT level of third formula
                f3 = self.Forall(ltn.diag(u_ukn, i_ukn, g_ukn),
                                 self.Implies(
                                     self.Exists(self.genres,
                                                 self.And(
                                                     self.Not(self.LikesGenre(u_ukn, self.genres)),
                                                     self.HasGenre(i_ukn, self.genres)
                                                 ), p=self.p_exists),
                                     self.Not(self.Likes(u_ukn, i_ukn, g_ukn))),
                                 p=self.p_forall)
                f3_sat.append(f3.value.item())
            if u_pos is not None and u_neg is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos, g_pos), self.Likes(u_pos, i_pos, g_pos), p=self.p_pos)
                f2 = self.Forall(ltn.diag(u_neg, i_neg, g_neg), self.Not(self.Likes(u_neg, i_neg, g_neg)), p=self.p_neg)
                f1_sat.append(f1.value.item())
                f2_sat.append(f2.value.item())
                train_sat = self.SatAgg(f1, f2, f3) if epoch > 5 else self.SatAgg(f1, f2)
            elif u_pos is not None:
                f1 = self.Forall(ltn.diag(u_pos, i_pos, g_pos), self.Likes(u_pos, i_pos, g_pos), p=self.p_pos).value
                f1_sat.append(f1.item())
                train_sat = self.SatAgg(f1, f3) if epoch > 5 else f1
            else:
                f2 = self.Forall(ltn.diag(u_neg, i_neg, g_neg), self.Not(self.Likes(u_neg, i_neg, g_neg)), p=self.p_neg).value
                f2_sat.append(f2.item())
                train_sat = self.SatAgg(f2, f3) if epoch > 5 else f2
            train_sat_agg += train_sat.item()
            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader), {"training_overall_sat": train_sat_agg / len(train_loader),
                                                "pos_sat": np.mean(f1_sat),
                                                "neg_sat": np.mean(f2_sat),
                                                "f3_sat": np.mean(f3_sat) if f3_sat else -1.}

    def predict(self, x, dim=1):
        """
        Method for performing a prediction of the model.

        :param x: tensor containing the user-item pair for which the prediction has to be computed. The first position
        is the user index, while the second position is the item index
        :param dim: dimension across which the dot product of the MF has to be computed
        :return: the prediction of the model for the given user-item pair
        """
        with torch.no_grad():
            preds = self.model(x, dim=dim)
        preds = preds >= self.threshold
        return preds
