import pandas as pd
import os
from sklearn.model_selection import train_test_split as train_test_split_sklearn
import numpy as np
from scipy.sparse import csr_matrix
import torch


def increase_sparsity(train_set, p, seed=0, user_level=False):
    """
    It makes the given dataset more sparse by randomly sampling `p`% ratings.

    :param train_set: training set from which the ratings have to be sampled
    :param p: percentage of ratings that has to be sampled
    :param seed: seed used for randomly sampling of ratings
    :param user_level: whether the sampling of ratings has to be performed at the user level or not. If True, `p`%
    ratings are randomly picked from each user. If False, the ratings are randomly sampled independently from the user
    :return: the new sparse training set
    """
    if p == 1.00:
        return train_set
    else:
        # convert dataset to pandas
        train_set_df = pd.DataFrame(train_set, columns=['userId', 'uri', 'sentiment'])
        if user_level:
            # take the given percentage of ratings from each user
            sampled_ids = train_set_df.groupby(by=["userId"]).apply(
                lambda x: x.sample(frac=p, random_state=seed)
            )
            sampled_ids = sampled_ids.index.get_level_values(1)
            return train_set_df.iloc[sampled_ids].reset_index(drop=True).to_numpy()
        else:
            # we use scikit-learn to take the ratings
            # note that stratify is set to True to maintain the distribution of positive and negative ratings
            _, train_set_df = train_test_split_sklearn(train_set_df, random_state=seed,
                                                       stratify=train_set_df["sentiment"], test_size=p)
            return train_set_df.to_numpy()


def train_test_split(seed, ratings, frac=None, user_level=False):
    """
    It splits the dataset into training and test sets.

    :param seed: seed for random reproducibility
    :param ratings: dataframe containing the dataset ratings that have to be split
    :param frac: proportion of ratings to be sampled to create the test set. If None, it randomly sample one positive
    rating (LOO)
    :param user_level: whether the train test split has to be performed at the user level or ratings can
    be sampled randomly independently from the user. Defaults to False, meaning that the split is not done at the
    user level. In case the split is not done at the user level, a stratified split is performed to mantain the
    distribution of the classes
    :return: train and test set dataframes
    """
    if user_level:
        test_ids = ratings.groupby(by=["userId"]).apply(
            lambda x: x.sample(frac=frac, random_state=seed)
        )
        test_ids = test_ids.index.get_level_values(1)
        train_ids = np.setdiff1d(ratings.index.values, test_ids)
        train_set = ratings.iloc[train_ids].reset_index(drop=True)
        test_set = ratings.iloc[test_ids].reset_index(drop=True)
        return train_set, test_set
    else:
        # we use scikit-learn train-test split
        return train_test_split_sklearn(ratings, random_state=seed, stratify=ratings["sentiment"],
                                        test_size=frac)


def process_mindreader(seed, version, genre_val_size=0.2, genre_user_level_split=False, movie_val_size=0.2,
                       movie_test_size=0.2, movie_user_level_split=False):
    """
    This function processes the MindReader dataset and prepares it for the experiment. In particular, it executes
    the following pipeline:
    1. it selects the correct dataset version
    2. it removes unknown ratings
    3. it removes users who rated only movies or only genres
    4. it creates train and validation sets for the genre ratings
    5. it creates train, validation and test sets for the movie ratings
    6. it creates the items X genres matrix, containing 1 if the the movie i belongs to genre j, 0 otherwise

    :param seed: seed for reproducibility
    :param version: version of the dataset (i.e., mindreader-100k or mindreader-200k)
    :param genre_val_size: size of validation set for genre ratings
    :param genre_user_level_split: whether the split for the movie ratings has to be done at the user level or not
    :param movie_val_size: size of validation set for movie ratings
    :param movie_test_size: size of test set for movie ratings
    :param movie_user_level_split: whether the split for the movie ratings has to be done at the user level or not
    """
    # check the version is correct
    assert version in ("mindreader-100k", "mindreader-200k"), "The version of the dataset must be " \
                                                              "'mindreader-100k' or 'mindreader-200k'."
    # get MindReader entities
    mr_entities = pd.read_csv(os.path.join("datasets", version, "entities.csv"))
    mr_entities_dict = mr_entities.to_dict("records")
    # get MindReader genres
    mr_genres = [entity["uri"] for entity in mr_entities_dict if "Genre" in entity["labels"]]
    # get MindReader ratings
    mr_ratings = pd.read_csv(os.path.join("datasets", version, "ratings.csv"))
    # remove useless columns from dataframe
    mr_ratings = mr_ratings.drop(mr_ratings.columns[[0, 3] if version == "mindreader-200k" else [0, 1, 4]],
                                 axis=1).reset_index(drop=True)
    # get ratings on movie genres and remove unknown ratings
    genre_ratings = mr_ratings[mr_ratings["uri"].isin(mr_genres) & mr_ratings["sentiment"] != 0]
    # get movie ratings and remove unknown ratings
    mr_movies = [entity["uri"] for entity in mr_entities_dict if "Movie" in entity["labels"]]
    movie_ratings = mr_ratings[mr_ratings["uri"].isin(mr_movies) & mr_ratings["sentiment"] != 0]
    # get ids of users who rated at least one genre genre
    u_who_rated_genres = genre_ratings["userId"].unique()
    # get ids of users who rated at least one movie
    u_who_rated_movies = movie_ratings["userId"].unique()
    # get intersection of previous users to get the set of users of the dataset
    u_who_rated = set(u_who_rated_movies) & set(u_who_rated_genres)
    # remove all movie ratings of users who did not rate at least one genre
    movie_ratings = movie_ratings[movie_ratings["userId"].isin(u_who_rated)]
    # remove all genre ratings of user who did not rate at least one movie
    genre_ratings = genre_ratings[genre_ratings["userId"].isin(u_who_rated)]
    # get integer user, genre, and item indexes
    u_ids = genre_ratings["userId"].unique()
    int_u_ids = list(range(len(u_ids)))
    user_string_to_id = dict(zip(u_ids, int_u_ids))
    g_ids = genre_ratings["uri"].unique()
    int_g_ids = list(range(len(g_ids)))
    g_string_to_id = dict(zip(g_ids, int_g_ids))
    i_ids = movie_ratings["uri"].unique()
    int_i_ids = list(range(len(i_ids)))
    i_string_to_id = dict(zip(i_ids, int_i_ids))
    # apply the new indexing
    genre_ratings = genre_ratings.replace(user_string_to_id).replace(g_string_to_id).reset_index(drop=True)
    movie_ratings = movie_ratings.replace(user_string_to_id).replace(i_string_to_id).reset_index(drop=True)
    # convert negative ratings from -1 to 0
    genre_ratings = genre_ratings.replace(-1, 0)
    movie_ratings = movie_ratings.replace(-1, 0)
    # get number of users, genres, and movies
    n_users = genre_ratings["userId"].nunique()
    n_genres = genre_ratings["uri"].nunique()
    n_items = movie_ratings["uri"].nunique()
    # create train and validation set for genre ratings
    g_train_set, g_val_set = train_test_split(seed, genre_ratings, frac=genre_val_size,
                                              user_level=genre_user_level_split)
    # create train, validation and test set for movie ratings
    train_set, test_set = train_test_split(seed, movie_ratings, frac=movie_test_size, user_level=movie_user_level_split)
    train_set_small, val_set = train_test_split(seed, train_set, frac=movie_val_size, user_level=movie_user_level_split)
    # create items X genres matrix
    mr_triples = pd.read_csv(os.path.join("datasets", version, "triples.csv"))
    # get only triples with HAS_GENRE relationship, where the head is a movie and the tail a movie genre
    mr_genre_triples = mr_triples[mr_triples["head_uri"].isin(i_string_to_id.keys()) &
                                  mr_triples["tail_uri"].isin(g_string_to_id.keys()) &
                                  (mr_triples["relation"] == "HAS_GENRE")]
    # replace URIs with integer indexes
    mr_genre_triples = mr_genre_triples.replace(i_string_to_id).replace(g_string_to_id)
    # construct the item X genres matrix
    i_g_matrix = csr_matrix((np.ones(len(mr_genre_triples)),
                             (list(mr_genre_triples["head_uri"]),
                              list(mr_genre_triples["tail_uri"]))), shape=(n_items, n_genres))

    return {"n_users": n_users, "n_genres": n_genres, "n_items": n_items, "g_tr": g_train_set.to_numpy(),
            "g_val": g_val_set.to_numpy(), "i_tr": train_set.to_numpy(), "i_tr_small": train_set_small.to_numpy(),
            "i_val": val_set.to_numpy(), "i_test": test_set.to_numpy(),
            "i_g_matrix": torch.tensor(i_g_matrix.toarray())}
