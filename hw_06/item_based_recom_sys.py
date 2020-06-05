import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_prediction_model import generate_prediction_model
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

plt.style.use('ggplot')

random.seed(10)


def read_data():
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    ratings = pd.merge(movies, ratings).drop(['genres', 'timestamp'], axis=1)

    # plot data distribution and remove unpopular movies
    df_movies_cnt = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])
    print('Movies:')
    __plot_and_get_quantile(df_movies_cnt, 'Rating Frequency of All Movies')

    popularity_thres = 10
    popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
    df_ratings_drop_movies = ratings[ratings.movieId.isin(popular_movies)]
    print('shape of original ratings data: ', ratings.shape)
    print('shape of ratings data after dropping unpopular movies: ', df_ratings_drop_movies.shape)

    df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])
    print('Users:')
    __plot_and_get_quantile(df_users_cnt, 'Rating Frequency of All Users')

    # return a matrix where rows are userId and columns are moveIds
    # NaNs at reatings are replaced with each column's mean
    ratings = df_ratings_drop_movies.pivot_table(index=['userId'], columns=['movieId'], values='rating')
    # return ratings, ratings.fillna(ratings.mean())
    return ratings, ratings.fillna(0)


def write_overlaps(overlap_dict):
    with open('overlap_results.txt', 'w') as f:
        f.write(json.dumps(overlap_dict))


def __plot_and_get_quantile(df, title):
    ax = df \
        .sort_values('count', ascending=False) \
        .reset_index(drop=True) \
        .plot(figsize=(12, 8), title=title, fontsize=12)
    ax.set_xlabel("movie Id")
    ax.set_ylabel("number of ratings")
    plt.show()

    print(df['count'].quantile(np.arange(1, 0.6, -0.05)))


def create_train_data_per_proportion(ratings, test_data):
    proportion_of_train_data = [i / 10 for i in range(1, 10)]
    # proportion_of_train_data = [0.1, 0.9]
    train_data = {}
    for proportion in proportion_of_train_data:
        train_data_length = int(len(ratings) * proportion)
        train_data[proportion] = ratings.drop(ratings.index[list(test_data.index - 1)]).sample(n=train_data_length)

    return train_data


def execute_experiment_a(train_data, similarities, test_data, metric_type):
    """ A question"""
    k_nearest = 20
    proportion_of_train_data = [i / 10 for i in range(1, 10)]
    # proportion_of_train_data = [0.1, 0.9]
    saved_models = {}
    for proportion in proportion_of_train_data:
        rmse_value, f1_metric, model_pred = generate_prediction_model(test_data, train_data[proportion], similarities,
                                                                      k_nearest)
        print(
            "For K=%d, similarity_metric=%s test_set_size = 0.1 and train_set_size=%.1f  we got f1=%f and rmse=%f" % (
                k_nearest, metric_type, proportion, f1_metric, rmse_value))
        if proportion == 0.1 or proportion == 0.9:
            # save models for later usage
            saved_models[proportion] = {
                'train_data': train_data[proportion],
                'rmse_value': rmse_value,
                'f1_metric': f1_metric,
                "model_pred": model_pred
            }

    return saved_models


def execute_experiment_b(similarities, test_data, model_90, metric_type):
    """ B question"""
    k_nearest_values = [5, 10, 20, 50, 100]
    train_set_proportion = 0.9
    for k_value in k_nearest_values:
        if k_value == 20:
            # we have already the results from this execution
            print(
                "For K=%d, similarity_metric=%s test_set_size = 0.1 and train_set_size=%.1f we got f1=%f and rmse=%f" %
                (k_value, metric_type, train_set_proportion, model_90['f1_metric'], model_90['rmse_value']))
        else:
            train_data = model_90['train_data']
            rmse_value, f1_metric, model_pred = generate_prediction_model(test_data, train_data, similarities, k_value)
            print(
                "For K=%d, similarity_metric=%s test_set_size = 0.1 and train_set_size=%.1f we got f1=%f and rmse=%f" % (
                    k_value, metric_type, train_set_proportion, f1_metric, rmse_value))


def calculate_overlap(recommendations1, recommendations2):
    iteration_fraction = []
    for index in range(len(recommendations1)):
        number_of_same_nodes = len(set(recommendations1[:index + 1]).intersection(set(recommendations2[:index + 1])))
        iteration_fraction.append(number_of_same_nodes / (index + 1))
    return np.round(sum(iteration_fraction) / len(recommendations1), 5)


def calculate_rbo(recommendations1, recommendations2, p=0.98):
    """
        Calculates Ranked Biased Overlap (RBO) score.
        code taken from: https://github.com/ragrawal/measures/blob/master/measures/rankedlist/RBO.py
    """
    if recommendations1 == None: recommendations1 = []
    if recommendations2 == None: recommendations2 = []

    sl, ll = sorted([(len(recommendations1), recommendations1), (len(recommendations2), recommendations2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
            # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


def predict_for_random_user(model, user_ratings, user_id, k_best=10):
    user_predicted_movies = model.loc[user_id]
    unrated_movies = user_predicted_movies.drop(labels=user_ratings.loc[user_id].dropna().to_dict().keys()).dropna()
    top_k_items = unrated_movies.sort_values(ascending=False)
    return list(top_k_items.index[:k_best])


def main():
    init_user_ratings, user_ratings = read_data()

    # ref: https://stackoverflow.com/questions/37003272/how-to-compute-jaccard-similarity-from-a-pandas-dataframe
    jaccard_similarity_matrix = 1 - pairwise_distances(user_ratings.T, metric="hamming")
    cosine_similarity_matrix = cosine_similarity(user_ratings.T)
    adjusted_cosine_similarity_matrix = 1 - pairwise_distances(user_ratings.T, metric="correlation")

    # get the 10% of dataFrame as test_set. This data set will be used for all experiments
    test_data = user_ratings.sample(frac=0.1, random_state=10)
    # get a dict in format: proportion: train_data
    train_data = create_train_data_per_proportion(user_ratings, test_data)

    print("\nStart question A")
    jaccard_models = execute_experiment_a(train_data, jaccard_similarity_matrix, test_data, 'jaccard')
    print('\n')
    cosine_models = execute_experiment_a(train_data, cosine_similarity_matrix, test_data, 'cosine')
    print('\n')
    adjusted_cosine_models = execute_experiment_a(train_data, adjusted_cosine_similarity_matrix, test_data,
                                                  'adjusted cosine')

    # improve performance
    del train_data

    print("\nStart question B")
    execute_experiment_b(jaccard_similarity_matrix, test_data, jaccard_models[0.9], 'jaccard')
    print('\n')
    execute_experiment_b(cosine_similarity_matrix, test_data, cosine_models[0.9], 'cosine')
    print('\n')
    execute_experiment_b(adjusted_cosine_similarity_matrix, test_data, adjusted_cosine_models[0.9], 'adjusted cosine')

    number_of_users_to_recom = 5

    overlap_dict = {}
    for proportion in jaccard_models.keys():
        overlap_dict[proportion] = {}
        if proportion < 0.9:
            users_id = random.sample(list(jaccard_models[proportion]['train_data'].index), number_of_users_to_recom)
            k_users_data = init_user_ratings[init_user_ratings.index.isin(users_id)]
        else:
            k_users_data = init_user_ratings.drop(init_user_ratings.index[list(test_data.index - 1)]).sample(
                n=number_of_users_to_recom, random_state=10)

        for user_id in list(k_users_data.index):
            overlap_dict[proportion][user_id] = {
                'overlap': {},
                'rbo': {}
            }
            jaccard_recom = predict_for_random_user(
                pd.DataFrame(
                    jaccard_models[proportion]['model_pred'],
                    columns=jaccard_models[proportion]['train_data'].columns,
                    index=jaccard_models[proportion]['train_data'].index),
                init_user_ratings,
                user_id)
            cosine_recom = predict_for_random_user(
                pd.DataFrame(
                    cosine_models[proportion]['model_pred'],
                    columns=cosine_models[proportion]['train_data'].columns,
                    index=cosine_models[proportion]['train_data'].index),
                init_user_ratings,
                user_id)
            adjusted_cosine_recom = predict_for_random_user(
                pd.DataFrame(
                    adjusted_cosine_models[proportion]['model_pred'],
                    columns=adjusted_cosine_models[proportion]['train_data'].columns,
                    index=adjusted_cosine_models[proportion]['train_data'].index),
                init_user_ratings,
                user_id)

            overlap_dict[proportion][user_id]['overlap']['jaccard_cosine'] = calculate_overlap(jaccard_recom,
                                                                                               cosine_recom)
            overlap_dict[proportion][user_id]['overlap']['jaccard_adjust_cosine'] = \
                calculate_overlap(jaccard_recom, adjusted_cosine_recom)
            overlap_dict[proportion][user_id]['overlap']['cosine_adjust_cosine'] = \
                calculate_overlap(cosine_recom, adjusted_cosine_recom)

            overlap_dict[proportion][user_id]['rbo']['jaccard_cosine'] = calculate_rbo(jaccard_recom, cosine_recom)
            overlap_dict[proportion][user_id]['rbo']['jaccard_adjust_cosine'] = calculate_rbo(jaccard_recom,
                                                                                              adjusted_cosine_recom)
            overlap_dict[proportion][user_id]['rbo']['cosine_adjust_cosine'] = calculate_rbo(cosine_recom,
                                                                                             adjusted_cosine_recom)

    write_overlaps(overlap_dict)


if __name__ == '__main__':
    main()
