import numpy as np
from sklearn.metrics import mean_squared_error,  f1_score


def __rmse(pred, actual):
    __pred = pred[:actual.shape[0],:actual.shape[1]].flatten()
    __actual = actual[:actual.shape[0],:actual.shape[1]].flatten()
    return mean_squared_error(__pred, __actual, squared=False)


def __calculate_f1_metric(y_pred, y_true):
    __pred = []
    __actual = []

    for value in y_pred[:y_true.shape[0],:y_true.shape[1]].flatten():
        if value >= 3:
            __pred.append(1)
        else:
            __pred.append(0)

    for value in y_true[:y_true.shape[0],:y_true.shape[1]].flatten():
        if value >= 3:
            __actual.append(1)
        else:
            __actual.append(0)

    __f1_score = f1_score(__actual, __pred)
    return __f1_score


def __predict(ratings, similarities, k_nearest):
    pred = np.zeros(ratings.shape)
    for j in range(ratings.shape[1]):
        top_k_items = list(np.argsort(similarities[:, j])[:-k_nearest - 1:-1])
        for i in range(ratings.shape[0]):
            pred[i, j] = similarities[j, :][top_k_items].dot(ratings.values[i, :][top_k_items].T) / np.sum(
                np.abs(similarities[j, :][top_k_items]))

    return pred


def generate_prediction_model(test_data, train_data, similarities, k_nearest):
    pred = __predict(train_data, similarities, k_nearest)

    test_data_matrix = test_data.values
    rmse_value = __rmse(pred, test_data_matrix)

    f1_metric = __calculate_f1_metric(pred, test_data_matrix)

    return rmse_value, f1_metric, pred

# pred = __predict(train_data, similarities).values
# model_knn = NearestNeighbors(metric='jaccard', algorithm='ball_tree', n_neighbors=20, n_jobs = -1)
# model_knn.fit(train_data)
# def __predict(ratings, similarities):
#     pred = ratings.dot(similarities) / np.array([np.abs(similarities).sum(axis=1)])
#     # similarities = pd.DataFrame(similarities, index=list(range(1, similarities.shape[0] + 1)),
#     #                             columns=list(range(1, similarities.shape[0] + 1)))
#     # order = np.argsort(similarities.values, axis=1)[:, :20] # find k_nearest movies for each movie
#     # df = similarities.apply(lambda x: pd.Series(x.sort_values()
#     #                                   .iloc[:20].values,
#     #                                   index=['top{}'.format(i) for i in range(1, 20 + 1)]), axis=1)
#     z = 1
#     return pred
