# __author__ = 'Bhanu Verma'
# GTid = '903151012'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def test_run():
    df_a = pd.read_csv('ml-100k/ua.base', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    df_b = pd.read_csv('ml-100k/ub.base', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    df = pd.merge(df_a, df_b, how='outer')

    test_a = pd.read_csv('ml-100k/ua.test', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    test_b = pd.read_csv('ml-100k/ub.test', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    test_df = pd.merge(test_a, test_b, how='outer')

    user_df = df.groupby(['user_id']).count()
    item_df = df.groupby(['item_id']).count()

    plt.figure(1, figsize=(12, 7))
    plt.xlabel('User Ids')
    plt.ylabel('Number of movies user has rated')
    plt.title('Histogram for User Ratings')
    plt.bar(list(user_df.index), user_df['item_id'].values)

    plt.figure(2, figsize=(12, 7))
    plt.xlabel('Movie Ids')
    plt.ylabel('Number of user ratings for a movie')
    plt.title('Histogram for Movie Ratings')
    plt.bar(list(item_df.index), item_df['user_id'].values)

    user = df['user_id'].values
    item = df['item_id'].values
    ratings = df['rating'].values
    mat = csr_matrix((ratings, (user, item)))

    u_test = test_df['user_id'].values
    i_test = test_df['item_id'].values
    t_rating = test_df['rating'].values

    mat_sum = mat.sum()
    mu = mat_sum/float(len(user))
    u_count = mat.getnnz(axis=1)
    i_count = mat.getnnz(axis=0)

    b_u = np.zeros(mat.shape[0])
    b_i = np.zeros(mat.shape[1])
    u_step = np.zeros(mat.shape[0])
    i_step = np.zeros(mat.shape[1])

    for i, val in enumerate(b_u):
        if mat.getrow(i).sum() == 0:
            b_u[i] = 0
        else:
            b_u[i] = (mat.getrow(i).sum()/float(u_count[i])) - mu

    for i, val in enumerate(b_i):
        if mat.getcol(i).sum() == 0:
            b_i[i] = 0
        else:
            b_i[i] = (mat.getcol(i).sum()/float(i_count[i])) - mu

    sigma = 0

    for i, val in enumerate(t_rating):
        sigma += (val - (mu + b_u[int(u_test[i])] + b_i[int(i_test[i])]))**2

    rmse = (sigma/float(len(t_rating)))**0.5
    # print rmse

    # plt.show()
    rmse_arr = []
    for z in range(10):
        # sigma of cost function
        lambda_val = 25
        sq_diff = 0
        b_u_square_sum = 0
        b_i_square_sum = 0

        for i, val in enumerate(ratings):
            u_index = int(user[i])
            i_index = int(item[i])
            sq_diff += (val - mu - b_u[u_index] - b_i[i_index])**2
            b_u_square_sum += b_u[u_index]**2
            b_i_square_sum += b_i[i_index]**2

        reg_val = lambda_val * (b_u_square_sum + b_u_square_sum)
        cost = sq_diff + reg_val
        # u step
        for i, val in enumerate(b_u):
            if mat.getrow(i).sum() == 0:
                u_step[i] = 0
            else:
                r_ui = mat.getrow(i).sum()
                u_sum = mu * u_count[i]
                bu_sum = b_u[i] * u_count[i]
                bi_sum = 0
                i_indices = mat.getrow(i).indices
                for index in i_indices:
                    bi_sum += b_i[index]

                u_step[i] = -2*(r_ui - u_sum - bu_sum - bi_sum) + 2*lambda_val*bu_sum

        # i step
        for i, val in enumerate(b_i):
            if mat.getcol(i).sum() == 0:
                i_step[i] = 0
            else:
                r_ui = mat.getcol(i).sum()
                u_sum = mu * i_count[i]
                bi_sum = b_i[i] * i_count[i]
                bu_sum = 0
                u_indices = mat.getcol(i).indices
                for index in u_indices:
                    bu_sum += b_u[index]

                i_step[i] = -2*(r_ui - u_sum - bu_sum - bi_sum) + 2*lambda_val*bi_sum

        b_u = b_u - (u_step * (1/float(2*len(ratings))))
        b_i = b_i - (i_step * (1/float(2*len(ratings))))
        sigma = 0

        for i, val in enumerate(t_rating):
            sigma += (val - (mu + b_u[int(u_test[i])] + b_i[int(i_test[i])]))**2

        rmse = (sigma/float(len(t_rating)))**0.5
        rmse_arr.append(rmse)

    print min(rmse_arr)

if __name__ == "__main__":
    test_run()
