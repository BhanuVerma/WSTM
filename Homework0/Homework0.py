# __author__ = 'Bhanu Verma'
# GTid = '903151012'

import numpy as np
import scipy
from scipy import linalg
from scipy.sparse.linalg import spsolve
import sklearn.preprocessing as sp
import time


def test_run():
    """Driver function."""
    # Define input parameters

    # Question 1

    mat_a = np.random.uniform(0, 1, (2, 2))
    mat_b = np.random.uniform(0, 1, (1, 1))
    mat_c = np.random.uniform(0, 1, (2, 2))
    block_mat = linalg.block_diag(mat_a, mat_b, mat_c)

    # Question 2

    u, s, v = linalg.svd(block_mat)
    # print u, s, v

    # Question 3

    new_mat = block_mat.copy()
    np.random.shuffle(new_mat)

    # Question 4

    u_prime, s_prime, v_prime = linalg.svd(new_mat)
    # print u_prime, s_prime, v_prime

    # Question 5
    # print s, s_prime
    # permutations of the rows of the matrix does not change the singular values

    # Question 6

    norm_vector = np.linalg.norm(new_mat, ord=1, axis=0)
    unit_norm = new_mat/norm_vector
    # print unit_norm

    # Question 7

    u_dash, s_dash, v_dash = linalg.svd(unit_norm)
    # print u_dash, s_dash, v_dash

    # Question 8

    vector = np.ones((1, unit_norm.shape[0]))
    eigen = vector.dot(unit_norm)
    # print eigen
    # Yes, this is an eigenvector

    # Question 10

    indices_arr = np.transpose(np.nonzero(new_mat))
    mat_prime = new_mat.copy()
    for index in indices_arr:
        mat_prime[index[0]][index[1]] = 1

    # Question 11

    vect = np.array(range(mat_prime.shape[1]))
    # print vect

    # Question 12

    result = np.multiply(vect, mat_prime)
    i = 3  # ith row
    # print max(result[i])

    # Question 13

    for i in range(1000):
        result = np.multiply(vect, mat_prime)
        for j in range(vect.shape[0]):
            vect[j] = max(result[j])

    # print vect
    # Value of each element of vect converges to n-1 where n x n is the size of the matrix

    # Question 14

    sparse_mat = scipy.sparse.rand(15000, 15000, density=0.000444445)
    arr_tuple = np.nonzero(sparse_mat)
    # print len(arr_tuple[0])

    # Question 15

    # start_time = time.time()
    ind_tuple = np.transpose(np.nonzero(sparse_mat))
    sum_list = ((sparse_mat.sum(axis=0)).tolist())
    sum_list = sum_list[0]
    sum_val = sum(sum_list)
    mean = sum_val/len(ind_tuple)
    csr_mat = scipy.sparse.csr_matrix(sparse_mat)
    indices_list = np.transpose(np.nonzero(csr_mat))
    for indices in indices_list:
        csr_mat[indices[0], indices[1]] -= mean
    # print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    scaler = sp.StandardScaler(copy=True, with_mean=False, with_std=True).fit(csr_mat)
    norm_mat = scaler.transform(csr_mat)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # print norm_mat

    rand_vect = np.random.randn(15000, 1)
    dot_prod = norm_mat.dot(rand_vect)
    # print dot_prod

    # this takes atleast twice as much time as the above code
    # start_time = time.time()
    # fast_tuple = scipy.sparse.find(sparse_mat)
    # elements = fast_tuple[2]
    # sum_val_1 = sum(elements)
    # mean_1 = sum_val_1/len(elements)
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    test_run()

