# __author__ = 'Bhanu Verma'
# GTid = '903151012'

import numpy as np
from scipy import linalg


def test_run():
    """Driver function."""
    # Define input parameters

    # Question 1

    mat_a = np.random.uniform(0, 1, (2, 2))
    mat_b = np.random.uniform(0, 1, (1, 1))
    mat_c = np.random.uniform(0, 1, (2, 2))
    block_mat = linalg.block_diag(mat_a, mat_b, mat_c)
    # print block_mat

    # Question 2

    u, s, v = linalg.svd(block_mat)
    # print u, s, v

    # Question 3

    new_mat = block_mat.copy()
    np.random.shuffle(new_mat)
    # print new_mat

    # Question 4

    u_prime, s_prime, v_prime = linalg.svd(new_mat)
    # print u_prime, s_prime, v_prime

    # Question 5
    # print s, s_prime
    # permutations of the rows of the matrix does not change the singular values

    # Question 6

    norm_vector = np.linalg.norm(new_mat, ord=1, axis=0)
    unit_norm = new_mat/norm_vector

    # Question 7

    u_dash, s_dash, v_dash = linalg.svd(unit_norm)
    # print u_dash, s_dash, v_dash

    # Question 8
    vector = np.ones(unit_norm.shape)
    print unit_norm
    print vector
    print
    print np.multiply(unit_norm, vector)



if __name__ == "__main__":
    test_run()

