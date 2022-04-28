import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
import pandas as pd


def make_regression_dataset(n=1000, noise=31):
    """
    param n: number of data points
    :param n:
    :param noise:
    :return:
    """
    x, y = make_regression(n_samples=n, n_features=1, noise=noise)
    print(x.shape, y.shape)
    y = y * .01  # make the data less vertical
    y = y.reshape(n, 1)
    data = np.hstack((x, y))
    return data


def make_blobs_dataset(n=1000, cluster_std=1.5):
    """
    param n: number of data points
    :param n:
    :param cluster_std:
    :return:
    """
    o_data, temp1, temp2 = make_blobs(n_samples=1000, centers=1, n_features=2, random_state=0,
                                      return_centers=True, cluster_std=1.5)
    data = o_data.reshape(1000, 2)
    return data


def compute_covariance_matrix(data: np.ndarray):
    """
    param data

    return covariance matrix

    Variance
    ___________
        > Get the mean of the data and remove the mean from the data (centering the data around the POO(point of origin))
        > Square all data points(The numbers all become positive and move to the right)
        > Add data points together
        > Divide by N(number of data points)
        > Variance of a single number is 0
    Covariance:
    ___________

    """
    n_data = data - np.mean(data, axis=0)
    c = n_data.shape[1]
    N = n_data.shape[0]
    covariance_matrix = np.empty([c, c])
    for i in range(c):
        for j in range(c):
            covariance = np.sum(n_data[:, i] * n_data[:, j]) / N
            covariance_matrix[i, j] = covariance

    return covariance_matrix, n_data


def compute_eigen_vectors(A):
    # v is eigen vector
    # lambda is just a number
    # A(v) = lambda(v)
    """
    :param A:
    :return:
    """

    eigen_values, eigen_vector = np.linalg.eig(A)
    return eigen_values, eigen_vector


def plot(data, ev):
    length = 10
    df = pd.DataFrame(ev, columns=["ev1", "ev2"], index=["x", "y"], dtype=np.float64)
    print(ev, df)
    plt.plot([df.loc["x", "ev1"] * length, df.loc["x", "ev1"] * -length],
             [df.loc["y", "ev1"] * length, df.loc["y", "ev1"] * -length], 'r', label='x')
    plt.plot([df.loc["x", "ev2"] * length, df.loc["x", "ev2"] * -length],
             [df.loc["y", "ev2"] * length, df.loc["y", "ev2"] * -length], 'b', label='y')
    plt.scatter(data[:, 0], data[:, 1], color='black', label='Data')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()


def main():
    # data = make_blobs_dataset()
    data = make_regression_dataset()
    covariance_matrix, data = compute_covariance_matrix(data)
    numpy_covariance_matrix = np.cov(data, bias=True, rowvar=False)
    print(np.allclose(covariance_matrix, numpy_covariance_matrix, atol=.0000001))
    lambda_, ev = compute_eigen_vectors(A=covariance_matrix)
    print(lambda_, ev)
    plot(data, ev)


if __name__ == "__main__":
    main()
