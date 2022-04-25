import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import pandas as pd


n = 1000000
x, y = make_regression(n_samples=n, n_features=1, noise=30)
print(x.shape, y.shape)
y = y * .01
y = y.reshape(n, 1)
data = np.hstack((x, y))

def Cov(data: np.ndarray):
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
        1. Get the mean of the data and remove the mean from the data (centering the data around the POO(point of origin))
        length. Square all data points(The numbers all become positive and move to the right)



    """
    n_data = data - np.mean(data, axis=0)
    c = n_data.shape[1]
    N = n_data.shape[0]
    covariance_matrix = np.empty([c, c])
    for i in range(c):
        for j in range(c):
            covariance = np.sum(n_data[:, i] * n_data[:, j]) / N
            covariance_matrix[i, j] = covariance

    return covariance_matrix


def Compute_eigen_vectors(data, A):
    # v is eigen vector
    # lambda is just a number
    # A(v) = lambda(v)

    eigen_values, eigen_vector = np.linalg.eig(A)
    return eigen_values, eigen_vector

def plot(data):
    lambda_, ev = Compute_eigen_vectors(data, A=Cov(data))
    length = 10
    df = pd.DataFrame(ev, columns = ["ev1","ev2"],index = ["x","y"],dtype=np.float64)
    print(ev,df)
    plt.plot([df.loc["x","ev1"] * length, df.loc["x","ev1"] * -length], [df.loc["y","ev1"] * length, df.loc["y","ev1"] * -length], 'r', label='x')
    plt.plot([df.loc["x","ev2"] * length, df.loc["x","ev2"] * -length], [df.loc["y","ev2"] * length, df.loc["y","ev2"] * -length], 'b', label='y')
    plt.plot([0, ev[0][0]], [0, ev[1][0]], 'r', label='eigen vector 1')
    plt.plot([0, ev[0][1]], [0, ev[1][1]], 'b', label='eigen vector')
    plt.scatter(data[:, 0], data[:, 1], color='black', label='Data')


npcov = np.cov(data, bias=True, rowvar=False)

print(np.allclose(Cov(data), npcov, atol=.0000001))

plot(data)
# plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
