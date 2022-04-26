# Principal Component Analysis

## What is Principal Component Analysis?
Principal Component Analysis, or PCA, is a dimensionality-reduction approach that is frequently used to decrease the dimensionality of big data sets by reducing a large collection of variables into a smaller set that retains the majority of the information in the large set. Obviously, reducing the number of variables in a data collection reduces accuracy, but the idea in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to study and display, and because machine learning algorithms can analyze data much more easily and quickly without having to deal with superfluous factors.

- - - - 
### Sources :
[builtin](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

[towardsdatascience](https://towardsdatascience.com/an-intuitive-guide-to-pca-1174055fc800)

- - - -
## How Principal Component Analysis work?
1. Compute the covariance matrix to identify correlations
2. Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
3. Recast the data along the principal components axes

### Flow Chart
<!-- ![picture alt](https://gyazo.com/2ccc1a24b8bc2fe51f6b19e9c780b834.png "Flow Chart") -->
- - - -
## PCA in action
### PCA using make_regression to generate data
![picture alt](https://i.gyazo.com/a3bc98c04c02781413ddfce8ed3e6787.png)
### PCA using make_blobs to generate data
![picture alt](https://i.gyazo.com/40e5577fb2e1bbed4cada3dff1ea2e97.png)
