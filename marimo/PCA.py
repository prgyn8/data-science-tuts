import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principal Component Analysis (PCA)
    ---

    What is it
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Overview
    ---
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("ecommerce_dataset.csv")

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Standardization
    """)
    return


@app.cell
def _(df):
    from sklearn.preprocessing import StandardScaler
    features = df[['Age', 'Avg Order Value (INR)', 'Website Visits/Month']]

    features
    return StandardScaler, features


@app.cell
def _(StandardScaler, features):
    scaler = StandardScaler()
    s_features = scaler.fit_transform(features.to_numpy())

    s_features
    return (s_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. Building the covariance matrix
    """)
    return


@app.cell
def _(np, s_features):
    cov_matrix = np.cov(s_features.T)

    cov_matrix
    return (cov_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. Getting the `eigenvalues` & `eigenvectors`
    """)
    return


@app.cell
def _(cov_matrix, np):
    e = np.linalg.eig(cov_matrix)

    eigenvals = e.eigenvalues

    # getting indices sorted in descending order
    sorted_indices = np.argsort(eigenvals)[::-1]

    sorted_eigenvals = eigenvals[sorted_indices]


    eigenvecs = e.eigenvectors[:, sorted_indices]

    eigenvecs
    return eigenvecs, sorted_eigenvals


@app.cell
def _(sorted_eigenvals):
    sorted_eigenvals
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4. Retrieving the principal components
    """)
    return


@app.cell
def _(eigenvecs):
    k = 2

    proj_matrix = eigenvecs[:, :k]

    proj_matrix
    return (proj_matrix,)


@app.cell
def _(np, proj_matrix, s_features):
    X_pca = np.dot(s_features, proj_matrix)

    X_pca
    return (X_pca,)


@app.cell
def _(np, sorted_eigenvals):
    explained_variance_ratios = (sorted_eigenvals / np.sum(sorted_eigenvals)) * 100
    total_variance = np.sum(explained_variance_ratios[:2])
    return (explained_variance_ratios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting
    ---
    """)
    return


@app.cell
def _(X_pca, eigenvecs, explained_variance_ratios, np, sorted_eigenvals):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot customer data points (scatter)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], color='red', s=80, label='Customers')

    # Plot the 2 Principal Component directions (quiver arrows from origin)
    for i in range(2):
        ax.quiver(
            0, 0, 
            eigenvecs[0, i] * np.sqrt(sorted_eigenvals[i]), 
            eigenvecs[1, i] * np.sqrt(sorted_eigenvals[i]), 
            angles='xy', scale_units='xy', scale=1, 
            color=['navy', 'steelblue'][i], 
            label=f'PC{i+1} ({explained_variance_ratios[i]:.1f}%)'
        )

    # Formatting labels and grid
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel(f'PC1 ({explained_variance_ratios[0]:.1f}% Variance)')
    ax.set_ylabel(f'PC2 ({explained_variance_ratios[1]:.1f}% Variance)')
    ax.set_title('PCA Scatter & Quiver Plot')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
