import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Google+Sans+Flex:wght@420;430;500;600&family=Google+Sans+Code:wght@400;500&display=swap');

    :root {
        --marimo-text-font: 'Google Sans Flex', sans-serif !important;
        --marimo-heading-font: 'Geist', sans-serif !important;
        /* --marimo-monospace-font: 'Google Sans Code', monospace !important; */
    }

    .prose {
        font-family: var(--marimo-text-font) !important;
        font-size: 17px !important;
        line-height: 1.5 !important;
    }

    .prose p,
    .prose ul,
    .prose ol,
    .prose li,
    .prose table {
        font-family: var(--marimo-text-font) !important;
        font-size: 16.5px !important;
        line-height: 1.5 !important;
        color: #1f2937;
    }

    .prose h1,
    .prose h2,
    .prose h3 {
        font-family: var(--marimo-heading-font) !important;
        font-weight: 500;
        letter-spacing: -0.02em;
    }

    </style>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Q(c) : Onehot Encoding
    ---

    In general, Machine Learning models like Linear Regression, SVM, they don't process the categorical raw data, So, onehot encoding (_a preprocessing method_), which helps in converting the categorical features into **numerical features** (_that the model understands_).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Q(c) : The City variable will be given to a machine learning algorithm and we know that this is the categorical data. Which method should be applied in order to convert this data into numeric data so that an algorithm will analyze it correctly. Using that method perform the conversion representation of C101, C102, C103.

    _(3rd question from 1st Internal)_
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/ecommerce_dataset.csv")

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Steps:**

    1. Find the unique values of certain column (categorical) -> identities
    2. Sub-lists for each row,entry. I'll be calling it as vectors.
    3. Combining all the vectors into main list i.e. onehot_encoded
    4. Give it to the model for training....
    """)
    return


@app.cell
def _(df):
    # get the unique values
    identities = df['City'].unique()

    print(identities)
    return (identities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```

    identities -> ['Bhopal', 'Indore', 'Ujjain']

    Bhopal -> [1, 0, 0]
    Indore -> [0, 1, 0]

    ```

    1: states True

    0: states False
    """)
    return


@app.cell
def _(df, identities):
    onehot_encoded = []

    for city in df['City']:
        # building vector for each row
        vector = [1 if city==category else 0 for category in identities]

        # storing the results
        onehot_encoded.append(vector)


    print(onehot_encoded)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > The len(vector) will be the no. of unique values of the column
    """)
    return


@app.cell
def _(pd):
    def onehot_encoding(feature: pd.Series):
        """
        Converts categorical feature into numerical feature.
        """
        identities = feature.unique()
        onehot_encoded = []

        for val in feature:
            vector = [1 if val==category else 0 for category in identities]
            onehot_encoded.append(vector)

        return onehot_encoded

    return (onehot_encoding,)


@app.cell
def _(df, onehot_encoding):
    print(onehot_encoding(df['Membership']))
    return


@app.cell
def _(df, onehot_encoding, pd):
    n_df = df[['Customer', 'Gender']]
    c_df = pd.DataFrame(onehot_encoding(df['Gender']), columns=df['Gender'].unique())

    pd.concat([n_df, c_df], axis=1)
    return


if __name__ == "__main__":
    app.run()
