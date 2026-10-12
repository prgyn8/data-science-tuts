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
    # Q(f) : Boxplot & Skewness

    ---
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Q.F) **Consider the Annual Income (₹) values of all 8 customers from the e-commerce dataset given above**:

    1. Arrange the data in ascending order and calculate the following five-number summary:
        - Minimum
        - First Quartile (Q1)
        - Median (Q2)
        - Third Quartile (Q3)
        - Maximum
        - IQR

    2. Draw a box plot for this data, clearly labeling _Q1_, Median, _Q3_, IQR, whiskers, and any outliers if exists.

    3. Based on the box plot, briefly comment on the skewness of the Annual Income distribution **(left-skewed / right-skewed / symmetric**).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview of Dataset

    ---
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/ecommerce_dataset.csv")

    df
    return (df,)


@app.cell
def _(df):
    annual_incomes = df['Annual Income (INR)']

    print(annual_incomes)
    return (annual_incomes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inter-Quartile Range (IQR)
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(".assets/box-whiskers.webp")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **What's a box plot ?**
    >
    > A box plot is a statistics tool (type of plot) to visualize the distribution of data in the form of box and whiskers. It tells where the middle value lies, how much the data is spread out, where are the extreme values.

    **Key Terminologies:**

    1. Outliers : Data points which deviate significantly from the rest of the data observations i.e (**Extreme values**)
    2. Q1 : 25th percentile of the dataset
    3. Q3 : 75th percentile of the dataset
    4. IQR : A measure of statistical dispersion that represents the spread of the middle 50% of a dataset
    6. Whiskers : The lines out of the box, which represent the **variability** of data outside the middle 50%, it extends 1.5 times the IQR.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Creating the box-plot**
    """)
    return


@app.cell
def _(annual_incomes):
    # Percentiles
    import numpy as np

    q1, q3 = np.percentile(annual_incomes, [25, 75])
    iqr = q3 - q1
    # lower & upper bound whiskers

    w1 = q1 - (1.5 * iqr)
    w2 = q3 - (1.5 * iqr)


    print(f"lower bound whisker : {w1}")
    print(f"upper bound whisker : {w2}")
    return iqr, np, q1, q3


@app.cell
def _(annual_incomes, plt):
    plt.figure(figsize=(12, 6))
    plt.boxplot(annual_incomes, showfliers=True)
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylabel("income")
    plt.title("Annual Income Boxplot")
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Annual Income Summary
    ---
    """)
    return


@app.cell
def _(annual_incomes, np):
    # minimum
    min_income = np.min(annual_incomes)
    print(min_income)
    return


@app.cell
def _(annual_incomes, np):
    # maximum
    max_income = np.max(annual_incomes)
    print(max_income)
    return


@app.cell
def _(annual_incomes, iqr, np, q1, q3):
    median = np.percentile(annual_incomes, 50)

    print("median : ", median)
    print("q1 : ", q1)
    print("q3 : ", q3)
    print("Inter-Quartile Range : ", iqr)
    return (median,)


@app.cell
def _(annual_incomes, median, plt, q1, q3):
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=annual_incomes, fill=True, cut=0)
    plt.axvline(median, linestyle=":", color="red", label="median")
    plt.axvline(q1, linestyle="-", color="green", label="q1")
    plt.axvline(q3, linestyle="-", color="blue", label="q3")
    plt.grid(True)
    plt.title("Annual Income Distribution")
    plt.ticklabel_format(style='plain', axis='y')
    plt.ylabel("income")
    plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > The distribution of Annual Income in the given dataset is rightly skewed.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
