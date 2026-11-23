import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is Logistic Regression ?
    ---

    Logistic Regression is a **classification** method that predicts the **probability** that something belongs to class 1 (for example, pass or fail, spam or not spam).

    It first calculates a straight-line score $z = wx + b$, and then squeezes that score into a value between 0 and 1 using the **sigmoid function**:

    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

    If the probability is 0.5 or more, we predict class 1. Otherwise, we predict class 0.

    > Despite the word "regression" in its name, it is used for classification.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **The Model**
    ---

    $z = w_1x_1 + w_2x_2 + \dots + b$

    $p = P(y = 1 \mid x) = \sigma(z)$

    **Meaning of the weights:** the model is a straight line in the **log-odds**:

    $$\ln\frac{p}{1-p} = wx + b$$

    So if $x$ increases by 1, the **odds** of class 1 get multiplied by $e^{w}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("public/lr.jpg.jpeg", width=650, height=400).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Cost Function (Log Loss / Binary Cross-Entropy)**

    $$J(w, b) = -\frac{1}{m}\sum_{i=1}^{m}\Big[\,y_i\ln(p_i) + (1 - y_i)\ln(1 - p_i)\,\Big]$$

    **How to read it:**

    - If the true label is $y = 1$, the cost is $-\ln(p)$. It is 0 when $p = 1$, and becomes very large when $p$ is close to 0.
    - If the true label is $y = 0$, the cost is $-\ln(1 - p)$. It is 0 when $p = 0$, and becomes very large when $p$ is close to 1.

    So the model is punished heavily when it is **confident and wrong**.

    **Why not use Mean Squared Error?** With the sigmoid, MSE gives a bumpy (non-convex) cost surface, and gradient descent can get stuck. Log loss is **convex**, so it has a single minimum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""

    1. **Prepare the data:** features $x$ and labels $y$ (0 or 1). Split into train and test sets.
    2. **Initialise:** start with $w = 0$ and $b = 0$.
    3. **Predict:** compute $z = wx + b$ and $p = \sigma(z)$.
    4. **Measure the error:** compute the log loss $J(w, b)$.
    5. **Update:** find the gradients and change $w$ and $b$ a little in the direction that reduces the cost.
    6. **Repeat** steps 3 to 5 until the cost stops decreasing.
    7. **Classify:** predict class 1 if $p \geq 0.5$, else class 0.
    8. **Evaluate** on the test set with a confusion matrix, precision, recall and ROC-AUC.
    """), kind="info", title="Steps of Logistic Regression")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dive into the code...
    ---
    """)
    return


@app.cell
def _(np):
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    def plot_logistic_curve(scores, y_true, threshold=0.5):
        """scores = model.decision_function(X); y_true = actual 0/1 labels"""
        z = np.linspace(scores.min() - 1, scores.max() + 1, 300)
        plt.figure(figsize=(8, 4.5))
        plt.plot(z, 1 / (1 + np.exp(-z)), c="black", lw=2, label="Sigmoid curve")
        plt.scatter(scores, y_true, c=y_true, cmap="coolwarm", edgecolor="k", alpha=0.7, label="Test samples")
        plt.axhline(threshold, c="gray", ls="--", label=f"Threshold = {threshold}")
        plt.xlabel("z = w·x + b (decision score)")
        plt.ylabel("Probability of class 1")
        plt.title("Logistic Curve")
        plt.legend()
        plt.show()

    # plotting roc-auc curve
    def plot_roc_auc(y_true, y_prob):
        """y_prob = model.predict_proba(X)[:, 1]"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.figure(figsize=(5, 4.5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", c="gray", label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()


    return (
        classification_report,
        confusion_matrix,
        plot_logistic_curve,
        plot_roc_auc,
    )


@app.cell
def _():
    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    data = load_breast_cancer()

    pd.DataFrame(data.data, columns=data.feature_names)
    return (data,)


@app.cell
def _(data):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, 
                                                       train_size=0.8, random_state=1)

    return x_test, x_train, y_test, y_train


@app.cell
def _(x_train, y_train):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(x_train, y_train)
    return (pipe,)


@app.cell
def _(y_pred):
    y_pred
    return


@app.cell
def _(pipe, x_test):
    y_pred = pipe.predict(x_test)
    y_prob = pipe.predict_proba(x_test)
    return y_pred, y_prob


@app.cell
def _(y_pred, y_test):
    from sklearn.metrics import accuracy_score

    accuracy_score(y_test, y_pred)
    return


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _(pipe, x_test):
    import numpy as np

    scores = pipe.decision_function(x_test).round(2)

    def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))

    sigmoid(scores).round(2)
    return np, scores


@app.cell
def _(confusion_matrix, y_pred, y_test):
    confusion_matrix(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **How is it different from Linear Regression?**

    | | Linear Regression | Logistic Regression |
    |---|---|---|
    | Predicts | A continuous number | A probability (0 to 1), then a class |
    | Function | Straight line | Sigmoid (S-shaped curve) |
    | Cost function | Mean Squared Error | Log Loss |
    | How it is solved | Direct formula possible | Iterative (gradient descent) |
    | Typical metrics | RMSE, $R^2$ | Accuracy, Precision, Recall, ROC-AUC |

    A straight line can go below 0 or above 1, which makes no sense for a probability. The sigmoid always stays between 0 and 1.

    ### **When to use it?**

    - The answer is **two classes** (yes or no, 0 or 1).
    - You need a **probability**, not just a label.
    - You want **interpretable** weights (effect of each feature on the odds).
    - You want a quick, strong **baseline** model.
    - The classes can be separated reasonably well by a line or a plane.

    **When not to use it:** when the boundary between classes is highly curved (try trees or neural networks), or when classes are perfectly separable without regularisation (the weights keep growing).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **ROC Curve and AUC**

    The model gives probabilities, and the 0.5 threshold is only one choice. The **ROC curve** shows what happens for **every** threshold from 1 down to 0:

    - **TPR (Recall)** = TP / (TP + FN), on the y-axis
    - **FPR** = FP / (FP + TN), on the x-axis

    **AUC** is the area under this curve:

    - **1.0**: perfect model
    - **0.5**: no better than random guessing (the diagonal line)
    - Another meaning: the chance that a random "pass" student gets a higher probability than a random "fail" student.
    """)
    return


@app.cell
def _(plot_roc_auc, y_prob, y_test):
    plot_roc_auc(y_test, y_prob[:,1])
    return


@app.cell
def _(plot_logistic_curve, scores, y_test):
    plot_logistic_curve(scores, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Strengths and Limits**

    - Simple, fast, and works well as a first model.
    - Gives probabilities and easy-to-explain weights.
    - Only draws a **straight** boundary, so it struggles with curved class shapes.
    - Sensitive to outliers and highly correlated features (regularisation helps).
    - With imbalanced classes, accuracy can be misleading, so check precision, recall and AUC.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    Logistic Regression passes a straight-line score through the sigmoid to get a probability, learns its weights by minimising log loss with gradient descent, and is judged with the confusion matrix and ROC-AUC.
    """), kind="success", title="In short")
    return


if __name__ == "__main__":
    app.run()
