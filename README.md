# 🚀 Machine Learning Algorithms From Scratch (Regression & Classification)

A comprehensive, end-to-end machine learning library built entirely from scratch using **Python and NumPy**. This project bypasses black-box library abstractions to implement core algorithms from first principles—covering cost functions, vectorized parameter updates, feature expansion, hyperparameter tuning, and real-world dataset evaluation.

---

## 💡 Key Highlights

* **Pure NumPy & Vectorization:** Avoided slow explicit loops in favor of vectorized matrix math for high-performance parameter updates across multi-feature datasets.
* **Robust Numerical Stability:** Engineered internal standardization (`StandardScaler`) post-feature expansion to eliminate `NaN` overflow errors during high-degree polynomial iterations.
* **Parsimonious Hyperparameter Tuning (Occam’s Razor):** Automated validation loops evaluating polynomial degrees ($d \in \{1, 2, 3, 4, 5\}$) with a strict improvement threshold to select optimal model complexity without overfitting.
* **Rigorously Benchmarked:** Validated all custom scratch models against Scikit-Learn production pipelines (`Pipeline`, `PolynomialFeatures`, `LinearRegression`, `LogisticRegression`) to ensure mathematical parity.
* **Professional Visualization:** Generated custom loss curves, training vs. validation performance charts, and multi-dimensional decision boundary contours using Matplotlib.

---

## 🏗️ Project Architecture & Modular Structure

```text
├── 01_Regression/
│   ├── 01_simple_linear_regression.ipynb        # Single-feature fitting, cost function, gradient descent
│   ├── 02_multiple_linear_regression.ipynb      # Multi-feature vectorization & Real-world House Price Prediction
│   └── 03_polynomial_regression.ipynb           # Non-linear feature expansion & parsimonious tuning
│
├── 02_Logistic_Regression/
│   ├── 01_simple_logistic_regression.ipynb      # Binary classification, sigmoid function & decision boundaries
│   ├── 02_multiple_logistic_regression.ipynb    # Multi-feature classification engine
│   └── 03_polynomial_logistic_regression.ipynb  # Curved decision boundaries & validation-based tuning
│
├── README.md
└── requirements.txt

