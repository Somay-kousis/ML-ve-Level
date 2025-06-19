# Machine Learning Study Notes - Key Terms & Concepts

## Cross-Validation

**What it is:** A technique to evaluate how well a model will perform on unseen data by splitting the dataset multiple times.

**How it works:**

- Split data into k "folds" (usually 5 or 10)
- Train on k-1 folds, test on the remaining fold
- Repeat k times, using each fold as test set once
- Average the results

**Why it's important:** Prevents overfitting and gives a more reliable estimate of model performance than a single train-test split.

**Leave-One-Out Cross-Validation (LOO-CV):** Special case where k = number of samples. Train on all data except one sample, test on that sample. Repeat for each sample.

---

## Regularization

**What it is:** Techniques to prevent overfitting by adding penalties to complex models.

**Common Types:**

- **L1 Regularization (Lasso):** Adds penalty proportional to sum of absolute values of coefficients. Creates sparse models (many coefficients become exactly 0).
- **L2 Regularization (Ridge):** Adds penalty proportional to sum of squared coefficients. Shrinks coefficients but doesn't make them exactly 0.
- **Elastic Net:** Combines L1 and L2 regularization.

**Why it helps:** Forces the model to be simpler, reducing overfitting and improving generalization.

---

## Loss Functions

### Hinge Loss

**What it is:** Loss function used in Support Vector Machines (SVMs). **Formula:** max(0, 1 - y × prediction) **Behavior:**

- Zero loss when prediction is correct with confidence
- Linear penalty for incorrect predictions
- Creates a "margin" around the decision boundary

### Pinball Loss (Quantile Loss)

**What it is:** Loss function used in quantile regression. **Why special:** Instead of predicting the mean, it predicts specific quantiles (like median, 25th percentile, etc.) **Use case:** When you want prediction intervals or care about different parts of the distribution.

### Log Loss (Logistic Loss)

**What it is:** Loss function for logistic regression. **Behavior:** Heavily penalizes confident wrong predictions, encourages probabilistic outputs.

---

## Overfitting vs Underfitting

### Overfitting

- Model learns training data too well, including noise
- High accuracy on training data, poor on new data
- Model is too complex for the amount of data

### Underfitting

- Model is too simple to capture underlying patterns
- Poor performance on both training and test data
- Model lacks capacity to learn the relationship

**The Sweet Spot:** Good generalization - performs well on both training and unseen data.

---

## Bias-Variance Tradeoff

### Bias

- Error from oversimplifying the model
- High bias = underfitting
- Model consistently misses the true relationship

### Variance

- Error from sensitivity to small changes in training data
- High variance = overfitting
- Model changes drastically with different training sets

**Goal:** Find balance between bias and variance to minimize total error.

---

## Feature Engineering

### Polynomial Features

**What it is:** Creating new features by raising existing features to powers or multiplying them together. **Example:** From [x₁, x₂] create [x₁, x₂, x₁², x₁x₂, x₂²] **Why useful:** Allows linear models to capture non-linear relationships.

### Feature Selection

**Purpose:** Choose the most relevant features, remove irrelevant ones. **Benefits:** Reduces overfitting, improves interpretability, faster training.

---

## Model Evaluation Metrics

### For Regression:

- **Mean Squared Error (MSE):** Average of squared differences
- **Mean Absolute Error (MAE):** Average of absolute differences
- **R² Score:** Proportion of variance explained by the model

### For Classification:

- **Accuracy:** Percentage of correct predictions
- **Precision:** Of predicted positives, how many were actually positive
- **Recall:** Of actual positives, how many were predicted positive
- **F1-Score:** Harmonic mean of precision and recall

---

## Hyperparameters vs Parameters

### Parameters

- Learned by the model during training
- Examples: coefficients in linear regression, weights in neural networks

### Hyperparameters

- Set before training begins
- Control the learning process
- Examples: regularization strength (alpha), learning rate, number of trees
- Often tuned using cross-validation

---

## Common Algorithms Overview

### Linear Models

- **Linear Regression:** Fits straight line through data
- **Ridge:** Linear regression + L2 regularization
- **Lasso:** Linear regression + L1 regularization (feature selection)
- **Logistic Regression:** For classification, outputs probabilities

### Ensemble Methods

- **Random Forest:** Many decision trees, vote on final prediction
- **Gradient Boosting:** Builds trees sequentially, each correcting previous errors

### Support Vector Machines (SVM)

- Finds optimal boundary between classes
- Can use different "kernels" for non-linear boundaries

---

## Key Concepts for Robust Models

### Outliers

**What they are:** Data points that are very different from the rest **Problem:** Can heavily influence model training**Solutions:** Robust regression methods (Huber, RANSAC, Theil-Sen)

### Multicollinearity

**What it is:** When input features are highly correlated with each other **Problem:** Makes model unstable and coefficients hard to interpret **Solutions:** Ridge regression, feature selection, dimensionality reduction

---

## Practical Tips

1. **Always split your data:** Train/validation/test or use cross-validation
2. **Start simple:** Begin with basic models before trying complex ones
3. **Visualize your data:** Understanding data distribution helps choose appropriate models
4. **Scale your features:** Many algorithms work better when features are on similar scales
5. **Validate assumptions:** Check if your chosen model's assumptions match your data

---

## Terminology Quick Reference

- **Estimator:** Any object that learns from data (all sklearn models)
- **Transformer:** Changes data (scaling, feature selection)
- **Pipeline:** Chains multiple steps together
- **Sparse:** Most values are zero (common after L1 regularization)
- **Dense:** Most values are non-zero
- **Convergence:** When optimization algorithm stops improving
- **Gradient Descent:** Optimization method that follows the slope to find minimum error

These concepts form the foundation for understanding machine learning. As you read the documentation, refer back to these notes when you encounter unfamiliar terms!