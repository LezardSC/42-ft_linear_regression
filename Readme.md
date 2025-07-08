# ft\_linear\_regression

An introduction to machine learning: a simple univariate linear regression model predicting car prices based on mileage.

## Table of Contents

* [Introduction](#introduction)
* [How it Works: Conceptual Overview](#how-it-works-conceptual-overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)
	* [Training the Model](#training-the-model)
	* [Predicting Prices](#predicting-prices)
* [Project Structure](#project-structure)
* [Running Tests](#running-tests)
* [Contributing](#contributing)
* [License](#license)

## Introduction

This repository contains a simple implementation of linear regression from scratch, using gradient descent and min–max normalization. The goal is to predict a target value *y* (e.g., car price) from a single feature *x* (e.g., mileage).

## How It Works: Conceptual Overview

This section outlines the core process for fitting a linear regression model to data:

1. **Model Representation**

   * A straight line is used to approximate the relationship between mileage (`x`) and price (`y`).
   * The line is defined by two parameters:

	 * **Intercept (`θ₀`)**: Value of `y` when `x` is zero.
	 * **Slope (`θ₁`)**: Rate of change in `y` for each unit change in `x`.

2. **Error Quantification**

   * Residuals are calculated for each data point as:

	 ```text
	 residual_i = (θ₀ + θ₁·x_i) - y_i
	 ```
   * These residuals are squared to form the **cost function**:

	 ```text
	 J(θ₀, θ₁) = (1/(2m)) * Σᵢ residual_i²
	 ```
	* Corresponding implementation in `_gradient_descent`:
		```python
		# residual and gradient calculation
		estimate_y   = theta1 * x + theta0	  # predicted values
		cost_function = estimate_y - y		   # residuals

		# gradient computation (derivative of squared-error cost)
		gradient0 = (1/m) * cost_function.sum()
		gradient1 = (1/m) * (cost_function * x).sum()

		# parameter updates
		theta0 -= learning_rate * gradient0
		theta1 -= learning_rate * gradient1
		```
   * Although the implementation does not explicitly compute `residual²`, the gradient formulas derive from this squared-error cost.
   * Averaging these squared residuals yields a single scalar reflecting overall fit quality.

3. **Optimization via Gradient Descent**

   * Initial parameter values (both set to zero) are iteratively updated to minimize mean squared error.
   * Each iteration computes the error gradient with respect to each parameter and adjusts parameters by a fraction (learning rate) of that gradient.
   * Iterations continue until the error reduction stabilizes.

4. **Feature Scaling**

   * Mileage and price values are scaled into the \[0, 1] range using min–max normalization.
   * Normalization ensures consistent parameter updates and accelerates convergence.
   * After optimization on normalized data, parameters are transformed back to the original scale.

This workflow produces a linear function capable of estimating car price from mileage input.


## Prerequisites

* Python 3.7 or higher
* [pip](https://pip.pypa.io/en/stable/) for package management

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd 42-ft_linear_regression
   python3 -m venv <env-name>
   source <env-name>/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Your data should be a CSV file with two columns (no header assumptions beyond first two columns):

1. **Mileage** (e.g., kilometers)
2. **Price** (e.g., euros)

Example (`data/data.csv`):

```csv
x,y
10000,12000
25000,9000
...
```

## Usage

### Training the Model

Train the linear regression model with your dataset:

```bash
python train_model.py \
  --data path/to/data.csv \
  --model path/to/model.csv \
  --learning-rate 0.1 \
  --iter 5000 \
  [--plot] [--evaluate]
```

* `--data` (`-d`): Path to the training data CSV (default: `data/data.csv`).
* `--model` (`-m`): Path where the trained parameters (theta0, theta1) will be saved (default: `model/model.csv`).
* `--learning-rate` (`-lr`): Gradient descent step size (0.001–1.0, default: 0.1).
* `--iter` (`-i`): Number of gradient descent iterations (1–50000, default: 5000).
* `--plot` (`-p`): Display a scatter plot of the data and the fitted regression line.
* `--evaluate` (`-e`): Print performance metrics (R², RMSE, MAE) after training.

After training, the model parameters are saved to the specified CSV.

### Evaluating Model Performance

If you prefer a standalone evaluation script, you can run:
```bash
python evaluate_model.py \
  --data path/to/data.csv \
  --model path/to/model.csv \
```

This will load the same data and model parameters.
There is no real point in using this program instead of `train_model.py --evaluate` with but the subject ask for a program to evaluate the model, so I did it.

### Predicting Prices

Use the trained model to estimate car prices:

```bash
python predict_price.py \
  --model path/to/model.csv
```

You will be prompted to enter a mileage value (0 to 1,000,000). The script loads `theta0` and `theta1` from the model CSV and prints the estimated price.

## Project Structure

```
ft_linear_regression/
├── data/
│   └── data.csv              # Sample or user-provided dataset
├── model/
│   └── model.csv             # Trained model parameters (theta0, theta1)
├── src/
│	├── linear_regression.py   # Core implementation of LinearRegression class
│	├── train_model.py		   # CLI for training the model
│	├── predict_price.py		# CLI for predicting prices
│	├── evaluate_model.py   # CLI for standalone evaluation
│	└── utils.py				# Path helper
├── tests/
│   ├── test_cli.py		 # End-to-end CLI tests
│   └── test_linear_regression.py # Unit tests for algorithm
├── pytest.ini # Used to say pytest where the source files are
└── requirements.txt		# Python dependencies
```

## Running Tests

Run all tests using `pytest`:

```bash
python3 -m pytest
```

Ensure all tests pass before submitting or merging.

## Contributing

Feel free to open issues or pull requests for improvements. For major changes, please open an issue first to discuss what you’d like to change.

## License

This project is part of the **42 School curriculum** and follows its evaluation and submission guidelines.

## Author

login: jrenault