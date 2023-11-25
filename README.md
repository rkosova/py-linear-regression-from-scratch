# Linear Regression with Batch Gradient Descent

This project implements a simple linear regression model using batch gradient descent in Python, with the help of the NumPy library.

## Overview

The `LinearRegression` class includes methods for training the model and making predictions. The training process uses batch gradient descent to optimize the model parameters (`w` and `b`) based on the Mean Squared Error (MSE) loss.

## Usage

1. **Initialization:**
   - Create an instance of the `LinearRegression` class.

   ```python
   model = LinearRegression()
   ```

2. **Training:**
   - Train the model by providing training data (`X_train` and `y_train`), batch size, and learning rate.

   ```python
   X_train = # your input features
   y_train = # corresponding target values
   batch_size = # choose an appropriate batch size
   learning_rate = # choose a suitable learning rate

   model.train(X_train, y_train, batch_size, learning_rate)
   ```

   The `train` method performs batch gradient descent to optimize the model parameters.

3. **Prediction:**
   - Use the trained model to make predictions on new data.

   ```python
   X_test = # new input features
   predictions = model.predict(X_test)
   ```

   The `predict` method computes predictions based on the trained model parameters.

## Parameters

- `X_train`: Input features for training.
- `y_train`: Target values for training.
- `batch_size`: Batch size for batch gradient descent.
- `lr`: Learning rate for the optimization process.

## Example

```python
# Example usage
model = LinearRegression()

# Generate synthetic data
X_train = np.random.rand(100)
y_train = 2.5 * X_train + 1.0 + np.random.normal(0, 0.5, size=100)

# Train the model
batch_size = 10
learning_rate = 0.01
model.train(X_train, y_train, batch_size, learning_rate)

# Make predictions
X_test = np.random.rand(10)
predictions = model.predict(X_test)
print("Predictions:", predictions)
```

Adjust the hyperparameters and data according to your specific use case.

## Dependencies

- NumPy: Install via `pip install numpy`
