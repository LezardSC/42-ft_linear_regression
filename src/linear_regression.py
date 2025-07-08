import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import os

from utils import get_path

class LinearRegression:
	"""
	Simple univariate linear regression using min-max normalization and gradient descent.

	Attributes:
		learning_rate (float): Step size for gradient descent updates.
		n_iter (int): Number of iterations for gradient descent.
		theta0 (float): Intercept term after training.
		theta1 (float): Slope term after training.
		x_min (float): Minimum of training feature values.
		x_max (float): Maximum of training feature values.
		y_min (float): Minimum of training target values.
		y_max (float): Maximum of training target values.

	Methods:
		fit(x, y): Train the model on feature array x and target y.
		predict(x): Predict target values for given feature(s).
		save(csv_path): Save learned parameters to CSV.
		load(csv_path): Load parameters from CSV, silent if missing or empty.
		draw_data(x, y): Plot data points and fitted regression line.
	"""

	def __init__(self, learning_rate:float = 0.1, n_iter: int = 1000):
		"""
		Initialize the linear regression model parameters.

		Parameters:
			learning_rate (float): Learning rate for gradient descent.
			n_iter (int): Number of iterations to run gradient descent.
		"""

		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.theta0 = 0.0
		self.theta1 = 0.0
		self.x_min = None
		self.x_max = None
		self.y_min = None
		self.y_max = None
	
	def _compute_min_max(self, values: np.ndarray) -> tuple:
		"""
		Compute the minimum and maximum of an array.

		Parameters:
			values (np.ndarray): Input numeric array.

		Returns:
			tuple: (min_value, max_value) of the array.
		"""

		return values.min(), values.max()

	def _normalize(self, values, min_values, max_values):
		"""
		Apply min-max normalization to an array.

		Parameters:
			values (np.ndarray): Original values.
			min_values (float): Minimum for normalization.
			max_values (float): Maximum for normalization.

		Returns:
			np.ndarray: Normalized values in [0, 1].
		"""

		if min_values == max_values:
			return np.zeros_like(values)
		return (values - min_values) / (max_values - min_values)
	
	def _denormalize_thetas(self, theta0_norm, theta1_norm):
		"""
		Convert normalized parameters back to original scale.

		Parameters:
			theta0_norm (float): Intercept on normalized data.
			theta1_norm (float): Slope on normalized data.

		Returns:
			tuple: (theta0, theta1) on original data scale.
		"""

		scale_x = self.x_max - self.x_min
		scale_y = self.y_max - self.y_min

		theta1 = theta1_norm * (scale_y / scale_x)
		theta0 = theta0_norm * scale_y - theta1 * self.x_min + self.y_min
		return theta0, theta1
	
	def _gradient_descent(self, x, y):
		"""
		Perform gradient descent on normalized data to learn parameters.

		Parameters:
			x (np.ndarray): Normalized feature array.
			y (np.ndarray): Normalized target array.

		Returns:
			tuple: (theta0, theta1) after training on normalized data.
		"""

		theta0 = 0.0
		theta1 = 0.0
		m = len(x)

		for _ in range(self.n_iter):
			estimate_y = theta1 * x + theta0
			cost_function = estimate_y - y

			gradient0 = (1 / m) * cost_function.sum()
			gradient1 = (1 / m) * (cost_function * x).sum()

			theta0 -= self.learning_rate * gradient0
			theta1 -= self.learning_rate * gradient1

		return theta0, theta1

	def fit(self, x, y):
		"""
		Train the linear regression model on provided data.

		Parameters:
			x (np.ndarray): Feature values.
			y (np.ndarray): Target values.

		Raises:
			ValueError: If x or y is empty.
		"""

		if x.size == 0 or y.size == 0:
			raise ValueError("x and y should contain at least one value.")
		
		self.x_min, self.x_max = self._compute_min_max(x)
		self.y_min, self.y_max = self._compute_min_max(y)

		x_norm = self._normalize(x, self.x_min, self.x_max)
		y_norm = self._normalize(y, self.y_min, self.y_max)

		theta0_norm, theta1_norm = self._gradient_descent(x_norm, y_norm)
		self.theta0, self.theta1 = self._denormalize_thetas(theta0_norm, theta1_norm)

	def draw_data(self, x, y):
		"""
		Plot the data points and the fitted regression line.

		Parameters:
			x (np.ndarray): Feature values.
			y (np.ndarray): Target values.
		"""

		fig, ax = plt.subplots()
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_title('linear regression')

		ax.scatter(x, y, alpha=0.6)

		x_min, x_max = self._compute_min_max(x)
		xs = np.linspace(x_min, x_max, 100)
		ys = self.theta1 * xs + self.theta0
		ax.plot(xs, ys, color='red', linewidth=1.5)

		plt.tight_layout()
		plt.show()

	def save(self, csv_path=None):
		"""
		Save learned theta parameters to a CSV file.

		Parameters:
			csv_path (str, optional): Path to save the CSV. Defaults to 'model/model.csv'.
		"""

		if csv_path is None:
			csv_path = get_path('model/model.csv')

		os.makedirs(os.path.dirname(csv_path), exist_ok=True)

		file = pds.DataFrame([{
			'theta0': self.theta0,
			'theta1': self.theta1,
		}])
		file.to_csv(csv_path, index=False)

	def load(self, csv_path=None):
		"""
		Load theta parameters from a CSV file.

		If the file is missing or empty, does nothing (keeps theta0/theta1 at 0).

		Parameters:
			csv_path (str, optional): Path to load the CSV. Defaults to 'model/model.csv'.

		Raises:
			ValueError: If CSV is malformed.
			PermissionError: If file cannot be read.
			KeyError: If required columns are missing.
		"""

		if csv_path is None:
			csv_path = get_path('model/model.csv')
		
		try:
			file = pds.read_csv(csv_path)
		except (FileNotFoundError, pds.errors.EmptyDataError):
			return
		except pds.errors.ParserError as e:
			raise ValueError(f"CSV bad format: {e}")
		except PermissionError as e:
			raise PermissionError(f"Can't read the file (permission) : {e}")
		
		if not {'theta0', 'theta1'}.issubset(file.columns):
			raise KeyError("CSV file must contain 'theta0' and 'theta1' columns.")

		try:
			row = file.loc[0]
		except (IndexError, KeyError):
			return

		try:
			self.theta0 = float(row['theta0'])
			self.theta1 = float(row['theta1'])
		except ValueError as e:
			raise ValueError(f"Can't convert θ in number : {e}")

	def predict(self, x: float):
		"""
		Predict target value for given input using the trained model.

		Parameters:
			x (float)

		Returns:
			float
		"""

		return float(self.theta0 + self.theta1 * x)