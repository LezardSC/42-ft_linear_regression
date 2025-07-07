import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import os

from utils import get_path

class LinearRegression:
	def __init__(self, learning_rate=0.1, n_iter=1000):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.theta0 = 0.0
		self.theta1 = 0.0
		self.x_min = None
		self.x_max = None
		self.y_min = None
		self.y_max = None
	
	def _compute_min_max(self, values):
		return values.min(), values.max()

	def _normalize(self, values, min_values, max_values):
		return (values - min_values) / (max_values - min_values)
	
	def _denormalize_thetas(self, theta0_norm, theta1_norm):
		scale_x = self.x_max - self.x_min
		scale_y = self.y_max - self.y_min

		theta1 = theta1_norm * (scale_y / scale_x)
		theta0 = theta0_norm * scale_y - theta1 * self.x_min + self.y_min
		return theta0, theta1
	
	def _gradient_descent(self, x, y):
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
		self.x_min, self.x_max = self._compute_min_max(x)
		self.y_min, self.y_max = self._compute_min_max(y)

		x_norm = self._normalize(x, self.x_min, self.x_max)
		y_norm = self._normalize(y, self.y_min, self.y_max)

		theta0_norm, theta1_norm = self._gradient_descent(x_norm, y_norm)
		self.theta0, self.theta1 = self._denormalize_thetas(theta0_norm, theta1_norm)

	def draw_data(self, x, y):
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
		if csv_path is None:
			csv_path = get_path('../thetas/thetas.csv')

		os.makedirs(os.path.dirname(csv_path), exist_ok=True)

		file = pds.DataFrame([{
			'theta0': self.theta0,
			'theta1': self.theta1,
		}])
		file.to_csv(csv_path, index=False)

	def load(self, csv_path=None):
		if csv_path is None:
			csv_path = get_path('../thetas/thetas.csv')
		
		file = pds.read_csv(csv_path)
		
		if not {'theta0', 'theta1'}.issubset(file.columns):
			raise KeyError("CSV file must contain 'theta0' and 'theta1' columns.")

		row = file.loc[0]
		self.theta0 = float(row['theta0'])
		self.theta1 = float(row['theta1'])

	def predict(self, x):
		return float(self.theta0 + self.theta1 * x)