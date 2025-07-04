import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import os

from utils import get_path, normalize, compute_min_max, denormalize_thetas

def draw_data(x, y, theta0, theta1):
	fig, ax = plt.subplots()

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('linear regression')

	ax.scatter(x, y, alpha=0.6)

	x_min, x_max = x.min(), x.max()
	xs = np.linspace(x_min, x_max, 100)
	ys = theta1 * xs + theta0
	ax.plot(xs, ys, color='red', linewidth=1.5)

	plt.tight_layout()
	plt.show()

def export_model_params(theta0, theta1):
	csv_path = get_path('../model_params/model_params.csv')

	os.makedirs(os.path.dirname(csv_path), exist_ok=True)

	file = pds.DataFrame([{
		'theta0':    theta0,
		'theta1':    theta1
	}])
	file.to_csv(csv_path, index=False)


def parse_data():
	csv_path = get_path('../data/data.csv')
	data = pds.read_csv(csv_path)
	
	x = data.iloc[:, 0].values.astype(float)
	y = data.iloc[:, 1].values.astype(float)

	return x, y

def gradient_descent(x_norm, y_norm):
	x_norm = x_norm.ravel()
	y_norm = y_norm.ravel()
	learning_rate = 0.1
	n_iterations = 1000
	m = len(x_norm)
	theta0, theta1 = 0.0, 0.0

	for _ in range(n_iterations):
		estimate_y = theta1 * x_norm + theta0
		cost_function = estimate_y - y_norm

		gradient0 = (1 / m) * cost_function.sum()
		gradient1 = (1 / m) * (cost_function * x_norm).sum()

		theta0 -= learning_rate * gradient0
		theta1 -= learning_rate * gradient1
	
	return theta0, theta1

def train_model():
	x, y = parse_data()
	x_min, x_max = compute_min_max(x)
	y_min, y_max = compute_min_max(y)
	x_norm = normalize(x, x_min, x_max)
	y_norm = normalize(y, y_min, y_max)
	theta0, theta1 = gradient_descent(x_norm, y_norm)
	theta0, theta1 = denormalize_thetas(theta0, theta1, x_min, x_max, y_min, y_max)
	export_model_params(theta0, theta1)
	draw_data(x, y, theta0, theta1)
	return


if __name__ == '__main__':
    train_model()