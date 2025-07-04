import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import os

from utils import get_path, normalize, compute_min_max, denormalize, denormalize_thetas

def draw_data(x, y, theta0, theta1):
	fig, ax = plt.subplots()

	ax.set_xlabel('km')
	ax.set_ylabel('price')
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
	
	km = data[['km']].values.astype(float)
	price = data[['price']].values.astype(float)

	return km, price

def gradient_descent(km_norm, price_norm):
	km_norm = km_norm.ravel()
	price_norm = price_norm.ravel()
	learning_rate = 0.1
	n_iterations = 1000
	m = len(km_norm)
	theta0, theta1 = 0.0, 0.0

	for _ in range(n_iterations):
		estimate_price = theta1 * km_norm + theta0
		cost_function = estimate_price - price_norm

		gradient0 = (1 / m) * cost_function.sum()
		gradient1 = (1 / m) * (cost_function * km_norm).sum()

		theta0 -= learning_rate * gradient0
		theta1 -= learning_rate * gradient1
	
	return theta0, theta1

def train_model():
	km, price = parse_data()
	km_min, km_max = compute_min_max(km)
	price_min, price_max = compute_min_max(price)
	km_norm = normalize(km, km_min, km_max)
	price_norm = normalize(price, price_min, price_max)
	theta0, theta1 = gradient_descent(km_norm, price_norm)
	theta0, theta1 = denormalize_thetas(theta0, theta1, km_min, km_max, price_min, price_max)
	export_model_params(theta0, theta1)
	draw_data(km, price, theta0, theta1)
	return


if __name__ == '__main__':
    train_model()