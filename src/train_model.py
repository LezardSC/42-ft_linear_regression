import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import os

from utils import get_path, normalize, compute_min_max

def draw_data(x, y):
	fig, ax = plt.subplots()
	ax.set_xlabel('km')
	ax.set_ylabel('price')
	ax.set_title('linear regression')
	ax.scatter(x, y, alpha=0.6)
	plt.tight_layout()
	plt.show()

def export_model_params(theta0, theta1, km_min, km_max, price_min, price_max):
	csv_path = get_path('../model_params/model_params.csv')

	os.makedirs(os.path.dirname(csv_path), exist_ok=True)

	file = pds.DataFrame([{
		'theta0':    theta0,
		'theta1':    theta1,
		'km_min':    km_min,
		'km_max':    km_max,
		'price_min': price_min,
		'price_max': price_max
	}])
	file.to_csv(csv_path, index=False)


def parse_data():
	csv_path = get_path('../data/data.csv')
	data = pds.read_csv(csv_path)
	
	km = data[['km']].values.astype(float)
	price = data[['price']].values.astype(float)

	return km, price

def gradient_descent():
	return 1.0, 2.0

def train_model():
	learning_rate = 0
	n_iterations = 0
	km, price = parse_data()
	km_min, km_max = compute_min_max(km)
	price_min, price_max = compute_min_max(price)
	km_norm    = normalize(km, km_min, km_max)
	price_norm = normalize(price, price_min, price_max)
	theta0, theta1 = gradient_descent()
	export_model_params(theta0, theta1, km_min, km_max, price_min, price_max)
	draw_data(km, price)
	return


if __name__ == '__main__':
    train_model()