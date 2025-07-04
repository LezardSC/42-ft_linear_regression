import numpy as np
import pandas as pds
import matplotlib.pyplot as plt

from utils import get_path, normalize

def draw_data(x, y):
	fig, ax = plt.subplots()
	ax.set_xlabel('km')
	ax.set_ylabel('price')
	ax.set_title('linear regression')
	ax.scatter(x, y, alpha=0.6)
	plt.tight_layout()
	plt.show()

def load_model_params(theta0, theta1, km_min, km_max, price_min, price_max):
	csv_path = get_path('../model_params/model_params.csv')

	with open(csv_path, 'w') as f:
		f.write(f"theta0,theta1,km_min,km_max,price_min,price_max\n{theta0},{theta1}")


def parse_data():
	csv_path = get_path('../data/data.csv')
	data = pds.read_csv(csv_path)
	
	km = data[['km']].values.astype(float)
	price = data[['price']].values.astype(float)

	return km, price


def train_model():
	km, price = parse_data()
	theta0 = 1.0
	theta1 = 2.0

	km_min, km_max = km.min(), km.max()
	price_min, price_max = price.min(), price.max()

	km_norm    = normalize(km, km_min, km_max)
	price_norm = normalize(price, price_min, price_max)

	load_model_params(theta0, theta1, km_min, km_max, price_min, price_max)
	draw_data(km, price)
	return


if __name__ == '__main__':
    train_model()