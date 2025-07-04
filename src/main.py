import numpy as np
import pandas as pds
import matplotlib.pyplot as plt

from predict_price import predict_price
from train_model import train_model
from utils import normalize
from utils import denormalize
from utils import get_path

def draw_data(x, y):
	fig, ax = plt.subplots()
	ax.set_xlabel('km')
	ax.set_ylabel('price')
	ax.set_title('linear regression')
	ax.scatter(x, y, alpha=0.6)
	plt.tight_layout()
	plt.show()

def parse_data():
	csv_path = get_path('../data/data.csv')
	data = pds.read_csv(csv_path)
	
	km = data[['km']].values
	price = data[['price']].values

	return km, price
	

def main():
	km, price = parse_data()
	theta0 = 0
	theta1 = 0

	km_scaled = normalize(km)
	price_scaled = normalize(price)

	# predict_price(theta0, theta1, km)
	train_model()

	# draw_data(km, price)

	# km_min, km_max = km.min(axis=0), km.max(axis=0)
	# price_min, price_max = price.min(axis=0), price.max(axis=0)
	# km_unscaled = denormalize(km_scaled, km.min(axis=0), km.max(axis=0))
	# price_unscaled = denormalize(price_scaled, price.min(axis=0), price.max(axis=0))

if __name__ == '__main__':
	main()
