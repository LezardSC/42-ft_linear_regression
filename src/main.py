import os
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt

def draw_data(x, y):
	fig, ax = plt.subplots()
	ax.set_xlabel('km')
	ax.set_ylabel('price')
	ax.set_title('linear regression')
	ax.scatter(x, y, alpha=0.6)
	plt.tight_layout()
	plt.show()

def parse_data():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(script_dir, '../data/data.csv')
	data = pds.read_csv(csv_path)
	
	km = data[['km']].values
	price = data[['price']].values
	km_scaled = (km - km.min(axis=0)) / (km.max(axis=0) - km.min(axis=0))
	price_scaled = (price - price.min(axis=0)) / (price.max(axis=0) - price.min(axis=0))
	km_unscaled = km_scaled * (km.max(axis=0) - km.min(axis=0)) + km.min(axis=0)
	price_unscaled = price_scaled * (price.max(axis=0) - price.min(axis=0)) + price.min(axis=0)
	# print(km)
	# print(price)
	# print(f'my price scaled: {price_scaled}')
	
	# draw_data(km, price)

# y = theta0 + (theta1 * x)

def main():
	parse_data()

if __name__ == '__main__':
	main()
