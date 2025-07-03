import os
import pandas as pds
from sklearn.preprocessing import MinMaxScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../data/data.csv')

def parse_data():
	data = pds.read_csv(csv_path)
	
	scaler = MinMaxScaler()
	km = data[['km']].values
	price = data[['price']].values
	scaler.fit(price)
	price_scaled = scaler.transform(price)

	print(km)
	print(price)
	print(f'price scaled: {price_scaled}')

if __name__ == '__main__':
	parse_data()