import pandas as pds
import os

from utils import get_path, normalize, denormalize

def get_model_params():

	path = get_path('../model_params/model_params.csv')

	try:
		if not os.path.exists(path):
			raise FileNotFoundError(f"File {path} doesn't exist")
		
		file = pds.read_csv(path)

		expected = {'theta0','theta1'}
		if not expected.issubset(file.columns):
			missing = expected - set(file.columns)
			raise KeyError(f"Missing columns: {', '.join(missing)}")

		row = file.loc[0]
		theta0 = float(row['theta0'])
		theta1 = float(row['theta1'])

		return theta0, theta1
	except Exception as e:
		print(f"Warning: Could not load model params in {path} : {e}")
		return 0.0, 0.0



def predict_price():
	try:
		theta0, theta1 = get_model_params()
	except Exception as e:
		print(f"Error: Failed to load parameters {e}")

	user_str = input("Enter the car’s mileage (km): ")
	try:
		km = float(user_str)
	except ValueError:
		print("Error : please enter a valid number for mileage.")
		return

	price = theta1 * km + theta0

	print(f'estimate price: {price:.2f}€')

if __name__ == '__main__':
	predict_price()