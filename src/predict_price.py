from linear_regression import LinearRegression
from utils import get_path

def predict_price():
	model = LinearRegression()

	try:
		model.load()
	except Exception as e:
		print(f"Error loading parameters: {e}")
		return
	
	user_str = input("Enter the car's mileage (km): ")
	try:
		km = float(user_str)
	except ValueError:
		print('Error: Please enter a valid number for mileage.')
		return
	
	price = model.predict(km)
	print(f"estimate price: {price:.2f}€")

if __name__ == '__main__':
	predict_price()