import argparse

from utils import get_path
from linear_regression import LinearRegression

def build_parser(description: str) -> argparse.ArgumentParser:
	"""
	Construct and return an ArgumentParser for the prediction script.

	Parameters:
		description (str): Description to display in the help text.

	Returns:
		argparse.ArgumentParser: Configured parser with the --model option.
	"""

	parser = argparse.ArgumentParser(description=description)
	
	parser.add_argument(
		"-m", "--model",
		type=str,
		default=get_path("model/model.csv"),
		help="The path to the model csv (theta0, theta1)."
	)
	return parser


def predict_price():
	"""
	Load a trained linear regression model and interactively predict
	the price of a car based on user-entered mileage.

	Steps:
	  1. Parse the --model argument to locate the saved theta CSV.
	  2. Load theta0 and theta1 from the CSV (silent fallback to 0.0 if missing).
	  3. Prompt the user for the car's mileage (km).
	  4. Validate the input is a float between 0 and 1e6.
	  5. Compute and print the estimated price.
	"""

	parser = build_parser("Predict the price of a car based on its mileage.")
	args = parser.parse_args()
	
	model = LinearRegression()

	try:
		model.load(csv_path=args.model)
	except Exception as e:
		print(f"Error loading the model: {e}")
		return
	
	try:
		user_str = input("Enter the car's mileage (km): ")
	except KeyboardInterrupt:
		print("\nAborted by user.")
		return

	try:
		km = float(user_str)
	except ValueError:
		print('Error: Please enter a valid number for mileage.')
		return
	
	if not (0 <= km <= 1e6):
		print("Error: mileage must be between 0 and 1 million km.")
		return
	
	price = model.predict(km)
	print(f"Estimate price: {price:.2f}€")

if __name__ == '__main__':
	predict_price()