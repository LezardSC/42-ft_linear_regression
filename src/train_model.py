import pandas as pds
import argparse

from utils import get_path
from linear_regression import LinearRegression


def check_iter(value: str) -> int:
	"""
	Validate and convert the iteration count.

	Parameters:
		value (str): The raw string input for number of iterations.

	Returns:
		int: The validated number of iterations between 1 and 50000.

	Raises:
		argparse.ArgumentTypeError: If conversion fails or value is out of range.
	"""

	try:
		int_value = int(value)
	except ValueError:
		raise argparse.ArgumentTypeError(f"Iterations has to be an integer, not '{value}'")
	if not (1 <= int_value <= 50000):
		raise argparse.ArgumentTypeError("The number of iterations has to be between 1 and 50000")
	return int_value


def check_lr(value: str) -> float:
	"""
	Validate and convert the learning rate.

	Parameters:
		value (str): The raw string input for learning rate.

	Returns:
		float: The validated learning rate between 0.001 and 1.0.

	Raises:
		argparse.ArgumentTypeError: If conversion fails or value is out of range.
	"""

	try:
		float_value = float(value)
	except ValueError:
		raise argparse.ArgumentTypeError(f"The learning rate has to be a number, not '{value}'")
	if not (0.001 <= float_value <= 1.0):
		raise argparse.ArgumentTypeError("The learning rate has to be between 0.001 and 1.0")
	return float_value


def build_parser(description: str) -> argparse.ArgumentParser:
	"""
	Construct the command-line argument parser for the training script.

	Parameters:
		description (str): Text to display at the top of the help message.

	Returns:
		argparse.ArgumentParser: Configured parser with data, model, lr, iter, and plot options.
	"""
	
	parser = argparse.ArgumentParser(description=description)
	
	parser.add_argument(
		"-d", "--data",
		type=str,
		default=get_path("data/data.csv"),
		help="The path to the data training csv (x, y)"
	)
	parser.add_argument(
		"-m", "--model",
		type=str,
		default=get_path("model/model.csv"),
		help="The path to the model csv (theta0, theta1)."
	)
	parser.add_argument(
		"-lr", "--learning-rate",
		dest="learning_rate",
		type=check_lr,
		default=0.1,
		metavar="[0.001-1.0]",
		help="The learning rate for the gradient descent."
	)
	parser.add_argument(
		"-i", "--iter",
		dest="n_iter",
		type=check_iter,
		default=5000,
		metavar="[1, 50000]",
		help="The number of iterations for the gradient descent."
	)
	parser.add_argument(
		"-p", "--plot",
		action="store_true",
		dest="draw_linear_regression",
		help="Draw a graph of the linear regression."
	)
	parser.add_argument(
		"-e", "--evaluate",
		action="store_true",
		dest="evaluate_prediction",
		help="Evaluate the precision of the linear regression."
	)
	return parser


def main():
	"""
	Load training data, fit a LinearRegression model, save its parameters,
	and optionally display the regression plot.
	"""

	parser = build_parser("Train a linear regression model.")
	args = parser.parse_args()

	try:
		df = pds.read_csv(args.data)
	except FileNotFoundError:
		print(f"Error: File not found ({args.data}).")
		return
	except pds.errors.ParserError as e:
		print(f"Error: Can't parse the CSV: {e}")
		return
	
	if df.shape[1] < 2:
		print("Error: The CSV should have at least two columns.")
		return
	
	x = df.iloc[:, 0].to_numpy(dtype=float)
	y = df.iloc[:, 1].to_numpy(dtype=float)

	model = LinearRegression(learning_rate=args.learning_rate, n_iter=args.n_iter)
	try:
		model.fit(x, y)
	except ValueError as e:
		print(f"Error fitting model: {e}")
		return

	model.save(csv_path=args.model)
	if args.draw_linear_regression:
		model.draw_data(x, y)
	if args.evaluate_prediction:
		try:
			model.evaluate(x, y)
		except ValueError as e:
			print(f"Error evaluating model: {e}")
			return

if __name__ == '__main__':
	main()