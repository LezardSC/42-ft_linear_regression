import pandas as pds
import argparse
import numpy as np
from utils import get_path
from linear_regression import LinearRegression

def build_parser():
	parser = argparse.ArgumentParser("Evaluate linear regression model")
	parser.add_argument("-d", "--data", type=str,
						default=get_path("data/data.csv"),
						help="Path to (x, y) data CSV")
	parser.add_argument("-m", "--model", type=str,
						default=get_path("model/model.csv"),
						help="Path to saved model CSV")
	return parser

def main():
	args = build_parser().parse_args()
	df = pds.read_csv(args.data)
	x = df.iloc[:,0].to_numpy(dtype=float)
	y = df.iloc[:,1].to_numpy(dtype=float)

	model = LinearRegression()
	model.load(csv_path=args.model)

	try:
		metrics = model.evaluate(x, y)
	except ValueError as e:
		print(f"Error evaluating model: {e}")
		return
	print("Model performance on dataset:")
	print(f"R²:   {metrics['r2']:.4f}")
	print(f"RMSE: {metrics['rmse']:.2f}")
	print(f"MAE:  {metrics['mae']:.2f}")

if __name__ == "__main__":
	main()
