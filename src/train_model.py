from utils import get_path
from linear_regression import LinearRegression
import pandas as pds

def parse_data():
	csv_path = get_path('../data/data.csv')
	data = pds.read_csv(csv_path)
	
	x = data.iloc[:, 0].values.astype(float)
	y = data.iloc[:, 1].values.astype(float)

	return x, y

def main():
	x, y = parse_data()

	model = LinearRegression(learning_rate=0.1, n_iter=5000)
	model.fit(x, y)
	model.save()
	model.draw_data(x, y)



if __name__ == '__main__':
	main()