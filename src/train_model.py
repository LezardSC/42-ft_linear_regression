from utils import get_path

def train_model():
	theta0 = 1.0
	theta1 = 2.0
	km_min = 0
	km_max = 0
	price_min = 0
	price_max = 0

	csv_path = get_path('../theta/theta.csv')

	with open(csv_path, 'w') as f:
		f.write(f"theta0,theta1,km_min,km_max,price_min,price_max\n{theta0},{theta1}")
	return