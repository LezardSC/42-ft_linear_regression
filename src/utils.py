import os 

def normalize(n, min, max):
	return (n - min) / (max - min)

def denormalize(n_norm, min, max):
	return n_norm * (max - min) + min

def denormalize_thetas(theta0_norm, theta1_norm, km_min, km_max, price_min, price_max):
	scale_km = km_max - km_min
	scale_price = price_max - price_min

	theta1 = theta1_norm * (scale_price / scale_km)
	theta0 = theta0_norm * scale_price - theta1 * km_min + price_min
	return theta0, theta1

def get_path(file_name):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(script_dir, file_name)

	return path

def compute_min_max(n):
	return n.min(), n.max()