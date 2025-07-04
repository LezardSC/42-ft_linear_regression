import os 

def normalize(n, min, max):
	return (n - min) / (max - min)

def denormalize_thetas(theta0_norm, theta1_norm, x_min, x_max, y_min, y_max):
	scale_x = x_max - x_min
	scale_y = y_max - y_min

	theta1 = theta1_norm * (scale_y / scale_x)
	theta0 = theta0_norm * scale_y - theta1 * x_min + y_min
	return theta0, theta1

def get_path(file_name):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(script_dir, file_name)

	return path

def compute_min_max(n):
	return n.min(), n.max()