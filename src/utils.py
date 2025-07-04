import os 

def normalize(n, min, max):
	return (n - min) / (max - min)

def denormalize(n_norm, min, max):
	return n_norm * (max - min) + min

def get_path(file_name):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(script_dir, file_name)

	return csv_path

def compute_min_max(n):
	return n.min(), n.max()