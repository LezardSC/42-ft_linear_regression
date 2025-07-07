import os

def get_path(file_name):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(script_dir, file_name)

	return path