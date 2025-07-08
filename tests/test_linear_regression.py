# tests/test_linear_regression.py
import os
import numpy as np
import pandas as pd
import pytest

from linear_regression import LinearRegression
from utils import get_path


def test_compute_min_max_and_normalization():
	model = LinearRegression()
	arr = np.array([5.0, 10.0, 15.0])
	mn, mx = model._compute_min_max(arr)
	assert mn == 5.0 and mx == 15.0
	norm = model._normalize(arr, mn, mx)
	# normalized: (x-5)/(10) -> [0, 0.5, 1]
	assert np.allclose(norm, np.array([0.0, 0.5, 1.0]))
	# edge case: min == max
	arr2 = np.array([7.0, 7.0])
	norm2 = model._normalize(arr2, 7.0, 7.0)
	assert np.all(norm2 == 0)


def test_gradient_descent_simple_line():
	# y = 2x + 1
	x = np.linspace(0, 10, 50)
	y = 2 * x + 1
	model = LinearRegression(learning_rate=0.1, n_iter=2000)
	model.fit(x, y)
	# The learned parameters should be close to (1,2)
	assert pytest.approx(model.theta0, rel=1e-2) == 1.0
	assert pytest.approx(model.theta1, rel=1e-2) == 2.0


def test_predict_array_raises():
	model = LinearRegression()
	model.theta0 = 3.0
	model.theta1 = -0.5
	arr = np.array([0.0, 2.0, 4.0])
	with pytest.raises(TypeError):
		_ = model.predict(arr)


def test_save_and_load(tmp_path, monkeypatch):
	# create model and save
	model = LinearRegression()
	model.theta0 = 5.5
	model.theta1 = -1.25
	dest = tmp_path / "mymodel.csv"
	model.save(csv_path=str(dest))
	assert dest.exists()
	# load into new instance
	loaded = LinearRegression()
	loaded.load(csv_path=str(dest))
	assert loaded.theta0 == pytest.approx(5.5)
	assert loaded.theta1 == pytest.approx(-1.25)


def test_load_missing_and_empty(tmp_path, caplog):
	model = LinearRegression()
	missing = tmp_path / "nofile.csv"
	# missing file: should silently keep zeros
	model.load(csv_path=str(missing))
	assert model.theta0 == 0.0 and model.theta1 == 0.0
	# empty file
	empty = tmp_path / "empty.csv"
	empty.write_text("")
	model.load(csv_path=str(empty))
	assert model.theta0 == 0.0 and model.theta1 == 0.0