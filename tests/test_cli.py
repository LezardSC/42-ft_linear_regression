import os
import csv
import argparse
import pandas as pd
import pytest
import builtins
from pathlib import Path
import sys

# Import the CLI functions
import train_model as tm
import predict_price as pp

# Helpers to isolate file paths
@pytest.fixture(autouse=True)
def clear_env(tmp_path, monkeypatch):
	# Redirect default paths under tmp_path
	# data.csv
	data_dir = tmp_path / "data"
	data_dir.mkdir()
	sample_data = data_dir / "data.csv"
	# write simple data: y = 2x + 1
	rows = [(x, 2*x+1) for x in range(5)]
	with open(sample_data, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['x','y'])
		writer.writerows(rows)
	# model dir
	model_dir = tmp_path / "model"
	model_dir.mkdir()
	model_file = model_dir / "model.csv"

	# Monkeypatch get_path to map relative to tmp_path
	def fake_get_path(rel):
		return str(tmp_path / rel)
	monkeypatch.setattr('utils.get_path', fake_get_path)
	return {'data': str(sample_data), 'model': str(model_file)}





def test_train_model_creates_model(clear_env, tmp_path, capsys, monkeypatch):
    paths = clear_env
    # Simulate CLI args for training
    testargs = ["train_model.py",
                "-d", paths['data'],
                "-m", paths['model'],
                "-lr", "0.1",
                "--iter", "5000"]
    monkeypatch.setattr(sys, 'argv', testargs)

    # Run the training script
    tm.main()

    # Verify model file was created and contains correct columns
    assert os.path.isfile(paths['model']), "Model CSV was not created"
    df = pd.read_csv(paths['model'])
    assert set(df.columns) >= {'theta0', 'theta1'}

    # Check learned parameters approximate y=2x+1
    theta0 = df.loc[0, 'theta0']
    theta1 = df.loc[0, 'theta1']
    assert pytest.approx(1.0, rel=1e-1) == theta0
    assert pytest.approx(2.0, rel=1e-1) == theta1


def test_predict_price_output(clear_env, tmp_path, capsys, monkeypatch):
	paths = clear_env
	# prepare a model file with known parameters
	df = pd.DataFrame([{'theta0': 1.0, 'theta1': 2.0}])
	pd.DataFrame(df).to_csv(paths['model'], index=False)

	# Simulate CLI args for predict_price: just specify model
	testargs = ["predict_price.py", "-m", paths['model']]
	monkeypatch.setattr(sys, 'argv', testargs)

	# Monkeypatch input to return '3'
	monkeypatch.setattr(builtins, 'input', lambda _: '3')

	# Capture output
	pp.predict_price()
	captured = capsys.readouterr()
	# Estimated price = 1 + 2*3 = 7.00
	assert "7.00" in captured.out
