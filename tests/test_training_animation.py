# tests/test_training_animation.py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pytest

from linear_regression import LinearRegression
from training_animation import TrainingAnimation


@pytest.fixture
def trained():
	x = np.array([20000.0, 60000.0, 100000.0, 150000.0, 240000.0])
	y = np.array([8000.0, 7500.0, 6500.0, 5000.0, 3500.0])
	model = LinearRegression(learning_rate=0.1, n_iter=500)
	model.fit(x, y, record_history=True)
	yield x, y, model
	plt.close("all")


def test_requires_history(trained):
	x, y, _ = trained
	model = LinearRegression()
	model.fit(x, y)
	with pytest.raises(ValueError):
		TrainingAnimation(x, y, model)


@pytest.mark.parametrize("mileage", [None, 120000.0])
def test_draws_every_frame_and_ends_on_the_fit(trained, mileage):
	x, y, model = trained
	animation = TrainingAnimation(x, y, model, mileage)
	animation.draw(0)
	assert np.all(animation.points.get_sizes() == 0)
	for frame in range(animation.n_frames):
		animation.draw(frame)
	# the last frame shows every point and the fitted line
	assert np.all(animation.points.get_sizes() > 0)
	assert animation.regression.get_alpha() == pytest.approx(1)
	_, ys = animation.regression.get_data()
	assert ys[0] == pytest.approx(model.predict(animation.ax.get_xlim()[0]))


def test_answers_the_question(trained):
	x, y, model = trained
	animation = TrainingAnimation(x, y, model, 120000.0)
	animation.draw(round(animation.answer_start * 30) - 1)
	assert animation.reply.get_text() == "? €"
	animation.draw(animation.n_frames - 1)
	assert animation.reply.get_alpha() == pytest.approx(1)
	assert animation.reply.get_text() == f"≈ {round(model.predict(120000.0)):,} €".replace(",", " ")
