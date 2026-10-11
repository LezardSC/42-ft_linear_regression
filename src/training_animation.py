import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends import backend_registry, BackendFilter

from linear_regression import LinearRegression


BACKGROUND = "#0f1117"
FOREGROUND = "#e6e8ee"
MUTED = "#8b90a0"
POINTS = "#c9ced8"
ACCENT = "#ffb000"

FPS = 30
STYLE = {
	"font.family": ["Inter", "Ubuntu", "DejaVu Sans"],
	"toolbar": "None",
	"figure.figsize": (12.8, 7.2),
	"figure.dpi": 100,
}

INTRO = 2.5
INTRO_WITHOUT_QUESTION = 1.8
LEARNING = 6.5
ANSWER = 1.0


def ease(t: float) -> float:
	"""
	Smoothstep easing, clamped to [0, 1].

	Parameters:
		t (float): Linear progress.

	Returns:
		float: Eased progress.
	"""

	t = min(max(t, 0.0), 1.0)
	return t * t * (3 - 2 * t)


def progress(t: float, start: float, duration: float) -> float:
	"""
	Linear progress of time t through the window [start, start + duration], clamped to [0, 1].
	"""

	return min(max((t - start) / duration, 0.0), 1.0)


def format_number(value: float) -> str:
	"""
	Format a number with spaces as thousands separators (120 000).
	"""

	return f"{round(value):,}".replace(",", " ")


class TrainingAnimation:
	"""
	Animation of a gradient descent: the data appears, then the line learns. The final state stays on screen.
	With a mileage, the model is also asked the price of a car at that mileage, and answers once trained.

	Methods:
		draw(frame): Update the figure for the given frame.
		show(): Play the animation in a window.
	"""

	def __init__(self, x: np.ndarray, y: np.ndarray, model: LinearRegression, mileage: float = None):
		"""
		Parameters:
			x (np.ndarray): Feature values.
			y (np.ndarray): Target values.
			model (LinearRegression): Model trained with record_history=True.
			mileage (float, optional): Mileage to ask the price of.

		Raises:
			ValueError: If the model has no recorded history.
		"""

		if model.history is None:
			raise ValueError("The model must be trained with record_history=True to be animated.")

		self.x = x
		self.y = y
		self.mileage = mileage
		self.history = np.array(model.history)
		self.answer = model.predict(mileage) if mileage is not None else None

		self.learning_start = INTRO if mileage is not None else INTRO_WITHOUT_QUESTION
		self.answer_start = self.learning_start + LEARNING
		end = self.answer_start + (ANSWER if mileage is not None else 0)
		# One more frame so that the last one shows the final state.
		self.n_frames = round(end * FPS) + 1

		# Gradient descent does most of its work in the first iterations: sample them on a log scale.
		n_iter = len(self.history) - 1
		u = np.linspace(0, 1, round(LEARNING * FPS))
		self.iterations = np.round(np.expm1(np.array([ease(v) for v in u]) * np.log1p(n_iter))).astype(int)

		# Points pop in one by one, in order of mileage.
		self.pop_order = np.argsort(np.argsort(x)) / max(len(x) - 1, 1)

		logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
		with plt.rc_context(STYLE):
			self._build_figure()

	def _build_figure(self):
		"""
		Create every artist, invisible until drawn.
		"""

		self.fig = plt.figure(facecolor=BACKGROUND)
		if self.fig.canvas.manager is not None:
			self.fig.canvas.manager.set_window_title("ft_linear_regression - training")
		ax = self.fig.add_axes([0.1, 0.13, 0.85, 0.65], facecolor=BACKGROUND)
		self.ax = ax

		x_max = max(self.x.max(), self.mileage or 0) * 1.08
		y_low, y_high = self.y.min(), self.y.max()
		margin = (y_high - y_low) * 0.35
		ax.set_xlim(0, x_max)
		ax.set_ylim(max(0, y_low - margin), y_high + margin)

		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("bottom", "left"):
			ax.spines[side].set_color(MUTED)
			ax.spines[side].set_linewidth(1.5)
		ax.tick_params(colors=MUTED, labelsize=22, length=0, pad=10)
		ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
		ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
		ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k km" if v else "0"))
		ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k €" if v else "0"))

		self.points = ax.scatter(self.x, self.y, s=0, color=POINTS, edgecolors="none", zorder=3)
		self.regression, = ax.plot([], [], color=ACCENT, linewidth=4, solid_capstyle="round", zorder=2)
		self.counter = ax.text(0.99, 0.02, "", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=20)

		# The top right of the chart is empty (expensive cars with a high mileage): room for the cost curve.
		self.inset = self.fig.add_axes([0.72, 0.56, 0.21, 0.18], facecolor=BACKGROUND)
		self.inset.set_axis_off()
		# Cost against log(iteration), drawn from every iteration so the curve stays smooth.
		self.cost_x = np.log1p(np.arange(len(self.history)))
		self.inset.set_xlim(0, self.cost_x[-1])
		self.inset.set_ylim(0, self.history[:, 2].max() * 1.1)
		self.cost_curve, = self.inset.plot([], [], color=ACCENT, linewidth=2.5)
		self.cost_label = self.inset.text(0, 1.1, "cost", transform=self.inset.transAxes, color=MUTED, fontsize=18, va="bottom")

		if self.mileage is None:
			return

		self.question_line, = ax.plot([], [], color=MUTED, linestyle=(0, (4, 4)), linewidth=2, zorder=1)
		self.answer_line, = ax.plot([], [], color=ACCENT, linestyle=(0, (4, 4)), linewidth=2, zorder=1)
		self.halo = ax.scatter([self.mileage], [self.answer], s=0, color=ACCENT, edgecolors="none", zorder=4)
		self.dot = ax.scatter([self.mileage], [self.answer], s=0, color=ACCENT, edgecolors=BACKGROUND, linewidths=3, zorder=5)

		# The question, written as two pieces so that only the answer changes.
		self.question = self.fig.text(0.1, 0.875, f"{format_number(self.mileage)} km  →  ", color=FOREGROUND, fontsize=40, fontweight="bold", va="center")
		self.fig.canvas.draw()
		extent = self.question.get_window_extent().transformed(self.fig.transFigure.inverted())
		self.reply = self.fig.text(extent.x1, 0.875, "", fontsize=40, fontweight="bold", va="center")

	def draw(self, frame: int):
		"""
		Update every artist for the given frame.

		Parameters:
			frame (int): Frame index, from 0 to n_frames - 1.
		"""

		t = frame / FPS
		x_left, x_right = self.ax.get_xlim()

		# The data points pop in.
		pop = np.array([ease(progress(t, 0.1 + 1.3 * order, 0.35)) for order in self.pop_order])
		overshoot = 1 + 0.35 * np.sin(np.pi * pop)
		self.points.set_sizes(140 * pop * overshoot)
		self.points.set_alpha(0.8)

		# Learning: the line moves from its initial state to the fitted one.
		step = min(round(progress(t, self.learning_start, LEARNING) * (len(self.iterations) - 1)), len(self.iterations) - 1)
		iteration = self.iterations[step]
		theta0, theta1, _ = self.history[iteration]
		self.regression.set_data([x_left, x_right], [theta0 + theta1 * x_left, theta0 + theta1 * x_right])

		shown = ease(progress(t, self.learning_start, 0.3))
		self.regression.set_alpha(shown)
		self.counter.set_text(f"iteration {format_number(iteration)}")
		self.counter.set_alpha(shown)
		self.cost_curve.set_data(self.cost_x[:iteration + 1], self.history[:iteration + 1, 2])
		self.cost_curve.set_alpha(shown)
		self.cost_label.set_alpha(shown)

		if self.mileage is not None:
			self._draw_question(t)

	def _draw_question(self, t: float):
		"""
		The question drops down with the data, then the dot lands on the line and the "?" becomes the price.
		"""

		y_bottom, y_top = self.ax.get_ylim()
		x_left = self.ax.get_xlim()[0]

		drop = ease(progress(t, 1.6, 0.6))
		self.question_line.set_data([self.mileage, self.mileage], [y_top, y_top - (y_top - y_bottom) * drop])
		self.question.set_alpha(ease(progress(t, 1.6, 0.5)))

		landing = progress(t, self.answer_start, 0.4)
		self.dot.set_sizes([260 * ease(landing) * (1 + 0.4 * np.sin(np.pi * landing))])
		ripple = progress(t, self.answer_start + 0.1, 0.9)
		self.halo.set_sizes([260 + 2200 * ease(ripple)])
		self.halo.set_alpha(0.35 * (1 - ripple) if ripple > 0 else 0)

		guide = ease(progress(t, self.answer_start + 0.2, 0.5))
		self.answer_line.set_data([self.mileage, self.mileage - (self.mileage - x_left) * guide], [self.answer, self.answer])
		self.answer_line.set_alpha(0.6)

		if t < self.answer_start + 0.2:
			self.reply.set_text("? €")
			self.reply.set_color(MUTED)
			self.reply.set_alpha(self.question.get_alpha())
		else:
			self.reply.set_text(f"≈ {format_number(self.answer)} €")
			self.reply.set_color(ACCENT)
			self.reply.set_alpha(ease(progress(t, self.answer_start + 0.2, 0.3)))

	def show(self):
		"""
		Play the animation once in a window, which keeps the final state until it is closed.

		Raises:
			RuntimeError: If matplotlib has no interactive backend to open a window.
		"""

		backend = matplotlib.get_backend()
		if backend.lower() not in backend_registry.list_builtin(BackendFilter.INTERACTIVE):
			raise RuntimeError(
				f"matplotlib can't open a window (backend '{backend}'). "
				"Install tkinter, e.g. 'sudo apt install python3-tk'."
			)
		with plt.rc_context(STYLE):
			animation = FuncAnimation(self.fig, self.draw, frames=self.n_frames, interval=1000 / FPS, repeat=False)
			plt.show()
		return animation
