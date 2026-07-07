import random

import numpy as np
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False

NUM_TRIALS = 1000
MONTE_CARLO = 1000
bound = np.array([[-1, 1], [-1, 1]])


def random_coord(axis):
	low = bound[axis][0]
	high = bound[axis][1]
	return np.random.uniform(low, high)


def line_side(point, np_line):
	return int(np.sign(point @ np_line))


def generate_target_line():
	points = [
		[random_coord(0), random_coord(1)],
		[random_coord(0), random_coord(1)],
	]
	p0, p1 = points[0], points[1]
	return np.array([
		p0[0] * p1[1] - p1[0] * p0[1],
		p0[1] - p1[1],
		p1[0] - p0[0],
	])


def generate_dataset(np_line, n):
	points = [
		[1, random_coord(0), random_coord(1)]
		for _ in range(n)
	]
	np_points = np.array(points)
	y = np.array([line_side(p, np_line) for p in np_points])
	return np_points, y


def generate_separable_dataset(n):
	while True:
		np_line = generate_target_line()
		np_points, y = generate_dataset(np_line, n)
		if len(np.unique(y)) > 1:
			return np_line, np_points, y


def train_perceptron(np_points, y):
	weights = np.zeros(3)
	while True:
		preds = np.sign(np_points @ weights)
		if not np.any(preds != y):
			break
		mis_idx = np.where(preds != y)[0]
		i = random.choice(mis_idx)
		weights += y[i] * np_points[i]
	return weights


def train_svm(np_points, y):
	# Hard-margin primal SVM on augmented points [1, x1, x2]:
	# minimize  1/2 ||w||^2
	# subject to  y_n (w^T x_n) >= 1  for all n
	#
	# cvxopt.solvers.qp minimizes  1/2 x'Px + q'x  s.t.  Gx <= h
	n, d = np_points.shape
	P = matrix(np.eye(d))
	q = matrix(np.zeros(d))
	G = matrix(-y[:, np.newaxis] * np_points)
	h = matrix(-np.ones(n))
	solution = solvers.qp(P, q, G, h)
	weights = np.array(solution["x"]).flatten()
	margins = y * (np_points @ weights)
	n_support = int(np.sum(margins <= 1 + 1e-4))
	return weights, n_support


def disagreement(weights, np_line):
	mc_points = np.array([
		[1, random_coord(0), random_coord(1)]
		for _ in range(MONTE_CARLO)
	])
	preds_g = np.sign(mc_points @ weights)
	preds_f = np.sign(mc_points @ np_line)
	return np.mean(preds_f != preds_g)


def run_comparison(n):
	svm_better = 0
	total_support = 0

	for _ in range(NUM_TRIALS):
		np_line, np_points, y = generate_separable_dataset(n)

		w_pla = train_perceptron(np_points, y)
		w_svm, n_support = train_svm(np_points, y)

		e_pla = disagreement(w_pla, np_line)
		e_svm = disagreement(w_svm, np_line)
		if e_svm < e_pla:
			svm_better += 1

		total_support += n_support

	return svm_better / NUM_TRIALS, total_support / NUM_TRIALS


if __name__ == "__main__":
	for n in (10, 100):
		svm_win_rate, avg_support = run_comparison(n)
		print(f"N={n}")
		print(f"  SVM better than PLA: {svm_win_rate:.2%}")
		print(f"  Avg support vectors: {avg_support:.2f}")
