import numpy as np
import random

N_in = 10
NUM_TRIALS = 1000
bound = np.array([[-1, 1], [-1, 1]])
g = np.empty((NUM_TRIALS, 3))
runs = 0

def random_coord(axis):
	low = bound[axis][0]
	high = bound[axis][1]
	return np.random.uniform(low, high)

def line_side(point, np_line):
	return int(
		np.sign(point @ np_line)
	)

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

def generate_dataset(np_line):
	points = [
		[1, random_coord(0), random_coord(1)]
		for _ in range(N_in)
	]
	np_points = np.array(points)
	y = np.array([line_side(p, np_line) for p in np_points])
	return np_points, y

def linear_regression(X, y):
	pseudoInverse = np.linalg.inv((X.T @ X)) @ X.T
	w = pseudoInverse @ y
	return w

def train_perceptron(np_points, y, weights):
	trial_runs = 0
	while True:
		scores = np_points @ weights
		preds = np.sign(scores)
		if not np.any(preds != y):
			break
		mis_idx = np.where(preds != y)[0]
		i = random.choice(mis_idx)
		weights += y[i] * np_points[i]
		trial_runs += 1
	return trial_runs

for i in range(NUM_TRIALS):
	np_line = generate_target_line()
	np_points, y = generate_dataset(np_line)
	g[i] = linear_regression(np_points, y)
	runs += train_perceptron(np_points, y, g[i].copy())
	

print("Average runs: "+ str(runs / NUM_TRIALS))