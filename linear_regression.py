import numpy as np
import random

N_in = 100
N_out = 1000
NUM_TRIALS = 1000
bound = np.array([[-1, 1], [-1, 1]])
g = np.empty((NUM_TRIALS, 3))
E_in = 0
E_out = 0

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

def get_in_sample(weights, points, y):
	error = np.mean(np.sign(points @ weights) != y)
	return error

def get_out_sample(weights, line):
	points = np.array([
		[1, random_coord(0), random_coord(1)]
		for _ in range(N_out)
		])
	preds_g = np.sign(points @ weights)
	preds_f = np.sign(points @ line)
	return np.mean(preds_f != preds_g)

for i in range(NUM_TRIALS):
	np_line = generate_target_line()
	np_points, y = generate_dataset(np_line)
	g[i] = linear_regression(np_points, y)
	E_in += get_in_sample(g[i], np_points, y)
	E_out += get_out_sample(g[i], np_line)

print("Average E_in: "+ str(E_in/NUM_TRIALS))
print("Average E_out: "+ str(E_out/NUM_TRIALS))