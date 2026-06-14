import numpy as np
import random

N = 100
MONTE_CARLO = 1000
NUM_TRIALS = 1000
bound = np.array([[-1, 1], [-1, 1]])


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
		for _ in range(N)
	]
	np_points = np.array(points)
	y = np.array([line_side(p, np_line) for p in np_points])
	return np_points, y


def train_perceptron(np_line, np_points, y):
	weights = np.zeros(3)
	runs = 0
	while True:
		scores = np_points @ weights
		preds = np.sign(scores)
		if not np.any(preds != y):
			pointTemp = []
			for _ in range(MONTE_CARLO):
				point = [1, random_coord(0), random_coord(1)]
				pointTemp.append(point)
			mc_points = np.array(pointTemp)
			scores_g = mc_points @ weights
			preds_g = np.sign(scores_g)
			scores_f = mc_points @ np_line
			preds_f = np.sign(scores_f)
			disagree = np.mean(preds_f != preds_g)
			break
		mis_idx = np.where(preds != y)[0]
		i = random.choice(mis_idx)
		weights += y[i] * np_points[i]
		runs += 1
	return runs, disagree

total_runs = 0
total_disagreement_mean = 0
for _ in range(NUM_TRIALS):
	np_line = generate_target_line()
	np_points, y = generate_dataset(np_line)
	runs, disagree = train_perceptron(np_line, np_points, y)
	total_runs += runs
	total_disagreement_mean += disagree

print("N: "+ str(N))
print("Runs: "+ str(total_runs / NUM_TRIALS))
print("Average disagreement: " + str(total_disagreement_mean / NUM_TRIALS))
