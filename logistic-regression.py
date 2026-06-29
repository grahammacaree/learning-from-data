import numpy as np

N = 100
N_out = 1000
NUM_TRIALS = 100
eta = 0.01
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


def get_out_sample(weights, np_line):
	points = np.array([
		[1, random_coord(0), random_coord(1)]
		for _ in range(N_out)
	])
	y = np.sign(points @ np_line)
	# cross-entropy error: mean of ln(1 + exp(-y * w.x))
	return np.mean(np.log(1 + np.exp(-y * (points @ weights))))


def logistic_regression(np_points, y):
	w = np.zeros(3)
	w_prev = np.array([1.0, 1.0, 1.0])
	epochs = 0
	running = True
	while running:
		w_prev = w.copy()
		shuffled = np.random.permutation(N)
		epochs += 1
		for i in range(0, len(shuffled)):
			w += eta * (y[shuffled[i]] * np_points[shuffled[i]]) / (1 + np.exp(y[shuffled[i]] * (w @ np_points[shuffled[i]])))
		if np.linalg.norm(w_prev - w) < 0.01:
			break
	return(w, epochs)


total_eout = 0
total_epochs = 0
for _ in range(NUM_TRIALS):
	np_line = generate_target_line()
	np_points, y = generate_dataset(np_line)
	w, epochs = logistic_regression(np_points, y)
	total_epochs+= epochs
	total_eout += get_out_sample(w, np_line)

print("Average E_out: " + str(total_eout / NUM_TRIALS))
print("Average epochs: " + str(total_epochs / NUM_TRIALS))
