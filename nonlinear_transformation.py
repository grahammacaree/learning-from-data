import numpy as np

N_in = 1000
N_out = 1000
NUM_TRIALS = 1000
g = np.empty((NUM_TRIALS, 6))
E_out = np.empty(NUM_TRIALS)

def add_error(n, y):
	flip = np.random.random(n) < 0.1
	y[flip] *= -1
	return y

def generate_dataset():
	raw = np.random.uniform(-1, 1, (N_in, 2))
	x1 = raw[:, 0]
	x2 = raw[:, 1]
	y = np.sign(x1**2 + x2**2 - 0.6)
	y = add_error(N_in, y.copy())
	transformed_points = np.column_stack([
		np.ones(N_in), x1, x2, x1 * x2, x1**2, x2**2
	])
	return transformed_points, y

def linear_regression(X, y):
	pseudoInverse = np.linalg.inv((X.T @ X)) @ X.T
	w = pseudoInverse @ y
	return w

def get_out_sample(weights):
	Z, y = generate_dataset()
	preds_g = np.sign(Z @ weights)

	return np.mean(preds_g != y)

for i in range(NUM_TRIALS):
	transformed_points, y = generate_dataset()
	g[i] = linear_regression(transformed_points, y)
	E_out[i] = get_out_sample(g[i])


print("weights:" + str(np.mean(g, axis=0)))
print("Average E_out: "+ str(np.mean(E_out)))