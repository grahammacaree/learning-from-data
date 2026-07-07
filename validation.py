import numpy as np

TRAIN_URL = "http://work.caltech.edu/data/in.dta"
TEST_URL = "http://work.caltech.edu/data/out.dta"

N_TRAIN = 10
N_VAL = 25
K_VALUES = [3, 4, 5, 6, 7]


def load_data(path):
	data = np.loadtxt(path)
	x = data[:, :2]
	y = data[:, 2]
	return x, y


def transform(x):
	x1 = x[:, 0]
	x2 = x[:, 1]
	return np.column_stack([
		np.ones(len(x1)), x1, x2, x1**2, x2**2, x1 * x2,
		np.absolute(x1 - x2), np.absolute(x1 + x2),
	])


def linear_regression(X, y, lam=0):
	d = X.shape[1]
	pseudo_inverse = np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T
	return pseudo_inverse @ y


def classification_error(weights, Z, y):
	return np.mean(np.sign(Z @ weights) != y)


def split_train_val(x, y):
	Z = transform(x)
	return (
		Z[:N_TRAIN], y[:N_TRAIN],
		Z[N_TRAIN:N_TRAIN + N_VAL], y[N_TRAIN:N_TRAIN + N_VAL],
	)


def fit_and_errors(Z_train, y_train, Z_val, y_val, Z_test, y_test, k):
	Z_train_k = Z_train[:, :k + 1]
	Z_val_k = Z_val[:, :k + 1]
	Z_test_k = Z_test[:, :k + 1]

	w = linear_regression(Z_train_k, y_train)
	return {
		"w": w,
		"e_train": classification_error(w, Z_train_k, y_train),
		"e_val": classification_error(w, Z_val_k, y_val),
		"e_test": classification_error(w, Z_test_k, y_test),
	}


if __name__ == "__main__":
	x_in, y_in = load_data(TRAIN_URL)
	x_test, y_test = load_data(TEST_URL)

	Z_train, y_train, Z_val, y_val = split_train_val(x_in, y_in)
	Z_test = transform(x_test)

	best_k = None
	best_e_val = float("inf")

	for k in K_VALUES:
		results = fit_and_errors(Z_train, y_train, Z_val, y_val, Z_test, y_test, k)
		print(
			f"k={k}: E_train={results['e_train']:.4f} "
			f"E_val={results['e_val']:.4f} E_test={results['e_test']:.4f}"
		)
		if results["e_val"] < best_e_val:
			best_e_val = results["e_val"]
			best_k = k

	print(f"Best k by validation: {best_k} (E_val={best_e_val:.4f})")
