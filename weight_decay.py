import numpy as np

TRAIN_URL = "http://work.caltech.edu/data/in.dta"
TEST_URL = "http://work.caltech.edu/data/out.dta"

k=0

def load_data(path):
	data = np.loadtxt(path)
	x = data[:, :2]
	y = data[:, 2]
	return x, y


def transform(x):
	x1 = x[:, 0]
	x2 = x[:, 1]
	transformed_points = np.column_stack([
		np.ones(len(x1)), x1, x2, x1**2,x2**2, x1*x2, np.absolute(x1-x2), np.absolute(x1+x2) 
	])
	return transformed_points


def linear_regression(X, y, lam):
	d = X.shape[1]
	pseudo_inverse = np.linalg.inv(X.T @ X + lam * np.eye(d)) @ X.T
	print(lam)
	return pseudo_inverse @ y


def classification_error(weights, Z, y):
	return np.mean(np.sign(Z @ weights) != y)


if __name__ == "__main__":
	x_train, y_train = load_data(TRAIN_URL)
	x_test, y_test = load_data(TEST_URL)

	Z_train = transform(x_train)
	Z_test = transform(x_test)

	if k!=0:
		lam = 10**k
	else:
		lam = 0
	w = linear_regression(Z_train, y_train, lam)
	print(w)

	e_in = classification_error(w, Z_train, y_train)
	e_out = classification_error(w, Z_test, y_test)

	print("E_in:", e_in)
	print("E_out:", e_out)
