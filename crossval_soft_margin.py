import numpy as np
from sklearn import svm

TRAIN_PATH = "data/features.train"
TEST_PATH = "data/features.test"

# Soft-margin SVM: 0 <= alpha_n <= C
# Do not scale the data.
POLY_DEGREE = 2
N_FOLDS = 10


def load_data(path):
	# each row: digit, intensity, symmetry
	data = np.loadtxt(path)
	digits = data[:, 0].astype(int)
	x = data[:, 1:]
	return digits, x


def one_versus_all(digits, x, target):
	# target digit -> +1, every other digit -> -1
	y = np.where(digits == target, 1, -1)
	return x, y


def one_versus_one(digits, x, positive, negative):
	# keep only the two digits; positive -> +1, negative -> -1
	mask = (digits == positive) | (digits == negative)
	x_sub = x[mask]
	y = np.where(digits[mask] == positive, 1, -1)
	return x_sub, y


def train_soft_margin_svm(x, y, C, degree=POLY_DEGREE):
	poly_svm = svm.SVC(kernel="poly", degree=degree, gamma=1.0, coef0=1.0, C=C)
	poly_svm.fit(x, y)
	return poly_svm


def classification_error(model, x, y):
	return np.mean(model.predict(x) != y)


def make_folds(n, n_folds=N_FOLDS, rng=None):
	indices = np.random.permutation(n)
	folds = np.array_split(indices, n_folds)
	return folds


def cross_validate(x, y, C, folds, degree=POLY_DEGREE):
	# Evaluate one C on a fixed fold partition (pass the same folds for every C in a run).
	n_folds = len(folds)
	e_cv = 0.0
	for i in range(n_folds):
		val_idx = folds[i]
		train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
		model = train_soft_margin_svm(x[train_idx], y[train_idx], C=C, degree=degree)
		e_cv += classification_error(model, x[val_idx], y[val_idx])
	return e_cv / n_folds


def select_C(x, y, C_values, degree=POLY_DEGREE):
	# One random run: one shuffle of folds, then pick C with lowest E_cv.
	# Tie-break: smaller C (iterate ascending; update only on strict <).
	folds = make_folds(len(y))
	best_C = None
	best_ecv = float("inf")
	for C in sorted(C_values):
		e_cv = cross_validate(x, y, C=C, folds=folds, degree=degree)
		if e_cv < best_ecv:
			best_ecv = e_cv
			best_C = C
	return best_C, best_ecv


if __name__ == "__main__":
	digits_train, x_train = load_data(TRAIN_PATH)
	x, y = one_versus_one(digits_train, x_train, positive=1, negative=5)

	C_values = [0.001]
	NUM_RUNS = 100

	# Each run reshuffles folds; collect the chosen C (and optionally E_cv).
	# Use the histogram of selected C's (and/or stats of E_cv) to answer the MC question.
	selected_C = []
	average_ecv = 0
	for run in range(NUM_RUNS):
		best_C, best_ecv = select_C(x, y, C_values, degree=POLY_DEGREE)
		average_ecv+= best_ecv
		selected_C.append(best_C)
		print(f"run {run + 1}/{NUM_RUNS}: best C={best_C}, E_cv={best_ecv:.4f}")

	selected_C = np.array(selected_C)
	print("\nC selection counts over", NUM_RUNS, "runs:")
	for C in C_values:
		print(f"  C={C}: {(selected_C == C).sum()} times, average E_cv={average_ecv/NUM_RUNS:.4f}")
