import numpy as np
from sklearn import svm

TRAIN_PATH = "data/features.train"
TEST_PATH = "data/features.test"

# Soft-margin SVM: 0 <= alpha_n <= C
# Hard margin is the C -> infinity limit (see previous homework footnote)
C = 0.01
POLY_DEGREE = 5


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


def train_soft_margin_svm(x, y, C=C, degree=POLY_DEGREE):
	poly_svm = svm.SVC(kernel='poly', degree=degree, gamma=1.0, coef0=1.0, C=C)
	poly_svm.fit(x, y)   # this actually trains
	return poly_svm


def predict(model, x):
	return model.predict(x)


def classification_error(model, x, y):
	return np.mean(predict(model, x) != y)


if __name__ == "__main__":
	digits_train, x_train = load_data(TRAIN_PATH)
	digits_test, x_test = load_data(TEST_PATH)

	for i in range (5):
		C = 10**(i-4)
		positive = 1
		negative = 5
		x_tr, y_tr = one_versus_one(digits_train, x_train, positive, negative)
		x_te, y_te = one_versus_one(digits_test, x_test, positive, negative)

		model = train_soft_margin_svm(x_tr, y_tr, C=C, degree=POLY_DEGREE)
		e_in = classification_error(model, x_tr, y_tr)
		e_out = classification_error(model, x_te, y_te)

		print(f"classifier: {positive} vs. {negative}")
		print(f"C={C}, Q={POLY_DEGREE}")
		print(f"E_in:  {e_in:.4f}")
		print(f"E_out: {e_out:.4f}")
		print("n_support:", len(model.support_vectors_)) 
