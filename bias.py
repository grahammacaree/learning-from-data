import numpy as np

NUM_TRIALS = 1000

xs = np.random.uniform(-1, 1, (NUM_TRIALS, 2))
ys = np.sin(np.pi * xs)
slopes = np.sum(xs*ys, axis=1) / np.sum(xs**2, axis=1)
a = np.mean(slopes)

#calc bias
xb = np.random.uniform(-1, 1, (NUM_TRIALS))
yb = np.sin(np.pi * xb)
ya = a*xb
bias = np.mean((ya - yb) ** 2)

#calc variance
xt = np.random.uniform(-1, 1, (NUM_TRIALS, 2))
variance = np.mean((slopes - a)**2) * np.mean(xt**2)

print("a: "+str(a))
print("bias: "+str(bias))
print("variance: "+str(variance))