import numpy as np

eta = 0.1
w = np.array([1.0,1.0])
threshold = 1e-14
count = 0

def function(w):
    return (w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0]))**2

def gradient(w):
	return np.array([2*(np.exp(w[1])+2*w[1]*np.exp(-w[0]))*(w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0])),2*(w[0]*np.exp(w[1])-2*np.exp(-w[0]))*(w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0]))])

def step(w):
	w_step = -1*gradient(w)
	return w+eta*w_step

#start gradient descent
error = function(w)
while error > threshold:
	w = step(w)
	count += 1
	error = function(w)

print(w)