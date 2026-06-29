import numpy as np

eta = 0.1
w = np.array([1.0,1.0])
threshold = 1e-14
count = 0

def function(w):
    return (w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0]))**2

def coord_u(w):
	return np.array([2*(np.exp(w[1])+2*w[1]*np.exp(-w[0]))*(w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0])),0])

def coord_v(w):
	return np.array([0,2*(w[0]*np.exp(w[1])-2*np.exp(-w[0]))*(w[0]*np.exp(w[1]) - 2*w[1]*np.exp(-w[0]))])

def step_u(w):
	w_step = -1*coord_u(w)
	return w+eta*w_step

def step_v(w):
	w_step = -1*coord_v(w)
	return w+eta*w_step

#start coordinate descent
error = function(w)
for count in range(0, 15):
	w = step_u(w)
	w = step_v(w)
	count += 1
	error = function(w)

print(error)