import numpy as np
import random

num_coins = 1000
num_trials = 10
num_experiments = 100000
#coin array is just a num_coins long array
coins = np.empty(num_coins)

v1 = 0
vrand = 0
vmin = 0

def flip_coin(num_trials):
	heads = 0
	for _ in range(num_trials):
		heads += random.randint(0, 1)
	return heads/num_trials

def experiment(array, trials):
	for i in range(len(array)):
		array[i] = flip_coin(trials)
	return array

for j in range(num_experiments):
	coins = experiment(coins, num_trials)
	v1+= coins[0]
	vrand+= coins[random.randint(0, len(coins) - 1)]
	vmin+= min(coins)
	print("experiment " + str(j) + " complete")

print("v1: "+ str(v1/num_experiments))
print("vRand: "+ str(vrand/num_experiments))
print("vMin: " + str(vmin/num_experiments))