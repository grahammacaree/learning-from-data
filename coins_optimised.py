import numpy as np

num_experiments = 100000
num_coins = 1000
num_trials = 10

# Generate all flips at once: 100,000 experiments × 1,000 coins × 10 flips
# Shape: (100000, 1000, 10)
flips = np.random.randint(0, 2, size=(num_experiments, num_coins, num_trials))

# Sum along the flips axis to get head counts, divide by 10 for frequencies
# Shape: (100000, 1000)
nu = flips.mean(axis=2)

# c1: first coin in each experiment
nu1 = nu[:, 0]

# crand: random coin index for each experiment
rand_indices = np.random.randint(0, num_coins, size=num_experiments)
nu_rand = nu[np.arange(num_experiments), rand_indices]

# cmin: coin with minimum heads in each experiment
nu_min = nu.min(axis=1)

print("v1: "+ str(nu1.mean()))
print("vRand: "+ str(nu_rand.mean()))
print("vMin: " + str(nu_min.mean()))