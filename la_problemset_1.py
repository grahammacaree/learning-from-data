import numpy as np

# Data setup (Do not alter this part)
np.random.seed(42)
X = np.random.uniform(-2, 2, (1000, 3)) # 1000 points in 3D space
v = np.array([1.5, -0.5, 2.0])          # Structural vector

# Messy, loop-heavy calculation to refactor:
squared_distances = np.zeros(1000)
for i in range(1000):
    diff = X[i] - v
    squared_distances[i] = np.dot(diff, diff)

print(np.mean(squared_distances))

#we can just np-sum across the primary axis of the data array
new_squared_distances = np.sum((X - v)**2, axis=1)
alt_squared_distances = np.mean((X - v)**2, axis=0) @ np.mean((X - v)**2, axis=0)

print(np.mean(new_squared_distances))
print(alt_squared_distances)

# 1. Compute the differences: shape (1000, 3)
diff = X - v
#obvious

# 2. Reshape into a batch of row vectors (1000, 1, 3) and column vectors (1000, 3, 1)
row_vectors = diff[:, np.newaxis, :] 
# this creates 1000 1 x 3 rows [xn yn zn]
col_vectors = diff[:, :, np.newaxis]
# this creates 1000 3 x 1 columns [[xn] [yn] [zn]]

# 3. Batch matrix multiply! (1000, 1, 3) @ (1000, 3, 1) -> (1000, 1, 1)
batch_squared_distances = (row_vectors @ col_vectors).squeeze()
#squeeze collapses extraneous dimensions

print(np.mean(batch_squared_distances))