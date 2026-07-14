import numpy as np

# Objective function (minimize)
def f(x):
    return x**2

# Parameters
n_particles = 20
iterations = 100

w = 0.7
c1 = 2.0
c2 = 2.0

# Initialize
positions = np.random.uniform(-10, 10, n_particles)
velocities = np.random.uniform(-1, 1, n_particles)

pbest = positions.copy()
pbest_values = f(pbest)

gbest = pbest[np.argmin(pbest_values)]

# Main loop
for _ in range(iterations):

    r1 = np.random.rand(n_particles)
    r2 = np.random.rand(n_particles)

    velocities = (
        w * velocities
        + c1 * r1 * (pbest - positions)
        + c2 * r2 * (gbest - positions)
    )

    positions = positions + velocities

    values = f(positions)

    better = values < pbest_values

    pbest[better] = positions[better]
    pbest_values[better] = values[better]

    gbest = pbest[np.argmin(pbest_values)]

print("Best solution:", gbest)
print("Best value:", f(gbest))
