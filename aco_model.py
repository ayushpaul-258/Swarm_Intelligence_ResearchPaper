import numpy as np

# -------------------------------
# Distance Matrix
# -------------------------------
distance = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10,4, 8, 0]
], dtype=float)

n = len(distance)

# -------------------------------
# Parameters
# -------------------------------
num_ants = 10
iterations = 100

alpha = 1          # Importance of pheromone
beta = 2           # Importance of heuristic
rho = 0.5          # Evaporation rate
Q = 100            # Pheromone deposit constant

# -------------------------------
# Initialize pheromone matrix
# -------------------------------
pheromone = np.ones((n, n))

best_path = None
best_length = float('inf')

# -------------------------------
# Function to calculate tour length
# -------------------------------
def tour_length(path):
    total = 0

    for i in range(len(path)-1):
        total += distance[path[i]][path[i+1]]

    total += distance[path[-1]][path[0]]

    return total

# -------------------------------
# Main Loop
# -------------------------------
for iteration in range(iterations):

    all_paths = []
    all_lengths = []

    for ant in range(num_ants):

        start = np.random.randint(n)

        path = [start]

        visited = set(path)

        while len(path) < n:

            current = path[-1]

            probabilities = []

            candidates = []

            for city in range(n):

                if city not in visited:

                    tau = pheromone[current][city]

                    eta = 1 / distance[current][city]

                    prob = (tau**alpha) * (eta**beta)

                    probabilities.append(prob)

                    candidates.append(city)

            probabilities = np.array(probabilities)

            probabilities /= probabilities.sum()

            next_city = np.random.choice(candidates, p=probabilities)

            path.append(next_city)

            visited.add(next_city)

        length = tour_length(path)

        all_paths.append(path)

        all_lengths.append(length)

        if length < best_length:

            best_length = length

            best_path = path

    # ----------------------------
    # Evaporation
    # ----------------------------
    pheromone *= (1 - rho)

    # ----------------------------
    # Deposit Pheromone
    # ----------------------------
    for path, length in zip(all_paths, all_lengths):

        deposit = Q / length

        for i in range(n - 1):

            a = path[i]

            b = path[i + 1]

            pheromone[a][b] += deposit

            pheromone[b][a] += deposit

        pheromone[path[-1]][path[0]] += deposit

        pheromone[path[0]][path[-1]] += deposit

print("Best Tour:", best_path)
print("Best Distance:", best_length)
