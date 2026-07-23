import numpy as np


# ---------------------------------------------------
# Example Objective Function
# Sphere Function
# ---------------------------------------------------
def sphere(x):
    return np.sum(x ** 2)


# ---------------------------------------------------
# Grey Wolf Optimizer
# ---------------------------------------------------
def GWO(obj_func, dim, lb, ub, num_wolves=30, max_iter=200):

    # ---------------------------------------
    # Initialize Wolves
    # ---------------------------------------
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))

    # Fitness
    fitness = np.array([obj_func(w) for w in wolves])

    # ---------------------------------------
    # Determine Alpha, Beta, Delta
    # ---------------------------------------
    sorted_index = np.argsort(fitness)

    alpha = wolves[sorted_index[0]].copy()
    beta = wolves[sorted_index[1]].copy()
    delta = wolves[sorted_index[2]].copy()

    alpha_score = fitness[sorted_index[0]]

    # ---------------------------------------
    # Main Loop
    # ---------------------------------------
    for t in range(max_iter):

        # Linearly decrease a
        a = 2 - (2 * t / max_iter)

        # Update every wolf
        for i in range(num_wolves):

            # -------------------------------------
            # Alpha
            # -------------------------------------
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = np.abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            # -------------------------------------
            # Beta
            # -------------------------------------
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            # -------------------------------------
            # Delta
            # -------------------------------------
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            # -------------------------------------
            # Average of three leaders
            # -------------------------------------
            wolves[i] = (X1 + X2 + X3) / 3

            # Keep inside search space
            wolves[i] = np.clip(wolves[i], lb, ub)

        # ---------------------------------------
        # Evaluate Population
        # ---------------------------------------
        fitness = np.array([obj_func(w) for w in wolves])

        # ---------------------------------------
        # Update Alpha Beta Delta
        # ---------------------------------------
        sorted_index = np.argsort(fitness)

        alpha = wolves[sorted_index[0]].copy()
        beta = wolves[sorted_index[1]].copy()
        delta = wolves[sorted_index[2]].copy()

        alpha_score = fitness[sorted_index[0]]

    return alpha, alpha_score


# ---------------------------------------------------
# Example Run
# ---------------------------------------------------
best_pos, best_fit = GWO(
    sphere,
    dim=30,
    lb=-100,
    ub=100,
    num_wolves=30,
    max_iter=500,
)

print("Best Position:\n", best_pos)
print("Best Fitness:", best_fit)
