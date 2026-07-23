import numpy as np


# ---------------------------------------------------
# Example Objective Function
# Sphere Function (Minimum = 0)
# ---------------------------------------------------
def sphere(x):
    return np.sum(x ** 2)


# ---------------------------------------------------
# Harris Hawks Optimization
# ---------------------------------------------------
def HHO(obj_func, dim, lb, ub, num_hawks=30, max_iter=200):

    # -----------------------------
    # Initialize Hawks Randomly
    # -----------------------------
    hawks = np.random.uniform(lb, ub, (num_hawks, dim))

    # Fitness of each hawk
    fitness = np.array([obj_func(h) for h in hawks])

    # Rabbit = Best Solution
    rabbit_index = np.argmin(fitness)
    rabbit = hawks[rabbit_index].copy()
    rabbit_fitness = fitness[rabbit_index]

    # -------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------
    for t in range(max_iter):

        # Mean position of all hawks
        X_mean = np.mean(hawks, axis=0)

        # Update every hawk
        for i in range(num_hawks):

            # ---------------------------------------------
            # Rabbit escaping energy
            # E = 2*E0*(1-t/T)
            # ---------------------------------------------
            E0 = 2 * np.random.rand() - 1
            E = 2 * E0 * (1 - t / max_iter)

            # Random probability
            q = np.random.rand()

            # ==================================================
            # EXPLORATION PHASE
            # ==================================================
            if abs(E) >= 1:

                if q < 0.5:

                    # -------------------------------
                    # Strategy 1
                    # Random Hawk
                    # -------------------------------
                    rand_index = np.random.randint(num_hawks)
                    X_rand = hawks[rand_index]

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    hawks[i] = X_rand - r1 * np.abs(
                        X_rand - 2 * r2 * hawks[i]
                    )

                else:

                    # -------------------------------
                    # Strategy 2
                    # Search around average position
                    # -------------------------------
                    r3 = np.random.rand()
                    r4 = np.random.rand()

                    hawks[i] = (
                        rabbit
                        - X_mean
                        - r3 * (lb + r4 * (ub - lb))
                    )

            # ==================================================
            # EXPLOITATION PHASE
            # ==================================================
            else:

                r = np.random.rand()

                # Jump strength
                J = 2 * (1 - np.random.rand())

                # ------------------------------------------------
                # Case 1
                # Soft Besiege
                # ------------------------------------------------
                if r >= 0.5 and abs(E) >= 0.5:

                    deltaX = rabbit - hawks[i]

                    hawks[i] = (
                        deltaX
                        - E * np.abs(J * rabbit - hawks[i])
                    )

                # ------------------------------------------------
                # Case 2
                # Hard Besiege
                # ------------------------------------------------
                elif r >= 0.5 and abs(E) < 0.5:

                    deltaX = rabbit - hawks[i]

                    hawks[i] = (
                        rabbit
                        - E * np.abs(deltaX)
                    )

                # ------------------------------------------------
                # Case 3
                # Soft Besiege + Rapid Dive
                # ------------------------------------------------
                elif r < 0.5 and abs(E) >= 0.5:

                    Y = rabbit - E * np.abs(
                        J * rabbit - hawks[i]
                    )

                    # Simple Gaussian approximation of Levy Flight
                    Z = Y + np.random.randn(dim)

                    if obj_func(Y) < fitness[i]:
                        hawks[i] = Y
                    elif obj_func(Z) < fitness[i]:
                        hawks[i] = Z

                # ------------------------------------------------
                # Case 4
                # Hard Besiege + Rapid Dive
                # ------------------------------------------------
                else:

                    Y = rabbit - E * np.abs(
                        J * rabbit - X_mean
                    )

                    Z = Y + np.random.randn(dim)

                    if obj_func(Y) < fitness[i]:
                        hawks[i] = Y
                    elif obj_func(Z) < fitness[i]:
                        hawks[i] = Z

            # Keep inside bounds
            hawks[i] = np.clip(hawks[i], lb, ub)

            # Evaluate Fitness
            fitness[i] = obj_func(hawks[i])

            # Update Rabbit
            if fitness[i] < rabbit_fitness:
                rabbit = hawks[i].copy()
                rabbit_fitness = fitness[i]

    return rabbit, rabbit_fitness


# ---------------------------------------------------
# Example Run
# ---------------------------------------------------
best_pos, best_fit = HHO(
    sphere,
    dim=30,
    lb=-100,
    ub=100,
    num_hawks=30,
    max_iter=500,
)

print("Best Position:\n", best_pos)
print("Best Fitness:", best_fit)
