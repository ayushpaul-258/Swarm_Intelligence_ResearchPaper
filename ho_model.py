import numpy as np

class HippopotamusOptimization:

    def __init__(self, obj_func, lb, ub,
                 dim,
                 pop_size=30,
                 max_iter=200):

        self.obj = obj_func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.N = pop_size
        self.max_iter = max_iter

    def initialize(self):
        return self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)

    def optimize(self):

        X = self.initialize()

        fitness = np.array([self.obj(x) for x in X])

        best_index = np.argmin(fitness)
        best = X[best_index].copy()
        best_fit = fitness[best_index]

        for t in range(self.max_iter):

            # ----------------------------
            # Phase 1
            # ----------------------------

            for i in range(self.N):

                r = np.random.rand(self.dim)
                I = np.random.randint(1,3)

                new = X[i] + r*(best - I*X[i])

                new = np.clip(new,self.lb,self.ub)

                f = self.obj(new)

                if f < fitness[i]:
                    X[i] = new
                    fitness[i] = f

            # ----------------------------
            # Phase 2
            # ----------------------------

            predator = self.lb + np.random.rand(self.dim)*(self.ub-self.lb)

            for i in range(self.N):

                D = np.abs(predator - X[i])

                levy = np.random.standard_cauchy(self.dim)

                new = X[i] + levy/(D+1e-9)

                new = np.clip(new,self.lb,self.ub)

                f = self.obj(new)

                if f < fitness[i]:
                    X[i]=new
                    fitness[i]=f

            # ----------------------------
            # Phase 3
            # ----------------------------

            scale = 1 - t/self.max_iter

            for i in range(self.N):

                step = np.random.randn(self.dim)

                new = X[i] + scale*step

                new = np.clip(new,self.lb,self.ub)

                f = self.obj(new)

                if f < fitness[i]:
                    X[i]=new
                    fitness[i]=f

            idx = np.argmin(fitness)

            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                best = X[idx].copy()

        return best, best_fit
