import numpy as np
from harmony_search import evaluate_fitness

# CONFIG
W = 0.7       # Inertia
C1 = 1.5      # Cognitive (self)
C2 = 1.5      # Social (swarm)
THRESHOLD = 0.5
MAX_ITER = 20

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize PSO 
def initialize_particles(HM):
    num_particles, num_features = HM.shape
    particles = HM.copy()
    velocities = np.random.uniform(-1, 1, size=(num_particles, num_features))
    return particles, velocities

# PSO Update
def binary_pso(particles, velocities, X, y,  iterations=20, w=0.72, c1=1.49, c2=1.49):
    num_particles, num_features = particles.shape

    # Personal bests
    pbest = particles.copy()
    pbest_fitness = np.array([evaluate_fitness(p, X, y) for p in pbest])

    # Global best
    gbest_index = np.argmin(pbest_fitness)
    gbest = pbest[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]
    accuracy_pso = []
    feature_counts_pso = []

    for iteration in range(iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(num_features), np.random.rand(num_features)

            velocities[i] = (
                W * velocities[i]
                + C1 * r1 * (pbest[i] - particles[i])
                + C2 * r2 * (gbest - particles[i])
            )

            prob = sigmoid(velocities[i])
            particles[i] = np.where(prob > THRESHOLD, 1, 0)

            # Evaluate new particle
            fitness = evaluate_fitness(particles[i], X, y)

            # Update personal best
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness

                # Update global best
                if fitness < gbest_fitness:
                    gbest = particles[i].copy()
                    gbest_fitness = fitness
        # Track accuracy and features selected for plotting
        acc = 1 - gbest_fitness  # Fitness = α*(1 - acc) + β*(features)
        accuracy_pso.append(acc)
        feature_counts_pso.append(np.sum(gbest))
        print(f"Iteration {iteration+1:02} | gBest Fitness: {gbest_fitness:.4f} | Features: {np.sum(gbest)}")

    return gbest, gbest_fitness, accuracy_pso, feature_counts_pso
