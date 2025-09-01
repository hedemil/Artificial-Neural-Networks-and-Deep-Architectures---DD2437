import numpy as np
import random as rd

random_seed = 42
rd.seed(random_seed)
np.random.seed(random_seed)

class Hopfield:
    def __init__(self, N: int):
        self.size = N
        self.weights = np.zeros((N, N))

    def train(self, patterns: np.ndarray, random_weights: bool = False, symmetric_weights: bool = False):
        self.weights.fill(0)  # Reset weights
        if random_weights:
            self.weights = np.random.normal(0, 1, (self.size, self.size))
        else:
            for pattern in patterns:
                pattern = pattern.reshape(-1, 1)
                self.weights += np.outer(pattern, pattern.T)

        if symmetric_weights:
            self.weights = 0.5 * (self.weights + self.weights.T)
            asymmetry = np.max(np.abs(self.weights - self.weights.T))
            print(f"Max asymmetry in symmetric case: {asymmetry}")

    def recall(self, pattern: np.ndarray, max_iter: int = 10000, update_mode: str = 'synchronous', save_states: bool = False, return_energy: bool = False, return_attractors: bool = False):
        current_state = pattern.copy()
        states = []
        energy_list = []
        attractor = None
        stable_count = 0
        current_energy = self.calc_energy(current_state)
        energy_list.append(current_energy)

        for iteration in range(max_iter):
            old_state = current_state.copy()

            if update_mode == 'asynchronous':
                i = rd.randint(0, self.size - 1)
                activation = np.dot(self.weights[i], current_state)
                current_state[i] = 1 if activation >= 0 else -1
            elif update_mode == 'synchronous':
                activations = np.dot(self.weights, old_state)
                current_state = np.where(activations >= 0, 1, -1)
            else:
                raise ValueError("update_mode must be either 'synchronous' or 'asynchronous'")

            current_energy = self.calc_energy(current_state)
            energy_list.append(current_energy)

            if np.array_equal(old_state, current_state):
                stable_count += 1
                if stable_count >= self.size:
                    attractor = current_state.copy()
                    break
            else:
                stable_count = 0

            if save_states and iteration % 500 == 0:
                states.append(current_state.copy())

        if save_states and return_energy:
            return current_state, states, energy_list
        elif save_states:
            return current_state, states
        elif return_energy:
            return current_state, energy_list
        elif return_attractors:
            return attractor
        else:
            return current_state

    def calc_energy(self, pattern: np.ndarray) -> float:
        return -np.dot(np.dot(self.weights, pattern), pattern)

    def is_stable(self, pattern: np.ndarray) -> bool:
        return np.array_equal(pattern, self.recall(pattern))