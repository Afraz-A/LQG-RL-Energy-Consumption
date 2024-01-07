from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools

class LQGMDP:
    def __init__(self, T, E_max, N_max, alpha):
        self.T = T
        self.E_max = E_max
        self.N_max = N_max
        self.alpha = alpha
        self.actions = [0, 1]
        self.Q = {(state, action): 0 for state in itertools.product(range(T + 1), range(E_max + 1)) for action in self.actions}
        self.rewards_history = []

        # define controlled dynamics matrices
        self.A = np.array([[1, 0], [0, 1]])  # state matrix
        self.B = np.array([[1], [0]])        # input-state matrix
        self.C = np.array([[1, 0]])           # state-output matrix
        self.W = np.array([[0.01, 0], [0, 0.01]])  # disturbance covariance matrix
        self.V = np.array([[0.1]])           # measurement noise covariance matrix

    def get_policy(self, state):
        epsilon = 0.01
        state_tuple = tuple(map(int, state))  # convert state components to integers
        if random.random() < epsilon:
            return random.choice(self.actions)  # exploration
        else:
            return max(self.actions, key=lambda a, s=state_tuple: self.Q[(s, a)])  # exploitation


    def transition(self, state, action):
        t, e = state
        tPrime = t - 1
        ePrime = e - self.E_cost(action)
        x = np.array([t, e])  # state vector
        u = action  # control input
        w = multivariate_normal(mean=np.zeros(len(x)), cov=self.W).rvs()  # stochastic disturbance
        xPrime = (tPrime, ePrime)
        return xPrime, None  # return xPrime and None as a placeholder for measurement

    def E_cost(self, action):
        if action == 0:
            return 0.1  # adjust actual energy cost for action 0
        elif action == 1:
            return 0.2  # adjust actual energy cost for action 1
        else:
            return 0.0  # placeholder for other actions

    def lqg_cost(self, state, action):
        x, u = np.array(state), action
        Q = np.array([[1, 0], [0, 1]])  # state cost matrix
        R = 1  # Control cost
        return np.dot(np.dot(x.T, Q), x) + R * u**2  # quadratic cost

    def update_Q(self, state, action, next_state, reward, episode):
        learning_rate = 0.45
        discount_factor = 1

        x, u = state, action
        xPrime, _ = self.transition(state, action)
        print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        x = tuple(map(int, x))  # convert state components to integers
        xPrime = tuple(map(int, xPrime))  # convert state components to integers

        print(f"Q-values before update: {self.Q[x, 0]} (action 0), {self.Q[x, 1]} (action 1)")

        best_next_action = max(self.actions, key=lambda a, ns=xPrime: self.Q[ns, a])
        self.Q[x, u] += learning_rate * (
            reward + discount_factor * self.lqg_cost(np.array(x), u) + self.Q[xPrime, best_next_action] - self.Q[x, u]
        )

        print(f"Q-values after update: {self.Q[x, 0]} (action 0), {self.Q[x, 1]} (action 1)")

    def run_simulation(self):
        num_of_episodes = 2000
        for episode in range(num_of_episodes):
            state = (self.T, self.E_max)  # initial state
            total_reward = 0
            for _ in range(self.T):
                action = self.get_policy(state)
                next_state, _ = self.transition(state, action)
                reward = -self.lqg_cost(state, action)  # (-) to minimize cost
                total_reward += reward
                self.update_Q(state, action, next_state, reward, episode)
                state = next_state
                # checks remaining energy budget, adjusts T
                if state[1] <= 0:
                    break  # stops episode if energy budget exhausted
            self.rewards_history.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}/{num_of_episodes}, Total Reward: {total_reward}")
                print(f"Q-values for state {state}: {self.Q[(tuple(map(int, state)), 0)]} (action 0), {self.Q[(tuple(map(int, state)), 1)]} (action 1)")
        self.plot_rewards()

    def plot_rewards(self):
          plt.plot(np.convolve(self.rewards_history, np.ones(100)/100, mode='valid'))
          plt.title('Smoothed Rewards over Episodes')
          plt.xlabel('Episode')
          plt.ylabel('Total Reward (Smoothed)')
          plt.draw()
          plt.pause(0.001)
          plt.show()

T = 10
E_max = 20
N_max = 5
alpha = 0.1

LQGMDP(T, E_max, N_max, alpha).run_simulation()
