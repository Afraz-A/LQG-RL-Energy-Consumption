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

    def get_policy(self, state, exploration_prob):
      state_tuple = tuple(state)
      epsilon = exploration_prob
      if random.random() < epsilon:
          return random.choice(self.actions)  # exploration
      else:
          return max(self.actions, key=lambda a: self.Q[(state_tuple, a)])  # exploitation

    def transition(self, state, action):
      t, e = state 
      tPrime = t - 1 
      ePrime = e - self.controlled_system_cost(state, action)  # Fix the method name
      return tPrime, ePrime


    def controlled_system_cost(self, state, action):
      t, e = state
      u = action

      # Define system dynamics matrices for a simple linear system
      A = np.array([[1, 0], [0, 0.9]])  # State transition matrix
      B = np.array([[0], [1]])  # Control input matrix
      Q = np.array([[1, 0], [0, 1]])  # State cost matrix
      R = 1  # Control cost

      # Calculate the next state using the system dynamics
      next_state = np.dot(A, np.array([t, e])) + np.dot(B, np.array([u]))

      # Calculate the instantaneous cost using the LQR cost function
      cost = np.dot(np.dot((np.array([t, e]) - next_state).T, Q), (np.array([t, e]) - next_state)) + R * u**2

      return cost

    def lqg_cost(self, state, action):
      t, e = state
      u = action
      Q = np.array([[1, 0], [0, 1]])  # state cost matrix
      R = 1  # control cost
      x = np.array([t, e])  # state vector
      return np.dot(np.dot(x.T, Q), x) + R * u**2  # quadratic cost

    def update_Q(self, state, action, next_state, reward, episode):
      learning_rate = 0.45
      discount_factor = 1

      state_tuple = tuple(state)
      next_state_tuple = tuple(next_state)

      if (state_tuple, 0) not in self.Q:
          self.Q[(state_tuple, 0)] = 0
      if (state_tuple, 1) not in self.Q:
          self.Q[(state_tuple, 1)] = 0
      if (next_state_tuple, 0) not in self.Q:
          self.Q[(next_state_tuple, 0)] = 0
      if (next_state_tuple, 1) not in self.Q:
          self.Q[(next_state_tuple, 1)] = 0

      print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
      print(f"Q-values before update: {self.Q[(state_tuple, 0)]} (action 0), {self.Q[(state_tuple, 1)]} (action 1)")

      best_next_action = max(self.actions, key=lambda a: self.Q[(next_state_tuple, a)])
      self.Q[(state_tuple, action)] += learning_rate * (
          reward + discount_factor * self.Q[(next_state_tuple, best_next_action)] - self.Q[(state_tuple, action)]
      )

      print(f"Q-values after update: {self.Q[(state_tuple, 0)]} (action 0), {self.Q[(state_tuple, 1)]} (action 1)")

    def kalman_filter(self, state, action, measurement):
      # Kalman filter parameters (replace with your specific matrices)
      A = np.array([[1, 0], [0, 0.9]])  # State transition matrix
      B = np.array([[0], [1]])  # Control input matrix
      W_covariance = np.array([[0.01, 0], [0, 0.01]])  # Disturbance covariance matrix (W)
      V_covariance = np.array([[0.1, 0], [0, 0.1]])  # Measurement noise covariance matrix (V)
      C = np.array([[1, 0], [0, 1]])  # State-output matrix

      # Predict (state prediction)
      state_pred = np.dot(A, state) + np.dot(B, np.array([action]))

      # Covariance prediction
      P_pred = np.dot(np.dot(A, W_covariance), A.T)

      # Update (measurement update)
      K = np.dot(np.dot(P_pred, C.T), np.linalg.inv(np.dot(np.dot(C, P_pred), C.T) + V_covariance))
      state_update = state_pred + np.dot(K, (measurement - np.dot(C, state_pred)))
      P_update = P_pred - np.dot(np.dot(K, C), P_pred)

      return state_update, P_update

    def run_simulation(self):
      num_of_episodes = 2000
      initial_learning_rate = 1.0
      initial_exploration_prob = 1.0
      min_learning_rate = 0.01
      min_exploration_prob = 0.01
    
      for episode in range(num_of_episodes):
          state = np.array([self.T, self.E_max])  # Initial state
          total_reward = 0
    
          # Exponential decay for learning rate and exploration probability
          learning_rate = max(min(initial_learning_rate * np.exp(-0.005 * episode), min_learning_rate), min_learning_rate)
          exploration_prob = max(min(initial_exploration_prob * np.exp(-0.005 * episode), min_exploration_prob), min_exploration_prob)
    
          for _ in range(self.T):
              action = self.get_policy(state, exploration_prob)
    
              # Simulate controlled system dynamics and measurement
              disturbance = np.random.multivariate_normal([0, 0], np.array([[0.01, 0], [0, 0.01]]))
              next_state_true = np.dot(np.array([[1, 0], [0, 0.9]]), state) + np.dot(np.array([[0], [1]]), np.array([action])) + disturbance
              measurement_noise = np.random.multivariate_normal([0, 0], np.array([[0.1, 0], [0, 0.1]]))
              measurement = np.dot(np.array([[1, 0], [0, 1]]), next_state_true) + measurement_noise
    
              # Kalman filter update
              state, _ = self.kalman_filter(state, action, measurement)
    
              reward = -self.lqg_cost(state, action)  # (-) to minimize cost
    
              # Additional penalty for negative rewards
              if reward < 0:
                  reward *= 0.1  # Adjust this multiplier based on your system's characteristics
    
              total_reward += reward
              self.update_Q(state, action, next_state_true, reward, episode)
    
              # checks remaining energy budget, adjusts T
              if state[1] <= 0:
                  break  # Stops episode if energy budget exhausted
    
          self.rewards_history.append(total_reward)
    
          if episode % 100 == 0:
              print(f"Episode {episode}/{num_of_episodes}, Total Reward: {total_reward}")
              print(f"Q-values for state {state}: {self.Q[(tuple(state), 0)]} (action 0), {self.Q[(tuple(state), 1)]} (action 1)")
    
      self.plot_rewards()


    def plot_rewards(self):
      plt.plot(np.convolve(self.rewards_history, np.ones(100)/100, mode='valid'))
      plt.title('Smoothed Rewards over Episodes')
      plt.xlabel('Episode')
      plt.ylabel('Total Reward (Smoothed)')
      plt.show()

T = 10
E_max = 20
N_max = 5
alpha = 0.1

LQGMDP(T, E_max, N_max, alpha).run_simulation()