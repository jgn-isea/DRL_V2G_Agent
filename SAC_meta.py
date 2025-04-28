import pickle
import warnings

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import argparse
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from env.EVChargingEnv import EVChargingEnvContinuous as EVChargingEnv
from helper import *


@tf.keras.utils.register_keras_serializable()
class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_width=256, max_action=1.0, hidden_depth=4):
        super(Actor, self).__init__()
        self.max_action = max_action  # output bounded between [-1, 1]
        # Define layers
        self.l = [layers.Dense(hidden_width, activation='relu') for _ in range(hidden_depth)]
        self.mean_layer = layers.Dense(action_dim)
        self.log_std_layer = layers.Dense(action_dim)

    def call(self, x, mask, alpha, alpha_embedding, deterministic=False, with_logprob=True):
        x = tf.where(mask > 0, x, tf.zeros_like(x))

        if alpha_embedding:
            if len(x.shape) > 1:
                alpha = tf.ones((len(x), 1), dtype=tf.float32) * alpha
                x = tf.concat([x, alpha], axis=1)
            else:
                x, alpha = tf.reshape(x, (1, -1)), tf.reshape(alpha, (1, -1))
                x = tf.concat([x, alpha], axis=1)

        # Forward pass through the network
        for l in self.l:
            x = l(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std to ensure stability
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)

        # Create a Gaussian distribution
        if deterministic:
            a = mean
        else:
            dist = tf.random.normal(tf.shape(mean))
            a = mean + dist * std

        if with_logprob:
            # Compute log probability
            log_pi = -0.5 * ((a - mean) / std) ** 2 - tf.math.log(std * tf.sqrt(2 * np.pi))
            log_pi = tf.reduce_sum(log_pi, axis=1, keepdims=True)
            # Adjustment for Tanh squashing
            log_pi -= tf.reduce_sum(2 * (np.log(2) - a - tf.nn.softplus(-2 * a)), axis=1, keepdims=True)
        else:
            log_pi = None

        # Use tanh to bound the actions and scale them
        a = self.max_action * tf.tanh(a)
        return a, log_pi


@tf.keras.utils.register_keras_serializable()
class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_width, hidden_depth):
        super(Critic, self).__init__()
        # Q1 network
        self.l1 = [layers.Dense(hidden_width, activation='relu') for _ in range(hidden_depth)]
        self.l1_out = layers.Dense(1)  # Output layer for Q1

        # Q2 network
        self.l2 = [layers.Dense(hidden_width, activation='relu') for _ in range(hidden_depth)]
        self.l2_out = layers.Dense(1)  # Output layer for Q2

    def call(self, s, a, mask, alpha, alpha_embedding):
        s = tf.where(mask > 0, s, tf.zeros_like(s))

        # Concatenate state and action inputs
        alpha = tf.ones((len(s), 1), dtype=tf.float32) * alpha
        if alpha_embedding:
            s_a = tf.concat([s, a, alpha], axis=1)
        else:
            s_a = tf.concat([s, a], axis=1)

        # Q1 forward pass
        for l in self.l1:
            s_a = l(s_a)
        q1 = self.l1_out(s_a)

        # Q2 forward pass
        for l in self.l2:
            s_a = l(s_a)
        q2 = self.l2_out(s_a)

        return q1, q2


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, mask_dim):
        # Maximum size of the buffer
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        # Allocate memory for states, actions, rewards, next states, and done flags
        self.s = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s_ = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dw = np.zeros((self.max_size, 1), dtype=np.float32)
        self.mask = np.zeros((self.max_size, mask_dim), dtype=np.float32)
        self.mask_ = np.zeros((self.max_size, mask_dim), dtype=np.float32)


    def store(self, s, a, r, s_, dw, mask, mask_):
        # Store a transition in the buffer
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.mask[self.count] = mask
        self.mask_[self.count] = mask_
        # Update the count and size of the buffer
        self.count = (self.count + 1) % self.max_size  # Reset to 0 when max_size is reached
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # Randomly sample a batch of transitions
        index = np.random.choice(self.size, size=batch_size, replace=False)
        batch_s = tf.convert_to_tensor(self.s[index], dtype=tf.float32)
        batch_a = tf.convert_to_tensor(self.a[index], dtype=tf.float32)
        batch_r = tf.convert_to_tensor(self.r[index], dtype=tf.float32)
        batch_s_ = tf.convert_to_tensor(self.s_[index], dtype=tf.float32)
        batch_dw = tf.convert_to_tensor(self.dw[index], dtype=tf.float32)
        batch_mask = tf.convert_to_tensor(self.mask[index], dtype=tf.float32)
        batch_mask_ = tf.convert_to_tensor(self.mask_[index], dtype=tf.float32)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_mask, batch_mask_


class Meta_SACAgent:
    """
    The Soft Actor-Critic (SAC) algorithm. This class implements the actor-critic architecture with twin critics, target networks, and automatic entropy tuning.
    """
    def __init__(self, state_dim, action_dim, mask_dim, max_action=1.0, batch_size=2048, gamma=0.99, tau=0.005, lr=3e-4, alpha=-1,
                 hidden_width=512, hidden_depth=5):
        self.max_action = max_action  # output bounded between [-1, 1]
        self.state_dim = state_dim  # Dimension of state space
        self.action_dim = action_dim  # Dimension of action space
        self.mask_dim = mask_dim
        self.hidden_width = hidden_width  # Number of neurons in hidden layers
        self.hidden_depth = hidden_depth  # Number of hidden layers
        self.batch_size = batch_size  # Batch size
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft target network update factor
        self.lr = lr  # Learning rate
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, mask_dim)  # Replay buffer
        self.alpha_embedding = True  # Whether to include alpha in the critic network
        self.var_hyperparameters = {"alpha": []}  # Dictionary to store the changing hyperparameters
        self.min_alpha = tf.convert_to_tensor(0.01, dtype=tf.float32)  # Minimum value of alpha

        # Initialize actor and critic networks
        self.actor = Actor(state_dim + 1 if self.alpha_embedding else state_dim, action_dim, self.hidden_width, self.max_action, self.hidden_depth)
        self.critic = Critic(state_dim + 1 if self.alpha_embedding else state_dim, action_dim, self.hidden_width, self.hidden_depth)
        self.critic_target = Critic(state_dim + 1 if self.alpha_embedding else state_dim, action_dim, self.hidden_width, self.hidden_depth)

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Log-alpha initialization for numerical stability
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, trainable=True)  # Start with alpha=1 (log_alpha=0)
        self.alpha = tf.exp(self.log_alpha)  # Entropy coefficient
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.history_meta_grad = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.meta_grad_decay_coef = 0.0
        self.meta_clip_norm = 0.05
        self.log_alpha_max = 0

        # A new actor for meta-learning
        self.new_actor = Actor(state_dim + 1 if self.alpha_embedding else state_dim, action_dim, self.hidden_width, self.max_action, self.hidden_depth)
        self.new_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Initialize networks by calling them with dummy data
        dummy_state = tf.convert_to_tensor(
            np.zeros((1, self.state_dim)), dtype=tf.float32)
        dummy_action = tf.convert_to_tensor(np.zeros((1, self.action_dim)), dtype=tf.float32)
        alpha = tf.zeros((1, 1), dtype=tf.float32)
        mask = tf.zeros((1, self.mask_dim), dtype=tf.float32)

        self.actor(dummy_state, mask, alpha, alpha_embedding=self.alpha_embedding)
        self.new_actor(dummy_state, mask, alpha, alpha_embedding=self.alpha_embedding)
        self.critic(dummy_state, dummy_action, mask, alpha, alpha_embedding=self.alpha_embedding)
        self.critic_target(dummy_state, dummy_action, mask, alpha, alpha_embedding=self.alpha_embedding)
        self.new_actor.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # Special buffer for initial states
        self.initial_state_buffer = ReplayBuffer(state_dim, 0, mask_dim)  # Only store states

    def fill_initial_state_buffer(self, env):
        """
        Fill the initial state buffer with initial states from the environment.
        """
        for _ in range(self.batch_size):
            init_state = env.reset(_)
            mask = env.get_mask()
            self.initial_state_buffer.store(init_state, None, None, None, None, mask, None)

    def meta_update_alpha(self):
        if self.replay_buffer.size < self.batch_size:
            return
        batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_mask, batch_mask_ = self.replay_buffer.sample(self.batch_size)
        batch_init_s, _, _, _, _, batch_init_mask, _ = self.initial_state_buffer.sample(self.batch_size)

        self.new_actor.set_weights(self.actor.get_weights())

        alpha = tf.exp(self.log_alpha)
        with tf.GradientTape() as tape:
            a, log_pi = self.new_actor(batch_s, batch_mask, alpha, alpha_embedding=self.alpha_embedding)
            q1, q2 = self.critic(batch_s, a, batch_mask, alpha, alpha_embedding=self.alpha_embedding)
            actor_loss = tf.reduce_mean(alpha * log_pi - tf.minimum(q1, q2))
        actor_grads = tape.gradient(actor_loss, self.new_actor.trainable_variables)
        self.new_actor_optimizer.apply_gradients(zip(actor_grads, self.new_actor.trainable_variables))

        with tf.GradientTape() as tape:
            alpha = tf.exp(self.log_alpha)
            deterministic_actions = self.new_actor(batch_init_s, batch_init_mask, alpha, alpha_embedding=self.alpha_embedding, deterministic=True, with_logprob=False)
            q1, q2 = self.critic(batch_init_s, deterministic_actions[0], batch_init_mask, alpha, alpha_embedding=self.alpha_embedding)
            meta_loss = -tf.reduce_mean(tf.minimum(q1, q2))
        alpha_grads = tape.gradient(meta_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        self.log_alpha.assign_add(self.history_meta_grad * self.meta_grad_decay_coef)
        self.history_meta_grad.assign(self.log_alpha.numpy())

        tf.clip_by_norm(self.log_alpha, self.meta_clip_norm)
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        self.log_alpha.assign(tf.clip_by_value(self.log_alpha, -np.inf, self.log_alpha_max))

        # Update alpha with regard to the minimum value of alpha
        self.alpha = max(self.min_alpha, tf.exp(self.log_alpha))

    def choose_action(self, s, mask, deterministic=False):
        if deterministic:
            alpha = tf.zeros((1, 1), dtype=tf.float32)
        else:
            alpha = tf.ones((1, ), dtype=tf.float32) * self.alpha
        a, _ = self.actor(s, mask, alpha, alpha_embedding=self.alpha_embedding, deterministic=deterministic, with_logprob=False)  # Get action from actor
        return a.numpy().flatten()

    def learn(self):
        if self.replay_buffer.size < self.batch_size:
            return
        self.meta_update_alpha()

        # Sample a batch of transitions from the replay buffer
        batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_mask, batch_mask_ = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape(persistent=True) as tape:
            # Compute target actions and log probabilities
            batch_a_, log_pi_ = self.actor(batch_s_, batch_mask_, self.alpha, alpha_embedding=self.alpha_embedding)
            # Compute target Q values
            target_q1, target_q2 = self.critic_target(batch_s_, batch_a_, batch_mask_, self.alpha, alpha_embedding=self.alpha_embedding)
            target_q = batch_r + self.gamma * (1 - batch_dw) * (
                    tf.minimum(target_q1, target_q2) - self.alpha * log_pi_)

            # Compute current Q values
            current_q1, current_q2 = self.critic(batch_s, batch_a, batch_mask, self.alpha, alpha_embedding=self.alpha_embedding)
            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(current_q1 - target_q)) + tf.reduce_mean(
                tf.square(current_q2 - target_q))

            # Actor loss
            a, log_pi = self.actor(batch_s, batch_mask, self.alpha, alpha_embedding=self.alpha_embedding)
            q1, q2 = self.critic(batch_s, a, batch_mask, self.alpha, alpha_embedding=self.alpha_embedding)
            actor_loss = tf.reduce_mean(self.alpha * log_pi - tf.minimum(q1, q2))

        # Update critic networks
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update actor network
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update target networks using soft updates
        for var, target_var in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)

    def save_networks_replaybuffer(self, folder, num_episodes=None):
        if num_episodes is not None:
            self.actor.save_weights(os.path.join(folder, f"SACActor_epi{num_episodes}.weights.h5"))
            self.critic.save_weights(os.path.join(folder, f"SACCritic_epi{num_episodes}.weights.h5"))
            self.critic_target.save_weights(os.path.join(folder, f"SACCriticTarget_epi{num_episodes}.weights.h5"))
            with open(os.path.join(folder, f"ReplayBuffer_epi{num_episodes}.pkl"), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
            with open(os.path.join(folder, f"ReplayBuffer_InitalState_epi{num_episodes}.pkl"), 'wb') as f:
                pickle.dump(self.initial_state_buffer, f)
        else:
            self.actor.save_weights(os.path.join(folder, "final_SACActor.weights.h5"))
            self.critic.save_weights(os.path.join(folder, "final_SACCritic.weights.h5"))
            self.critic_target.save_weights(os.path.join(folder, "final_SACCriticTarget.weights.h5"))
            with open(os.path.join(folder, "final_ReplayBuffer.pkl"), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
            with open(os.path.join(folder, "final_ReplayBuffer_InitalState.pkl"), 'wb') as f:
                pickle.dump(self.initial_state_buffer, f)

    def load_networks_replaybuffer(self, folder, num_episodes: int = None, eval=False):
        if num_episodes is not None:
            self.actor.load_weights(os.path.join(folder, f"SACActor_epi{num_episodes}.weights.h5"))
            self.new_actor.set_weights(self.actor.get_weights())
            self.critic.load_weights(os.path.join(folder, f"SACCritic_epi{num_episodes}.weights.h5"))
            self.critic_target.load_weights(os.path.join(folder, f"SACCriticTarget_epi{num_episodes}.weights.h5"))
            try:
                with open(os.path.join(folder, f"ReplayBuffer_InitalState_epi{num_episodes}.pkl"), 'rb') as f:
                    self.initial_state_buffer = pickle.load(f)
            except FileNotFoundError:
                warnings.warn("Initial state buffer not found. New initial state buffer should be generated.")
            if not eval:
                with open(os.path.join(folder, f"ReplayBuffer_epi{num_episodes}.pkl"), 'rb') as f:
                    self.replay_buffer = pickle.load(f)

        else:
            self.actor.load_weights(os.path.join(folder, "final_SACActor.weights.h5"))
            self.new_actor.set_weights(self.actor.get_weights())
            self.critic.load_weights(os.path.join(folder, "final_SACCritic.weights.h5"))
            self.critic_target.load_weights(os.path.join(folder, "final_SACCriticTarget.weights.h5"))
            try:
                with open(os.path.join(folder, "final_ReplayBuffer_InitalState.pkl"), 'rb') as f:
                    self.initial_state_buffer = pickle.load(f)
            except FileNotFoundError:
                warnings.warn("Initial state buffer not found. New initial state buffer should be generated.")
            if not eval:
                with open(os.path.join(folder, "final_ReplayBuffer.pkl"), 'rb') as f:
                    self.replay_buffer = pickle.load(f)

    def save_var_hyperparameters(self):
        """ Save the changing hyperparameters in a DataFrame"""
        for each_var in self.var_hyperparameters.keys():
            self.var_hyperparameters[each_var].append(getattr(self, each_var))

    def save_hyperparameters(self, folder, num_episodes: int = None):
        hyperparameter = pd.DataFrame({
            'state_dim': [self.state_dim],
            'action_dim': [self.action_dim],
            'mask_dim': [self.mask_dim],
            'max_action': [self.max_action],
            'hidden_width': [self.hidden_width],
            'hidden_depth': [self.hidden_depth],
            'batch_size': [self.batch_size],
            'gamma': [self.gamma],
            'tau': [self.tau],
            'lr': [self.lr],
            'alpha_embedding': [self.alpha_embedding],
            'alpha': [self.alpha],
            'min_alpha': [self.min_alpha],
            'log_alpha': [self.log_alpha]
        })
        if num_episodes is not None:
            hyperparameter.to_pickle(os.path.join(folder, f"hyperparameters_epi{num_episodes}.pkl"))
        else:
            hyperparameter.to_pickle(os.path.join(folder, "final_hyperparameters.pkl"))

    def load_hyperparameters(self, folder, num_episodes: int = None):
        if num_episodes is not None:
            hyperparameter = pd.read_pickle(os.path.join(folder, f"hyperparameters_epi{num_episodes}.pkl"))
        else:
            hyperparameter = pd.read_pickle(os.path.join(folder, "final_hyperparameters.pkl"))
        for each_var in hyperparameter.columns:
            setattr(self, each_var, hyperparameter.loc[0, each_var])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SAC agent for EV charging optimization.")
    # parameters training
    parser.add_argument('--epi_random', type=int, default=0, help='Number of episodes with random action')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of episodes for training')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=1000000, help='Interval for saving models')
    parser.add_argument('--eval_interval', type=int, default=500, help='Interval for evaluating the policy')
    parser.add_argument('--SAC_model', type=str, help='load a pre-trained SAC model')
    parser.add_argument('--alpha_d', type=float, default=1.0, help='Weight of degradation cost')
    parser.add_argument('--beta_ra', type=float, default=0.1, help='Range anxiety coefficient')
    parser.add_argument('--without_prediction', action='store_false', help='Use prediction for evaluation')
    parser.add_argument('--hidden_width', type=int, default=512, help='Number of neurons in hidden layers')
    parser.add_argument('--hidden_depth', type=int, default=5, help='Number of hidden layers')
    # hyperparameters SAC
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for the Actor')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor for future rewards')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target networks')
    parser.add_argument('--alpha', type=float, default=-1, help='Entropy coefficient')

    args = parser.parse_args()
    use_prediction = args.without_prediction

    # initialize environment, agent, and evaluation environment IDT
    env = EVChargingEnv(forecast_horizon=24, t_step=0.25, alpha_degradation=args.alpha_d, beta_range_anxiety=args.beta_ra)
    env_eval = EVChargingEnv(forecast_horizon=24, alpha_degradation=args.alpha_d, beta_range_anxiety=args.beta_ra, eval=True)
    env_eval.load_dataset(data_test=env.data_test)

    # Initialize the SAC agent
    agent = Meta_SACAgent(state_dim=env.state_size, action_dim=env.action_size, mask_dim=env.state_size, max_action=1, batch_size=args.batch_size,
                          gamma=args.gamma, tau=args.tau, lr=args.learning_rate,
                          hidden_width=args.hidden_width, hidden_depth=args.hidden_depth)
    if args.SAC_model is not None:
        # Load the history of the training
        reward_log, time_log, soc_log, eval_log, reference_results, agent = load_model(agent, env, args.SAC_model)
        episodes_done = len(reward_log)
        if reference_results is None:
            reference_results, total_reward_results = get_reference_results(env)
        else:
            total_reward_results = {"MILP Train": reference_results.loc["total_reward", "MILP train"],
                                    "MILP Test": reference_results.loc["total_reward", "MILP test"],
                                    "Uncontrolled Train": reference_results.loc["total_reward", "Uncontrolled train"],
                                    "Uncontrolled Test": reference_results.loc["total_reward", "Uncontrolled test"]}
    else:
        if os.path.exists("data/initial_state_buffer.pkl"):
            with open("data/initial_state_buffer.pkl", "rb") as f:
                agent.initial_state_buffer = pickle.load(f)
        else:
            agent.fill_initial_state_buffer(env)
            with open("data/initial_state_buffer.pkl", "wb") as f:
                pickle.dump(agent.initial_state_buffer, f)
        reward_log = []
        episodes_done = len(reward_log)
        time_log = []
        soc_log = []
        eval_log = []
        # Calculate best results with MILP and reference results with uncontrolled charging
        if os.path.exists("data/reference_results.pkl"):
            with open("data/reference_results.pkl", "rb") as f:
                reference_results = pickle.load(f)
            with open("data/total_reward_results.pkl", "rb") as f:
                total_reward_results = pickle.load(f)
        else:
            reference_results, total_reward_results = get_reference_results(env)
            with open("data/reference_results.pkl", "wb") as f:
                pickle.dump(reference_results, f)
            with open("data/total_reward_results.pkl", "wb") as f:
                pickle.dump(total_reward_results, f)

    episodes = args.episodes
    batch_size = args.batch_size

    # Initialize training plots
    fig, axes, lines, var_parameter = initialize_live_plot(var_parameter='alpha')
    plot_elements = (fig, axes, lines, var_parameter, "train_plot")

    # Initialize evaluation plots
    fig_eval, ax_eval, lines_eval = initialize_evaluation_live_plot()
    plot_elements_eval = (fig_eval, ax_eval, lines_eval, "eval_plot")

    for e in range(episodes_done, episodes_done + episodes):
        train_episode_mask(agent, env, e, reward_log=reward_log, time_log=time_log, soc_log=soc_log,
                           epi_random=args.epi_random, reference_values=total_reward_results,
                           plot_elements=plot_elements, use_prediction=use_prediction)
        print(f"Episode: {e + 1}/{episodes_done + episodes}, Total Reward: {reward_log[-1]}, Time: {time_log[-1]:.2f} s")

        # Evaluate the policy every eval_interval episodes
        if (e + 1) % args.eval_interval == 0:
            total_reward_eval, avg_reward_eval = evaluate_policy_mask(env_eval, agent, eval_log=eval_log,
                                                                      reference_values=total_reward_results,
                                                                      plot_elements=plot_elements_eval,
                                                                      use_prediction=use_prediction)
            print(f"Total Reward (Eval): {total_reward_eval}, Avg Reward (Eval): {avg_reward_eval}")

    plt.ioff()  # Turn off interactive mode
    save_results(reward_log, time_log, soc_log, eval_log, reference_results, env, agent, fig, fig_eval, args)
    plt.close()
