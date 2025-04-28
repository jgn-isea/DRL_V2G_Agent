import argparse  # 用于解析命令行参数
import os  # 操作文件路径和文件系统
import pickle  # 用于对象序列化与反序列化（保存和加载回放缓冲区等）
import numpy as np
import matplotlib

# 设置 Matplotlib 后端为 TkAgg，确保支持交互式绘图（特别是在桌面应用中）
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # 绘图工具包
import tensorflow as tf  # 深度学习库
from tensorflow.keras import layers  # 导入 Keras 的层，用于构建神经网络
from env.EVChargingEnv import EVChargingEnvDiscrete as EVChargingEnv  # 导入电动车充电环境
from helper import *  # 导入辅助函数，如 train_episode 和 evaluate_policy 等
import pandas as pd  # 数据处理库，用于存储和加载超参数等信息


# 定义 Dueling DQN 网络类
@tf.keras.utils.register_keras_serializable()  # 注册为 Keras 可序列化对象，方便保存和加载模型时序列化处理
class DuelDQNetwork(tf.keras.Model):
    """
    Dueling DQN 网络类，通过将 Q 值分解为状态价值 V(s) 和动作优势 A(s, a) 来提高性能。

    参数：
        state_dim (int): 状态空间维度，即输入层神经元数量。
        action_dim (int): 动作空间维度，即离散动作数量。
        hidden_width (int): 每个隐藏层中的神经元数量，默认值为 256。
        hidden_depth (int): 隐藏层的层数，默认值为 4。
    """

    def __init__(self, state_dim, action_dim, hidden_width=1024, hidden_depth=4):
        super(DuelDQNetwork, self).__init__()
        # 构建共享特征提取层（即多个全连接层），用于提取输入状态的特征信息
        # 注意：此处使用列表生成式创建多个 Dense 层，激活函数选用 ReLU
        self.feature_layers = [layers.Dense(hidden_width, activation='relu') for _ in range(hidden_depth - 1)]

        # 定义状态价值流（Value Stream）部分：
        # 首先通过一个隐藏层处理共享特征，再输出单一的状态价值 V(s)
        self.value_hidden = layers.Dense(hidden_width, activation='relu')  # 隐藏层用于处理共享特征
        self.value = layers.Dense(1)  # 输出层，得到状态价值（单个数值）

        # 定义动作优势流（Advantage Stream）部分：
        # 同样先通过一个隐藏层处理共享特征，再输出各个动作对应的优势值 A(s, a)
        self.advantage_hidden = layers.Dense(hidden_width, activation='relu')  # 隐藏层
        self.advantage = layers.Dense(action_dim)  # 输出层，输出每个动作的优势

    def call(self, x, mask):
        """
        定义前向传播计算过程，计算 Q 值：
            Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        其中，减去优势均值有助于减小模型不稳定性。

        参数：
            x (tf.Tensor): 输入状态张量，形状为 [batch_size, state_dim]。

        返回：
            q_values (tf.Tensor): 输出的 Q 值张量，形状为 [batch_size, action_dim]。
        """
        if mask is not None:
            x = tf.where(mask > 0, x, tf.zeros_like(x))

        # 依次通过每个共享特征层提取状态特征
        for layer in self.feature_layers:
            x = layer(x)
        # 通过状态价值流计算状态价值 V(s)
        v = self.value_hidden(x)
        v = self.value(v)
        # 通过动作优势流计算各个动作的优势 A(s, a)
        a = self.advantage_hidden(x)
        a = self.advantage(a)
        # 组合得到最终的 Q 值：使用 V(s) 加上去均值化的优势 A(s, a)
        q_values = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return q_values


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, mask_dim):
        # Maximum size of the buffer
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        # Allocate memory for states, actions, rewards, next states, and done flags
        self.s = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((self.max_size, 1), dtype=np.int32)  # ！！！！修改为 int32
        #self.a = np.zeros((self.max_size, action_dim), dtype=np.float32)
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
        batch_a = tf.convert_to_tensor(self.a[index], dtype=tf.int32)  # ！！！修改为 int32
        #batch_a = tf.convert_to_tensor(self.a[index], dtype=tf.float32)
        batch_r = tf.convert_to_tensor(self.r[index], dtype=tf.float32)
        batch_s_ = tf.convert_to_tensor(self.s_[index], dtype=tf.float32)
        batch_dw = tf.convert_to_tensor(self.dw[index], dtype=tf.float32)
        batch_mask = tf.convert_to_tensor(self.mask[index], dtype=tf.float32)
        batch_mask_ = tf.convert_to_tensor(self.mask_[index], dtype=tf.float32)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_mask, batch_mask_


# 定义 Duel DQN 代理类
class DuelDQNAgent:
    """
    Duel DQN 代理类，负责与环境的交互、选择动作、更新 Q 网络以及保存加载模型。

    参数：
        state_dim (int): 状态空间的维度（输入特征数量）。
        action_dim (int): 动作空间的维度（离散动作数量）。
        hidden_width (int): 隐藏层中神经元数量，默认 256。
        hidden_depth (int): 隐藏层的层数，默认 4。
        gamma (float): 折扣因子，用于未来奖励折扣，默认 0.99。
        epsilon (float): 初始探索率，默认 1.0（完全探索）。
        epsilon_min (float): 探索率下界，默认 0.01。
        epsilon_decay (float): 探索率衰减系数，每次更新时 epsilon 乘以该系数，默认 0.995。
        lr (float): 学习率，默认 0.0003。
        batch_size (int): 训练时从回放缓冲区中采样的批次大小，默认 2048。
        target_update_interval (int): 目标网络的更新频率（步数间隔），默认 100 步。
    """

    def __init__(self, state_dim, action_dim, mask_dim, hidden_width=1024, hidden_depth=4, gamma=1.0, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, lr=0.0003, batch_size=2048, target_update_interval=100):
        # 保存基本参数
        self.state_dim = state_dim
        self.action_dim = action_dim  # 动作空间大小: [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        self.mask_dim = mask_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.target_update_interval = target_update_interval
        self.lr = lr
        # 用于记录 epsilon 随训练变化的历史数据，便于后续分析
        self.var_hyperparameters = {"epsilon": []}

        # 初始化回放缓冲区，用于存储智能体与环境交互时获得的经验（状态、动作、奖励、下一状态和结束标志）
        self.replay_buffer = ReplayBuffer(state_dim, action_dim=1, mask_dim=self.mask_dim)

        # 初始化在线 Q 网络和目标 Q 网络，均采用 DuelDQNetwork 网络结构
        self.q_network = DuelDQNetwork(state_dim, action_dim, hidden_width, hidden_depth)
        self.target_network = DuelDQNetwork(state_dim, action_dim, hidden_width, hidden_depth)
        # 将目标网络权重初始化为与在线网络相同，以便开始时策略一致
        self.target_network.set_weights(self.q_network.get_weights())

        # 定义优化器，使用 Adam 优化器，并设置学习率
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # 记录训练步数，用于控制目标网络更新频率
        self.step_counter = 0

    def choose_action(self, s, mask, deterministic=False):
        """
        根据当前状态 s 选择动作，使用 ε-greedy 策略：
         - 以概率 epsilon 随机选择动作（探索），
         - 否则选择 Q 网络输出中最大的动作（利用）。

        参数：
            s (np.ndarray): 当前状态的数值表示。
            deterministic (bool): 是否禁用探索，True 表示只利用，不进行随机探索。

        返回：
            int: 被选动作在动作空间中的索引。
        """
        # 如果不使用确定性策略，并且随机数小于 epsilon，则进行随机探索
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        # 否则，将状态扩展成批次形式，并转换为 float32 类型，适配 TensorFlow 模型输入
        s = np.expand_dims(s, axis=0).astype(np.float32)
        # 通过在线 Q 网络计算当前状态下所有动作的 Q 值
        q_values = self.q_network(s, mask)
        # 选择具有最大 Q 值的动作并返回其索引
        return np.argmax(q_values.numpy())

    def learn(self):
        """
        学习函数，从回放缓冲区采样一批经验数据，并通过梯度下降方法更新在线 Q 网络的参数。
        同时更新 epsilon（减少探索率）并定期同步目标网络的权重。
        """
        # 如果回放缓冲区中的数据量不足一个批次大小，则跳过更新过程
        if self.replay_buffer.size < self.batch_size:
            return

        # 从回放缓冲区中随机采样一批训练数据
        batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_mask, batch_mask_ = self.replay_buffer.sample(self.batch_size)

        # 使用目标网络计算下一状态的 Q 值，并选择其中的最大值作为下一状态的估计最大 Q 值
        next_q_values = self.target_network(batch_s_, batch_mask_).numpy()
        max_next_q_values = np.max(next_q_values, axis=1)
        # 计算 TD 目标：当前奖励加上折扣后未来奖励的估计，注意使用 (1 - batch_dw) 去除终止状态
        targets = batch_r + self.gamma * max_next_q_values.reshape(-1, 1) * (1 - batch_dw)

        # 使用 TensorFlow 的 GradientTape 记录前向传播过程，以便自动求导
        with tf.GradientTape() as tape:
            # 计算在线网络当前状态的 Q 值预测
            # 1. 网络输出 (batch_size, action_dim)
            q_values = self.q_network(batch_s, batch_mask)
            # 2. 先将 batch_a squeeze 成 (batch_size,)
            batch_a_squeezed = tf.squeeze(batch_a, axis=1)
            # 3. One-hot 后得到 (batch_size, action_dim) 的矩阵
            one_hot_a = tf.one_hot(batch_a_squeezed, depth=self.action_dim)  # 直接使用 Tensor！！！！！！！！！！！
            #one_hot_a = tf.one_hot(batch_a_squeezed.numpy(), depth=self.action_dim)
            # 4. 点乘后对 action_dim 做 reduce_sum，只保留 (batch_size,) 或 (batch_size,1)
            #    常见用法：不加 keepdims => (batch_size,)
            #    如果想要 (batch_size,1)，可 keepdims=True
            action_q_values = tf.reduce_sum(q_values * one_hot_a, axis=1, keepdims=True)
            # 计算均方误差损失：目标 Q 值与当前 Q 值之间的误差
            loss = tf.reduce_mean(tf.square(targets - action_q_values))

        # 计算损失函数相对于在线网络参数的梯度
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        # 使用优化器应用梯度更新在线网络参数
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        # 更新 epsilon，使其逐步降低到最小探索率 epsilon_min
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.var_hyperparameters["epsilon"].append(self.epsilon)

        # 增加训练步数计数器，并在达到目标网络更新间隔时同步目标网络的参数
        self.step_counter += 1
        if self.step_counter % self.target_update_interval == 0:
            self.target_network.set_weights(self.q_network.get_weights())

    def save_networks_replaybuffer(self, folder, num_episodes=None):
        """
        将在线网络、目标网络以及回放缓冲区保存到指定文件夹中，便于训练过程的中断和恢复。

        参数：
            folder (str): 模型和数据保存的文件夹路径。
            num_episodes (int): 当前训练的 episode 编号，如果提供则在文件名中包含编号，否则保存最终模型。
        """
        if num_episodes is not None:  # 按 episode 保存
            self.q_network.save_weights(os.path.join(folder, f"DuelDQNetwork_epi{num_episodes}.weights.h5"))
            self.target_network.save_weights(os.path.join(folder, f"DuelTargetDQNetwork_epi{num_episodes}.weights.h5"))
            with open(os.path.join(folder, f"ReplayBuffer_epi{num_episodes}.pkl"), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
        else:  # 保存最终模型和回放缓冲区
            self.q_network.save_weights(os.path.join(folder, "final_DuelDQNetwork.weights.h5"))
            self.target_network.save_weights(os.path.join(folder, "final_DuelTargetDQNetwork.weights.h5"))
            with open(os.path.join(folder, "final_ReplayBuffer.pkl"), 'wb') as f:
                pickle.dump(self.replay_buffer, f)

    def load_networks_replaybuffer(self, folder, num_episodes=None, eval=False):
        """
        从指定文件夹加载在线网络、目标网络以及回放缓冲区（如果不是评估模式下则加载回放缓冲区）。

        参数：
            folder (str): 保存模型和数据的文件夹路径。
            num_episodes (int): 指定加载哪一个 episode 编号的模型和数据。
            eval (bool): 是否仅加载网络用于评估，默认 False 表示同时加载回放缓冲区。
        """
        # 为了确保网络结构正确，先用一个虚拟状态进行前向传播，避免加载权重时出现形状不匹配问题
        dummy_state = tf.convert_to_tensor(np.zeros((1, self.state_dim)), dtype=tf.float32)
        dummy_mask = tf.zeros((1, self.mask_dim), dtype=tf.float32)
        self.q_network(dummy_state, dummy_mask)
        self.target_network(dummy_state, dummy_mask)
        if num_episodes is not None:  # 如果指定了 episode 编号，则加载对应的模型文件
            self.q_network.load_weights(os.path.join(folder, f"DuelDQNetwork_epi{num_episodes}.weights.h5"))
            self.target_network.load_weights(os.path.join(folder, f"DuelTargetDQNetwork_epi{num_episodes}.weights.h5"))
            if not eval:  # 如果不是评估模式，则加载回放缓冲区数据
                with open(os.path.join(folder, f"ReplayBuffer_epi{num_episodes}.pkl"), 'rb') as f:
                    self.replay_buffer = pickle.load(f)
        else:  # 否则加载最终保存的模型和回放缓冲区
            self.q_network.load_weights(os.path.join(folder, "final_DuelDQNetwork.weights.h5"))
            self.target_network.load_weights(os.path.join(folder, "final_DuelTargetDQNetwork.weights.h5"))
            if not eval:
                with open(os.path.join(folder, "final_ReplayBuffer.pkl"), 'rb') as f:
                    self.replay_buffer = pickle.load(f)

    def save_var_hyperparameters(self):
        """ Save the changing hyperparameters in a DataFrame"""
        for each_var in self.var_hyperparameters.keys():
            self.var_hyperparameters[each_var].append(getattr(self, each_var))

    def save_hyperparameters(self, folder, num_episodes=None):
        """
        保存代理的超参数信息到文件中，便于后续分析和复现实验结果。

        参数：
            folder (str): 保存超参数的文件夹路径。
            num_episodes (int): 如果提供，则将超参数保存为特定 episode 的版本。
        """
        hyperparameter = pd.DataFrame({
            'state_dim': [self.state_dim],
            'action_dim': [self.action_dim],
            'mask_dim': [self.mask_dim],
            'hidden_width': [self.hidden_width],
            'hidden_depth': [self.hidden_depth],
            'gamma': [self.gamma],
            'epsilon': [self.epsilon],
            'epsilon_min': [self.epsilon_min],
            'epsilon_decay': [self.epsilon_decay],
            'lr': [self.lr],
            'batch_size': [self.batch_size],
            'target_update_interval': [self.target_update_interval]
        })
        if num_episodes is not None:
            hyperparameter.to_pickle(os.path.join(folder, f"hyperparameters_epi{num_episodes}.pkl"))
        else:
            hyperparameter.to_pickle(os.path.join(folder, "final_hyperparameters.pkl"))

    def load_hyperparameters(self, folder, num_episodes=None):
        """
        从文件加载超参数，并更新代理的内部变量。

        参数：
            folder (str): 超参数文件所在的文件夹路径。
            num_episodes (int): 指定加载哪一个 episode 的超参数文件。
        """
        if num_episodes is not None:
            hyperparameter = pd.read_pickle(os.path.join(folder, f"hyperparameters_epi{num_episodes}.pkl"))
        else:
            hyperparameter = pd.read_pickle(os.path.join(folder, "final_hyperparameters.pkl"))
        # 遍历所有超参数，将 DataFrame 中对应的值赋值给代理的属性
        for each_var in hyperparameter.columns:
            setattr(self, each_var, hyperparameter.loc[0, each_var])


# 主程序入口
if __name__ == "__main__":
    # 使用 argparse 定义并解析命令行参数，便于用户自定义训练过程中的超参数和设置
    parser = argparse.ArgumentParser(description="训练 Duel DQN 代理以优化电动车充电策略。")
    parser.add_argument('--epi_random', type=int, default=0, help='使用随机动作的 episode 数量')
    parser.add_argument('--episodes', type=int, default=20000, help='训练的总 episode 数量')
    parser.add_argument('--batch_size', type=int, default=2048, help='训练批次大小')
    parser.add_argument('--save_interval', type=int, default=1000000, help='保存模型的间隔（步数）')
    parser.add_argument('--eval_interval', type=int, default=500, help='评估策略的间隔（episode 数）')
    parser.add_argument('--DQN_model', type=str, help='加载预训练的 Duel DQN 模型路径')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='网络学习率')
    parser.add_argument('--gamma', type=float, default=1.0, help='未来奖励的折扣因子')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='探索率衰减系数')
    parser.add_argument('--target_update_interval', type=int, default=100, help='目标网络更新间隔')
    parser.add_argument('--hidden_width', type=int, default=1024, help='Number of neurons in hidden layers')
    parser.add_argument('--hidden_depth', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--alpha_d', type=float, default=1.0, help='Weight of degradation cost')
    parser.add_argument('--beta_ra', type=float, default=0.1, help='Range anxiety coefficient')
    parser.add_argument('--without_prediction', action='store_false', help='Use prediction for evaluation')

    args = parser.parse_args()
    use_prediction = args.without_prediction

    # 初始化训练环境：
    # 第一个环境用于训练，指定预测时域（forecast_horizon）、时间步长（t_step）以及电价数据文件
    env = EVChargingEnv(alpha_degradation=args.alpha_d, beta_range_anxiety=args.beta_ra, price_file="data/electricity_prices_ID1.csv")
    # 初始化评估环境，eval 参数为 True 表示该环境用于策略评估
    env_eval = EVChargingEnv(alpha_degradation=args.alpha_d, beta_range_anxiety=args.beta_ra, eval=True, price_file="data/electricity_prices_ID1.csv")
    # 评估环境加载训练环境中的测试数据，确保评估数据一致
    env_eval.load_dataset(data_test=env.data_test)

    # 初始化 Duel DQN 代理，传入状态空间维度、动作空间维度及其他超参数（批次大小、gamma、epsilon_decay、学习率、目标网络更新间隔等）
    agent = DuelDQNAgent(state_dim=env.state_size, action_dim=env.action_size, mask_dim=env.state_size,
                         hidden_width=args.hidden_width, hidden_depth=args.hidden_depth, batch_size=args.batch_size,
                         gamma=args.gamma, epsilon_decay=args.epsilon_decay, lr=args.learning_rate,
                         target_update_interval=args.target_update_interval)

    # 如果命令行参数中提供了预训练模型路径，则加载预训练模型和相关日志数据
    if args.DQN_model is not None:
        reward_log, time_log, soc_log, eval_log, reference_results, agent = load_model(agent, env, args.DQN_model)
        episodes_done = len(reward_log)
        if reference_results is None:
            # 如果未生成参考结果，则调用 MILP 模型获得参考结果
            reference_results, total_reward_results = get_reference_results(env)
        else:
            total_reward_results = {"MILP Train": reference_results.loc["total_reward", "MILP train"],
                                    "MILP Test": reference_results.loc["total_reward", "MILP test"],
                                    "Uncontrolled Train": reference_results.loc["total_reward", "Uncontrolled train"],
                                    "Uncontrolled Test": reference_results.loc["total_reward", "Uncontrolled test"]}
    else:
        # 如果没有加载预训练模型，则初始化空日志列表和参考结果
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

    # 初始化实时绘图，用于在训练过程中展示奖励、SOC（电池状态）等指标的变化曲线
    fig, axes, lines, var_parameter = initialize_live_plot()
    plot_elements = (fig, axes, lines, var_parameter, 'train_plot')
    # 初始化评估时的实时绘图
    fig_eval, ax_eval, lines_eval = initialize_evaluation_live_plot()
    plot_elements_eval = (fig_eval, ax_eval, lines_eval, 'eval_plot')

    # 开始训练循环，共进行 args.episodes 个 episode 的训练
    for e in range(episodes_done, episodes_done + episodes):
        # 调用辅助函数 train_episode 进行一次 episode 的训练，
        # 参数包括当前 episode 编号、日志列表以及实时绘图元素，训练过程中可能包含参考结果对比
        train_episode_mask(agent, env, e, reward_log=reward_log, time_log=time_log, soc_log=soc_log,
                           epi_random=args.epi_random, reference_values=total_reward_results,
                           plot_elements=plot_elements, use_prediction=use_prediction)
        print(f"Episode: {e + 1}/{episodes_done + episodes}, Total Reward: {reward_log[-1]}, Time: {time_log[-1]:.2f} s")

        # 每隔指定的 episode 数进行一次评估，评估时使用 eval 环境和 evaluate_policy 辅助函数
        if (e + 1) % args.eval_interval == 0:
            total_reward_eval, avg_reward_eval = evaluate_policy_mask(env_eval, agent, eval_log=eval_log,
                                                                      reference_values=total_reward_results,
                                                                      plot_elements=plot_elements_eval,
                                                                      use_prediction=use_prediction)
            print(f"Total Reward (Eval): {total_reward_eval}, Avg Reward (Eval): {avg_reward_eval}")

    # 训练完成后关闭实时绘图，保存训练结果、模型和超参数等数据
    plt.ioff()  # 关闭交互模式
    save_results(reward_log, time_log, soc_log, eval_log, reference_results, env, agent, fig, fig_eval, args)
    plt.close()
