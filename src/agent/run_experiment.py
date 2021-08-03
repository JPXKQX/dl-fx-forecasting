import warnings
import gym
import logging
import os
import numpy as np
import tensorflow as tf

from time import time, gmtime, strftime
from pathlib import Path
from src.data.constants import ROOT_DIR
from src.agent.ddqn_agent import DDQN
from src.visualization import agent_results
from gym.envs.registration import register

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
logger = logging.getLogger("RL Agent")


################################################################
################### Attributes  ################################
################################################################

gamma = .99
tau = 100

# Q-network hyperparameters
architecture = (256, 256)
learning_rate = 0.0001
dropout_rate = 0.1
l2_reg = 1e-6

# Experience Replay
replay_capacity = int(1e6)
batch_size = 4096

# e-greedy policy
epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_steps = 250
epsilon_exponential_decay = .99


def frictionless_rl_agent_5():
    register(
        id='fxtrading-v0',
        entry_point='src.agent.environment:TradingEnv',
        max_episode_steps=10000
    )
    env = gym.make('fxtrading-v0')
    env.seed(42)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    max_episode_steps = env.spec.max_episode_steps

    agent = DDQN(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        epsilon_exponential_decay=epsilon_exponential_decay,
        replay_capacity=replay_capacity,
        architecture=architecture,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        tau=tau,
        batch_size=batch_size
    )
    agent.online_network.summary()

    # Train Agent
    total_steps = 0
    max_episodes = 1000
    agent_pnl, market_pnl, alpha_pnl = [], [], []
    t0 = time()
    for episode in range(1, max_episodes + 1):
        this_state = env.reset()
        for episode_step in range(max_episode_steps):
            action = agent.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
            next_state, reward, done, _ = env.step(action)
        
            agent.memorize_transition(
                this_state, action, reward, next_state, 0.0 if done else 1.0
            )
            if agent.train:
                agent.experience_replay()
            if done:
                break
            this_state = next_state

        # get DataFrame with seqence of actions, returns and nav values
        result = env.env.ss.result()

        # Store episode results
        agent_pnl.append(result.pnl.sum())
        market_pnl.append(1e4 * (result.iloc[-1].ask - result.iloc[0].bid))
        alpha_pnl.append(agent_pnl[-1] - market_pnl[-1])

        if episode % 10 == 0:
            agent_pnl_ma10 = np.mean(agent_pnl[-10:])
            agent_pnl_ma100 = np.mean(agent_pnl[-100:])
            market_pnl_ma10 = np.mean(market_pnl[-10:])
            market_pnl_ma100 = np.mean(market_pnl[-100:])
            num_sessions_positive_alpha = np.sum([s > 0 for s in alpha_pnl[-100:]]) / \
                min(len(alpha_pnl), 100)
            logger.info(
                f"{episode:>4d} | {strftime('%H:%M:%S', gmtime(time() - t0))} | " \
                f"Agent: {agent_pnl_ma100:>6.2} ({agent_pnl_ma10:>6.2}) | " \
                f"Market: {market_pnl_ma100:>6.2} (" \
                f"{market_pnl_ma10:>6.2}) | Wins: {num_sessions_positive_alpha:>5.1%}" \
                f"| eps: {agent.epsilon:>6.3f}"
            )

        if episode % 100 == 0:
            # Cache results
            agent.online_network.save(
                Path(ROOT_DIR) / "models" / "agent" / f"q_network_frictionless_{episode}.h5"
            )
            results = pd.DataFrame({
                'Episode': list(range(1, episode+1)),
                'Agent': agent_pnl,
                'Market': market_pnl,
                'Difference': alpha_pnl
            }).set_index('Episode')
            results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
            results.info()
            results.to_csv(
                Path(ROOT_DIR) / "models" / "agent" / f"results_frictionless_{episode}.csv", 
                index=False
            )
    env.close()

    # Save results
    agent.online_network.save(
        ROOT_DIR / "models" / "agent" / "q_network_frictionless.h5"
    )
    results = pd.DataFrame({
        'Episode': list(range(1, episode+1)),
        'Agent': agent_pnl,
        'Market': market_pnl,
        'Difference': alpha_pnl
    }).set_index('Episode')
    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
    results.info()
    results.to_csv(
        Path(ROOT_DIR) / "models" / "agent" / "results_frictionless.csv", 
        index=False)

    agent_results.plot_results_agent(results, Path(ROOT_DIR) / "models" / "agent" )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        filename=Path(ROOT_DIR) / "models" / "agent" / "frictionless.log"
    )
    frictionless_rl_agent_5()
