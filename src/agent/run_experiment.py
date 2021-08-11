import warnings
import logging
import os
import click
import pandas as pd
import numpy as np
import tensorflow as tf

from time import time, gmtime, strftime
from typing import Tuple
from pathlib import Path
from src.data.constants import ROOT_DIR
from src.agent.ddqn_agent import DDQN
from src.agent.environment import TradingEnv
from src.visualization import agent_results

# TF details
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(gpus[1:], 'GPU')
print(f"There are a total of {len(gpus)} graphics cards.")
#for gpu in gpus[1:]: tf.config.experimental.set_memory_growth(gpu, True)

# Logging module
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
epsilon_end = .1
epsilon_decay_steps = 500
epsilon_exponential_decay = .995


def get_mode(scale: float) -> str:
    if scale == 1.0:
        return 'real_world'
    elif scale == 0.5:
        return 'half_way'
    elif scale == 0.:
        return 'frictionless'


def get_best_outcomes(result: pd.DataFrame) -> Tuple[int, int, float]:
    """ Compute the maximum potential benefit in an episode and counts the number of
    possible beneficial trades.

    Args:
        result (pd.DataFrame): Results of an episode. It must contain at least two
            columns for the bid and ask prices.

    Returns:
        int: number of long positions with benefits
        int: number of short positons with benefits
        float: maximum number of pip that could have been won in this episode
    """
    buys = result.bid[:-1].values - result.ask[1:].values
    sells = result.bid[1:].values - result.ask[:-1].values
    long_options = np.where(buys > 0)[0]
    short_options = np.where(sells > 0)[0]
    total_pips = buys[long_options] + sells[short_options]
    return len(long_options), len(short_options), total_pips


def rl_agent_5(scaling_difficulty: float = 0.0):
    env = TradingEnv(scaling_difficulty=scaling_difficulty, trading_sessions=25000)
    env.seed(42)

    mode_name = get_mode(scaling_difficulty)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    max_episode_steps = 20000

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
    agent_pnls, market_pnls, alpha_pnls, steps, n_sells, n_buys = [], [], [], [], [], []
    long_options, short_options,rf_pnls, regr_pnls, mlp_pnls = [], [], [], [], []
    potential_pips = []
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

        # get DataFrame of a seqence of actions, returns and pnl
        result = env.ss.result()
        n_long_options, n_short_options, n_pips = get_best_outcomes(result)

        # Store episode results
        agent_pnls.append(result.pnl.sum())
        market_pnls.append(1e4 * (result.iloc[-1].ask - result.iloc[0].bid))
        alpha_pnls.append(agent_pnls[-1] - market_pnls[-1])
        steps.append(result.shape[0])
        potential_pips.append(n_pips)
        long_options.append(n_long_options)
        short_options.append(n_short_options)
        n_buys.append(int(result.position.where(result.position == 1).sum()))
        n_sells.append(int(-result.position.where(result.position == -1).sum()))
        regr_pnls.append(result.regr_pnl.sum())
        mlp_pnls.append(result.mlp_pnl.sum())
        rf_pnls.append(result.rf_pnl.sum())

        if episode % 10 == 0:
            agent_pnl = np.mean(agent_pnls[-10:]), np.mean(agent_pnls[-100:])
            regr_pnl = np.mean(regr_pnls[-100:])
            rf_pnl = np.mean(rf_pnls[-100:])
            mlp_pnl = np.mean(mlp_pnls[-100:])
            mean_pips = np.mean(potential_pips[-10:]), np.mean(potential_pips[-100:])
            market_pnl = np.mean(market_pnls[-10:]), np.mean(market_pnls[-100:])
            n_steps = int(np.mean(steps[-10:]))
            n_buy = int(np.mean(n_buys[-10:]))
            n_sell = int(np.mean(n_sells[-10:]))
            num_sessions_positive_alpha = np.sum([s > 0 for s in alpha_pnls[-100:]]) / \
                min(len(alpha_pnls), 100)
            logger.info(
                f"{episode:>4d} | {strftime('%H:%M:%S', gmtime(time() - t0))} | "
                f"PnL Agent: {agent_pnl[1]:>6.2f}/{mean_pips[1]:>6.2f} "
                f"({agent_pnl[0]:>6.2f}/{mean_pips[0]:>6.2f}) | "
                f"PnL Regr & MLP & Rf: {regr_pnl:6.2f} & {mlp_pnl:6.2f} & {rf_pnl:6.2f}"
                f" | Market (Buy&Hold): {market_pnl[1]:>6.2f} ({market_pnl[0]:>6.2f})"
                f" | Wins: {num_sessions_positive_alpha:>5.1%}"
                f" | Short & Long Positions: -{n_sell}/+{n_buy}"
                f" | Optimal Short & Long: -{np.mean(short_options[-100:])}/+"
                f"{np.mean(long_options[-100:])} | "
                f"Steps: {n_steps} | eps: {agent.epsilon:>6.3f}"
            )

        # Cache results
        if episode % 100 == 0:
            agent.online_network.save(
                Path(ROOT_DIR) / "models" / "agent" / mode_name \
                    / f"q_network_{mode_name}_{episode}.h5"
            )
            results = pd.DataFrame({
                'Episode': list(range(1, episode+1)),
                'Agent': agent_pnls,
                'Market': market_pnls,
                'Steps': steps,
                'Longs': n_buys,
                'Shorts': n_sells,
                'Opt. Longs': long_options,
                'Opt. Shorts': short_options,
                'Potential Pips': potential_pips,
                'Regr(PnL)': regr_pnls,
                'MLP(PnL)': mlp_pnls,
                'RF(PnL)': rf_pnls,
                'Difference': alpha_pnls
            }).set_index('Episode')
            results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
            results.info()
            results.to_csv(
                Path(ROOT_DIR) / "models" / "agent" / mode_name \
                    / f"results_{mode_name}_{episode}.csv", 
                index=False
            )
            agent_results.plot_results_agent(
                results, Path(ROOT_DIR) / "models" / "agent" / mode_name, mode_name
            )
    env.close()

    # Save results
    agent.online_network.save(
        Path(ROOT_DIR) / "models" / "agent" / mode_name / f"q_network_{mode_name}.h5"
    )
    results = pd.DataFrame({
        'Episode': list(range(1, episode+1)),
        'Agent': agent_pnls,
        'Market': market_pnls,
        'Steps': steps,
        'Longs': n_buys,
        'Shorts': n_sells,
        'Opt. Longs': long_options,
        'Opt. Shorts': short_options,
        'Potential Pips': potential_pips,
        'Regr(PnL)': regr_pnls,
        'MLP(PnL)': mlp_pnls,
        'RF(PnL)': rf_pnls,
        'Difference': alpha_pnls
    }).set_index('Episode')
    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
    results.info()
    results.to_csv(
        Path(ROOT_DIR) / "models" / "agent" / mode_name / f"results_{mode_name}.csv", 
        index=False)

    agent_results.plot_results_agent(
        results, Path(ROOT_DIR) / "models" / "agent" / mode_name, mode_name)


@click.command()
@click.argument('scaling_difficulty', type=click.FloatRange(min=0, max=1))
@click.argument('gpu', type=click.IntRange(min=1, max=8))
def main(scaling_difficulty: float, gpu: int):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        filename=Path(ROOT_DIR) / "models" / "agent" / get_mode(scaling_difficulty) \
            / f"{get_mode(scaling_difficulty)}.log",
        filemode='a+'
    )
    with tf.device(f'/gpu:{gpu-1}'):
        rl_agent_5(scaling_difficulty)


if __name__ == '__main__':
    rl_agent_5(1)
   