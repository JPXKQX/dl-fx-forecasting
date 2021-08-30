import warnings
import logging
import os
import click
import pandas as pd
import numpy as np
import tensorflow as tf

from time import time, gmtime, strftime
from typing import Tuple, List
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
# tf.config.set_visible_devices(gpus[1:], 'GPU')
print(f"There are a total of {len(gpus)} graphics cards.")
# for gpu in gpus[1:]: tf.config.experimental.set_memory_growth(gpu, True)

# Logging module
logger = logging.getLogger("RL Agent")

################################################################
################### Attributes  ################################
################################################################


NUM_EPISODES = 2000
EPISODE_LENGTH = 1000
NUM_EPISODES_LOGGING = 10
NUM_EPISODES_CACHING = 100


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
    total_pips = sum(buys[long_options]) + sum(sells[short_options])
    return len(long_options), len(short_options), 1e4 * total_pips


def get_models_results(
        result: pd.DataFrame,
        models: List[str] = ['regr', 'mlp', 'rf']
) -> List[float]:
    """ Computes model statistics for an episode

    Args:
        result (pd.DataFrame): episode result of the model. For each model in models
        arguments, it must include at least two columns: {model}_pnl, {model}_positions
        models (List[str]): list of models to compute its statistics.

    Returns:
        List[float]: Total PnL (in pips), number (and PnL) of longs taken, number (and
        PnL) of successful longs taken, number (and PnL) of unsuccessful longs taken,
        number (and PnL) of shorts taken, number (and PnL) of successful shorts taken,
        number (and PnL) of unsuccessful shorts taken,
    """
    results = []
    for model in models:
        pnl = result[f"{model}_pnl"].sum()
        longs = result[f"{model}_pnl"].where(
            result[f"{model}_positions"] == 1
        ).dropna().index
        longs_pnl = result.iloc[longs+1, :][f"{model}_pnl"]
        successful_longs = longs_pnl.where(longs_pnl > 0).dropna()
        unsuccessful_longs = longs_pnl.where(longs_pnl < 0).dropna()

        shorts = result[f"{model}_pnl"].where(
            result[f"{model}_positions"] == -1
        ).dropna().index
        shorts_pnl = result.iloc[shorts+1, :][f"{model}_pnl"]
        successful_shorts = shorts_pnl.where(shorts_pnl > 0).dropna()
        unsuccessful_shorts = shorts_pnl.where(shorts_pnl < 0).dropna()
        results.extend([
            pnl, len(longs), longs_pnl.sum(),
            int(successful_longs.count()), successful_longs.sum(),
            int(unsuccessful_longs.count()), unsuccessful_longs.sum(),
            len(shorts), shorts_pnl.sum(),
            int(successful_shorts.count()), successful_shorts.sum(),
            int(unsuccessful_shorts.count()), unsuccessful_shorts.sum()
        ])
    return results


def rl_agent_5(target_pair: str, scaling_difficulty: float = 0.0):
    env = TradingEnv(
        target=target_pair,
        scaling_difficulty=scaling_difficulty, 
        trading_sessions=5*EPISODE_LENGTH
    )
    env.seed(42)

    mode_name = get_mode(scaling_difficulty)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

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
    rl_agent_results = pd.DataFrame(
        columns=['Agent', 'Market', 'Steps', 'Longs', 'Shorts', 'Optimal Longs',
                 'Optimal Shorts', 'Optimal Pips', 'Difference']
    )
    ind_models_results = pd.DataFrame(
        columns=[
            'Regr(PnL)', 'Regr Longs(Number)', 'Regr Longs(PnL)',
            'Regr Success Longs(Number)', 'Regr Success Longs(Pips)',
            'Regr Fail Longs(Number)', 'Regr Fails Longs(Pips)',
            'Regr Shorts(Number)', 'Regr Shorts(Pips)',
            'Regr Success Shorts(Number)', 'Regr Success Shorts(Pips)',
            'Regr Fail Shorts(Number)', 'Regr Fails Shorts(Pips)',
            'MLP(PnL)', 'MLP Longs(Number)', 'MLP Longs(Pips)',
            'MLP Success Longs(Number)', 'MLP Success Longs(Pips)',
            'MLP Fail Longs(Number)', 'MLP Fails Longs(Pips)',
            'MLP Shorts(Number)', 'MLP Shorts(Pips)',
            'MLP Success Shorts(Number)', 'MLP Success Shorts(Pips)',
            'MLP Fail Shorts(Number)', 'MLP Fails Shorts(Pips)',
            'RF(PnL)', 'RF Longs(Number)', 'RF Longs(Number)',
            'RF Success Longs(Number)', 'RF Success Longs(Pips)',
            'RF Fail Longs(Number)', 'RF Fails Longs(Pips)',
            'RF Shorts(Number)', 'RF Shorts(Pips)',
            'RF Success Shorts(Number)', 'RF Success Shorts(Pips)',
            'RF Fail Shorts(Number)', 'RF Fails Shorts(Pips)'
        ]
    )

    t0 = time()
    for episode in range(1, NUM_EPISODES + 1):
        this_state = env.reset()
        for episode_step in range(EPISODE_LENGTH):
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

        # get DataFrame of a sequence of actions, returns and pnl
        result = env.ss.result()
        n_long_options, n_short_options, n_pips = get_best_outcomes(result)

        # Store episode AGENT results
        agent_pnl = result.pnl.sum()
        market_pnl = 1e4 * (result.iloc[-1].ask - result.iloc[0].bid)
        alpha_pnl = agent_pnl - market_pnl
        steps = result.shape[0]
        n_buys = result.position.where(result.position == 1).sum()
        n_sells = result.position.where(result.position == -1).sum()
        rl_agent_results.loc[episode] = [
            agent_pnl, market_pnl, steps, n_buys, n_sells,
            n_long_options, n_short_options, n_pips, alpha_pnl
        ]

        # Store individual model results
        ind_models_results.loc[episode] = get_models_results(result)

        if episode % NUM_EPISODES_LOGGING == 0:
            models_pnl = ind_models_results.iloc[-100:][
                ['Regr(PnL)', 'MLP(PnL)', 'RF(PnL)']
            ].mean().values
            mean10 = rl_agent_results.iloc[-10:][
                ['Agent', 'Market', 'Optimal Pips', 'Steps', 'Optimal Longs',
                 'Optimal Shorts', 'Longs', 'Shorts']
            ].mean().values
            mean100 = rl_agent_results.iloc[-100:][
                ['Agent', 'Market', 'Optimal Pips']
            ].mean().values
            num_sessions_positive_alpha = (
                    rl_agent_results.iloc[-100:]['Difference'] > 0
            ).mean()
            logger.info(
                f"{episode:>4d} | {strftime('%H:%M:%S', gmtime(time() - t0))} | "
                f"PnL Agent: {mean100[0]:>6.2f}/{mean100[2]:>6.2f} "
                f"({mean10[0]:>6.2f}/{mean10[2]:>6.2f}) | "
                f"PnL Regr & MLP & Rf: {models_pnl[0]:6.2f} & {models_pnl[1]:6.2f} & "
                f"{models_pnl[2]:6.2f}"
                f" | Market (Buy&Hold): {mean100[1]:>6.2f} ({mean10[1]:>6.2f})"
                f" | Wins: {num_sessions_positive_alpha:>5.1%}"
                f" | Short & Long Positions: -{mean10[-1]}/+{mean10[-2]}"
                f" | Optimal Short & Long: -{mean10[5]}/+{mean10[4]}"
                f" | Steps: {mean10[3]} | eps: {agent.epsilon:>6.3f}"
            )

        # Cache results
        if episode % NUM_EPISODES_CACHING == 0:
            agent.online_network.save(
                Path(ROOT_DIR) / "models" / "agent" / mode_name
                / f"q_network_{mode_name}_{episode}.h5"
            )
            rl_agent_results.info()
            rl_agent_results.to_csv(
                Path(ROOT_DIR) / "models" / "agent" / mode_name
                / f"agent_results_{mode_name}_{episode}.csv",
                index=False
            )
            ind_models_results.to_csv(
                Path(ROOT_DIR) / "models" / "agent" / mode_name /
                f"models_results_{mode_name}_{episode}.csv",
                index=False
            )
            agent_results.plot_results_agent(
                rl_agent_results, Path(ROOT_DIR) / "models/agent" / mode_name, mode_name
            )
    env.close()

    # Save results
    agent.online_network.save(
        Path(ROOT_DIR) / "models" / "agent" / mode_name / f"q_network_{mode_name}.h5"
    )
    rl_agent_results.info()
    rl_agent_results.to_csv(
        Path(ROOT_DIR) / "models/agent" / mode_name / f"agent_results_{mode_name}.csv",
        index=False)
    ind_models_results.to_csv(
        Path(ROOT_DIR) / "models/agent" / mode_name / f"models_results_{mode_name}.csv",
        index=False)

    agent_results.plot_results_agent(
        rl_agent_results, Path(ROOT_DIR) / "models" / "agent" / mode_name, mode_name)


@click.command()
@click.argument('target', type=click.STRING)
@click.argument('scaling_difficulty', type=click.FloatRange(min=0, max=1))
@click.argument('gpu', type=click.IntRange(min=1, max=8))
def main(target: str, scaling_difficulty: float, gpu: int):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        filename=Path(ROOT_DIR) / "models" / "agent" / get_mode(scaling_difficulty)
                / f"{get_mode(scaling_difficulty)}.log",
        filemode='a+'
    )
    with tf.device(f'/gpu:{gpu - 1}'):
        rl_agent_5(target, scaling_difficulty)


if __name__ == '__main__':
    rl_agent_5('EURUSD', 1)
