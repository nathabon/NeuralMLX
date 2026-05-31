import gymnasium as gym
import numpy as np
import mlx.core as mx
import neuralNetwork2 as nn
from agent import Agent
from gymnasium import Env, Wrapper, ObservationWrapper
from wrappers_mario import apply_wrappers, CropObservation, BinaryPongObservation, TransposeObservation
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import time

from gymnasium import spaces
import ale_py

def app_wrappers(env):
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )

    env = FrameStackObservation(env, stack_size=4)
    env = TransposeObservation(env)

    return env

SAVE_PATH = "saves/pong/pong_dqn.h5"
LOAD_EXISTING = False
NUM_EPISODES = 50_000
gym.register_envs(ale_py)

# MARK: Env
def make_env(render=False):
    render_mode = "human" if render else "rgb_array"

    env = gym.make(
        "ALE/Pong-v5",
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0
    )

    return app_wrappers(env)


def pong_preprocess(x):
    x = mx.array(x, dtype=mx.float32)

    # Image seule : (84, 84, 4) -> (1, 84, 84, 4)
    if len(x.shape) == 3:
        return x.reshape(1, *x.shape)

    # Batch : (B, 84, 84, 4)
    return x

# MARK: Network
def get_network(shape, n_actions):
    H, W, C_in = shape

    conv_layers = nn.NeuralNetwork([
        nn.Layer.Conv2d(C_in, 32, 8, 8, nn.ReLU, stride=4),
        nn.Layer.Conv2d(32, 64, 4, 4, nn.ReLU, stride=2),
        nn.Layer.Conv2d(64, 64, 3, 3, nn.ReLU, stride=1)
    ])

    dummy = mx.zeros((1, H, W, C_in))
    dummy_out = conv_layers(dummy)
    flat_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

    print("dummy_out:", dummy_out.shape)
    print("flat_dim:", flat_dim)

    return nn.NeuralNetwork(conv_layers.layers + [
        nn.Layer.Flatten(),
        nn.Layer.Linear(flat_dim, 512, nn.ReLU),
        nn.Layer.Linear(512, n_actions, nn.fx)
    ])


def evaluate(agent, env, action_map, episodes=3):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    scores = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_idx = agent.choose_action(state)
            env_action = action_map[action_idx]

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        scores.append(total_reward)

    agent.epsilon = old_epsilon
    return float(np.mean(scores))

# MARK: main
def main():
    ACTION_MAP = [2, 3]

    env = make_env(render=False)
    state, _ = env.reset()

    n_actions = len(ACTION_MAP)

    if LOAD_EXISTING:
        print("Chargement du réseau existant...")
        network = nn.NeuralNetwork.fromFileH5(SAVE_PATH)
    else:
        network = get_network(env.observation_space.shape, n_actions)

    agent = Agent(
        input_dim=env.observation_space.shape,
        num_actions=n_actions,
        network=network,
        learning_rate=0.00035,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        sync_network_rate=5_000,
        batch_size=64,
        min_replay_size=5_000,
        state_preprocess=pong_preprocess
    )

    agent.learn_every = 4
    agent.eps_min = 0.1

    scores = []
    best_avg20 = -9999.0
    save = True

    try:
        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            action_counts = np.zeros(n_actions, dtype=np.int32)

            while not done:
                # t0 = time.perf_counter()
                action_idx = agent.choose_action(state)
                action_counts[action_idx] += 1
                env_action = ACTION_MAP[action_idx]
                # t_chose = time.perf_counter() - t0

                # t0 = time.perf_counter()
                next_state, reward, terminated, truncated, info = env.step(env_action)

                done = terminated or truncated
                # t_env = time.perf_counter() - t0

                if reward > 0: # type: ignore
                    shaped_reward = 1.0
                elif reward < 0: # type: ignore
                    shaped_reward = -1.0
                else:
                    shaped_reward = 0.001

                # t0 = time.perf_counter()
                agent.store_in_memory(
                    state,
                    action_idx,
                    shaped_reward,
                    next_state,
                    done
                )
                # t_store = time.perf_counter() - t0

                agent.env_step_counter += 1
                # t0 = time.perf_counter()
                if agent.env_step_counter % agent.learn_every == 0:
                    agent.learn()
                # t_learn = time.perf_counter() - t0

                # if agent.env_step_counter % agent.learn_every == 0 and len(agent.replay_buffer) > agent.min_replay_size:
                    # print(f"chose: {t_chose}, env: {t_env}, store: {t_store}, learn: {t_learn}")

                total_reward += reward # type: ignore
                state = next_state

            scores.append(total_reward)

            avg20 = float(np.mean(scores[-20:]))
            avg100 = float(np.mean(scores[-100:]))

            if avg20 > best_avg20:
                best_avg20 = avg20

            print(
                f"Episode {episode:5d} | "
                f"score: {total_reward:7.2f} | "
                f"avg20: {avg20:7.2f} | "
                f"avg100: {avg100:7.2f} | "
                f"best20: {best_avg20:7.2f} | "
                f"epsilon: {agent.epsilon:.3f} | "
                f"replay: {len(agent.replay_buffer)} | "
                f"actions: {action_counts}"
            )

            if episode % 50 == 0 and episode > 0:
                stats = agent.replay_buffer.reward_stats()
                print("Replay reward stats:", stats)

            if episode % 100 == 0 and episode > 0:
                eval_score = evaluate(agent, env, ACTION_MAP, episodes=3)
                print(f"Évaluation sans exploration : {eval_score:.2f}")

            agent.decay_epsilon()

    except KeyboardInterrupt:
        print()
        answer = input("Voulez-vous sauvegarder les paramètres ? (o/N) : ").strip().lower()
        save = answer == "o"

    finally:
        env.close()

        if save:
            print("Sauvegarde du réseau...")
            agent.online_network.saveH5(SAVE_PATH)
            print(f"Réseau sauvegardé dans : {SAVE_PATH}")
        else:
            print("Sauvegarde annulée.")


if __name__ == "__main__":
    main()