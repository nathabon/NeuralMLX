from neural.agent import Agent
import neural.neuralNetwork2 as nn
import gymnasium as gym
import numpy as np
import mlx.core as mx


NUM_EPISODES = 5000


def get_network():
    return nn.NeuralNetwork([
        nn.Layer.Linear(4, 64, nn.ReLU),
        nn.Layer.Linear(64, 64, nn.ReLU),
        nn.Layer.Linear(64, 2, nn.fx)
    ])


def preprocess_state(state):
    """
    CartPole donne un état de shape (4,).
    Le réseau attend un batch : (B, 4).
    """
    state = mx.array(state, dtype=mx.float32)

    if len(state.shape) == 1:
        return state.reshape(1, 4)

    return state.reshape(state.shape[0], 4)

def cartpole_preprocess(x):
    x = mx.array(x, dtype=mx.float32)
    if len(x.shape) == 1:
        return x.reshape(1, -1)
    return x.reshape(x.shape[0], -1)


def evaluate(agent, env, episodes=5):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    scores = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        scores.append(total_reward)

    agent.epsilon = old_epsilon

    return np.mean(scores)


def main():
    env = gym.make("CartPole-v1", render_mode="human")

    # network = get_network()
    network = nn.NeuralNetwork.fromFileH5("saves/cartpole/cartpole_dqn.h5")

    agent = Agent(
        input_dim=env.observation_space.shape,
        num_actions=env.action_space.n,
        network=network,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.05,
        epsilon_decay=0.9998,
        sync_network_rate=500,
        batch_size=64,
        min_replay_size=1000,
        state_preprocess=cartpole_preprocess
    )

    action_counts = np.zeros(env.action_space.n, dtype=int)

    # Si ton Agent n’a pas encore state_preprocess intégré,
    # on va forcer les shapes directement dans la boucle.
    scores = []
    best_avg = 0

    print("Début entraînement CartPole")

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frame = 0

        while not done:
            env.render()
            # Choix action : il faut que choose_action reshape en (1, 4)
            # Si ton choose_action ne le fait pas, voir correction plus bas.
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminated:
                reward = -10.0

            action_counts[action] += 1

            agent.store_in_memory(state, action, reward, next_state, done)

            agent.env_step_counter += 1
            if agent.env_step_counter % agent.learn_every == 0:
                agent.learn()

            state = next_state
            total_reward += reward
            frame += 1

        scores.append(total_reward)

        avg_20 = np.mean(scores[-20:])
        avg_100 = np.mean(scores[-100:])

        if avg_20 > best_avg:
            best_avg = avg_20

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d} | "
                f"score: {total_reward:6.1f} | "
                f"avg20: {avg_20:6.1f} | "
                f"avg100: {avg_100:6.1f} | "
                f"best20: {best_avg:6.1f} | "
                f"epsilon: {agent.epsilon:.3f} | "
                f"replay: {len(agent.replay_buffer)} | "
                f"actions: {action_counts}"
            )
            action_counts[:] = 0

        if episode % 100 == 0 and episode > 0:
            eval_score = evaluate(agent, env, episodes=5)
            print(f"Évaluation sans exploration : {eval_score:.1f}")

        if avg_100 >= 475:
            print(f"CartPole résolu à l’épisode {episode}, avg100={avg_100:.1f}")
            break

    env.close()

    print("Sauvegarde...")
    agent.online_network.saveH5("saves/cartpole/cartpole_dqn.h5")
    print("Terminé")


if __name__ == "__main__":
    main()