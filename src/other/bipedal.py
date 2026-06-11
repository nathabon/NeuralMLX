import gymnasium as gym
import numpy as np
from neural import mx
import neural.neuralNetwork2 as nn
from neural.agent import Agent
from itertools import product
import numpy as np


def image_preprocess(x: mx.array):
    return x

def get_network(shape, n_actions):
    n = shape[0]
    print(n)
    return nn.NeuralNetwork([
        nn.Layer.Linear(int(n), 256, nn.ReLU),
        nn.Layer.Linear(256, 256, nn.ReLU),
        nn.Layer.Linear(256, n_actions, nn.fx)
    ])


env = gym.make("LunarLander-v3", render_mode="human")



print("observation:", env.observation_space)



# network = get_network(env.observation_space.shape, 4)
network = nn.NeuralNetwork.fromFileH5("saves/acrobot/train.h5")

agent = Agent(
    input_shape=env.observation_space.shape,
    num_actions=4,
    network=network,
    learning_rate=0.0005,
    gamma=0.99,
    epsilon=0.05,
    epsilon_decay=0.99,
    sync_network_rate=1_000,
    batch_size=64,
    min_replay_size=5_000,
    buffer_capacity=100_000,
    state_shape=env.observation_space.shape,
    state_preprocess=image_preprocess
)

def main():
    scores = []

    for episode in range(2000):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frame = 0

        while not done:
            env.render()
            env_action = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            # print(reward)


            agent.store_in_memory(state, env_action, reward, next_state, done)

            agent.env_step_counter += 1
            if agent.env_step_counter % agent.learn_every == 0:
                agent.learn()

            state = next_state
            total_reward += reward # type: ignore
            frame += 1

        scores.append(total_reward)
        avg20 = np.mean(scores[-20:])

        print(
            f"Episode {episode:4d} | "
            f"score: {total_reward:8.2f} | "
            f"avg20: {avg20:8.2f} | "
            f"epsilon: {agent.epsilon:.3f} | "
            f"replay: {len(agent.replay_buffer)}"
        )

        agent.decay_epsilon()


save = True
print("Train")
try:
    main()
except KeyboardInterrupt:
    print()
    save = True if input("Voulez vous sauvegarder les paramètres (O, n) ? ") == "o" else False


if save:
    print("Saving data...", end="\r")
    agent.online_network.saveH5("saves/acrobot/train.h5")
    print("All datas have been saved")




env.close()