import gymnasium as gym
import numpy as np
import mlx.core as mx
import neuralNetwork2 as nn
from agent import Agent



def image_preprocess(x):
    x = mx.array(x, dtype=mx.float32)

    if len(x.shape) == 3:
        x = x.reshape(1, *x.shape)

    return x / 255.0


def get_network(shape, n_actions):
    H, W, C = shape

    conv_layers = nn.NeuralNetwork([
        nn.Layer.Conv2d(C, 16, 8, 8, nn.ReLU, stride=4),
        nn.Layer.Conv2d(16, 32, 4, 4, nn.ReLU, stride=2),
        nn.Layer.Conv2d(32, 32, 3, 3, nn.ReLU, stride=1),
    ])

    dummy = mx.zeros((1, H, W, C))
    dummy_out = conv_layers(dummy)
    flat_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

    return nn.NeuralNetwork(conv_layers.layers + [
        nn.Layer.Flatten(),
        nn.Layer.Linear(flat_dim, 256, nn.ReLU),
        nn.Layer.Linear(256, n_actions, nn.fx)
    ])


env = gym.make("CarRacing-v3", continuous=False)#, render_mode="human")

print("observation:", env.observation_space)
print("action:", env.action_space)



network = get_network(env.observation_space.shape, 5)
# network = nn.NeuralNetwork.fromFileH5("saves/car/train.h5")

agent = Agent(
    input_dim=env.observation_space.shape,
    num_actions=5,
    network=network,
    learning_rate=0.0005,
    gamma=0.99,
    epsilon=1,
    epsilon_decay=0.99,
    sync_network_rate=1_000,
    batch_size=64,
    min_replay_size=5000,
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
            # env.render()
            env_action = agent.choose_action(state)
            # env_action = DISCRETE_ACTIONS[action_idx]

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
    agent.online_network.saveH5("saves/car/train2.h5")
    print("All datas have been saved")




env.close()