from wrappers_mario import apply_wrappers
from agent import Agent
import neuralNetwork2 as nn
import mlx.core as mx
import gymnasium as gym
import numpy as np

def getNetwork(shape, n_actions):
    W, H, C_in = shape
    conv_layers = nn.NeuralNetwork([
        nn.Layer.Conv2d(C_in, 32, 8, 8, nn.ReLU, stride=4),
        nn.Layer.Conv2d(32, 64, 4, 4, nn.ReLU, stride=2),
        nn.Layer.Conv2d(64, 64, 3, 3, nn.ReLU, stride=1),
    ])

    dummy = mx.zeros((1, H, W, C_in))
    dummy_out = conv_layers(dummy)  # (1, oH, oW, 64)
    flat_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

    return nn.NeuralNetwork(conv_layers.layers + [
        nn.Layer.Flatten(),
        nn.Layer.Linear(flat_dim, 128, nn.ReLU),
        nn.Layer.Linear(128, n_actions, nn.fx)
    ])



NUM_OF_EPISODES = 50000

DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),   # ne rien faire
    np.array([-1.0, 0.2, 0.0], dtype=np.float32),  # gauche + gaz léger
    np.array([1.0, 0.2, 0.0], dtype=np.float32),   # droite + gaz léger
    np.array([0.0, 0.5, 0.0], dtype=np.float32),   # accélérer
    np.array([0.0, 0.0, 0.8], dtype=np.float32),   # freiner
]

env = gym.make('CarRacing-v3', render_mode="human")
env = apply_wrappers(env)

print(f"env shape : {env.observation_space.shape}")

net = getNetwork(env.observation_space.shape, len(DISCRETE_ACTIONS))
# net = nn.NeuralNetwork.fromFileH5("saves/mario/train3.h5")


agent = Agent(input_dim=env.observation_space.shape, num_actions=len(DISCRETE_ACTIONS), network=net, learning_rate=0.005)


print("Commence")

def main():
    for i in range(NUM_OF_EPISODES) :
        print(f"Épisode {i}")
        done = False
        state, _ = env.reset()
        frame = 0
        total_reward = 0

        while not done:
            env.render()
            # print("frame: ", frame, end="\r")
            action = agent.choose_action(state)

            new_state, reward, done, truncated, info = env.step(action)
            agent.store_in_memory(state, action, reward, (new_state), done)
            total_reward += reward # type: ignore
            

            # agent.env_step_counter += 1
            # if agent.env_step_counter % agent.learn_every == 0:
            agent.learn()

            state = new_state
            frame += 1
        # print()

save = True
print("Train")
try:
    main()
except KeyboardInterrupt:
    print()
    save = True if input("Voulez vous sauvegarder les paramètres (O, n) ? ") == "o" else False


if save:
    print("Saving data...", end="\r")
    agent.online_network.saveH5("saves/mario/train4.h5")
    print("All datas have been saved")




env.close()