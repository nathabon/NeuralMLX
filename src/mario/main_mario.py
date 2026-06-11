from mario.wrappers_mario import apply_wrappers
from neural.agent import Agent
import neural.neuralNetwork2 as nn
from neural import mx
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import time

def getNetwork(shape, n_actions):
    W, H, C_in = shape
    conv_layers = nn.NeuralNetwork([
        nn.Layer.Conv2d(C_in, 32, 8, 8, nn.ReLU, stride=4),
        nn.Layer.Conv2d(32, 64, 4, 4, nn.ReLU, stride=2),
        nn.Layer.Conv2d(64, 128, 3, 3, nn.ReLU, stride=1),
    ])

    dummy = mx.zeros((1, H, W, C_in))
    dummy_out = conv_layers(dummy)  # (1, oH, oW, 128)
    flat_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

    return nn.NeuralNetwork(conv_layers.layers + [
        nn.Layer.Flatten(),
        nn.Layer.Linear(flat_dim, 512, nn.ReLU),
        nn.Layer.Linear(512, n_actions, nn.fx)
    ])



NUM_OF_EPISODES = 50000

env = gym.make('SuperMarioBros-1-1-v0')#, render_mode="human")
env = JoypadSpace(env, RIGHT_ONLY) # type: ignore
env = apply_wrappers(env)

print(f"env shape : {env.observation_space.shape}")

net = getNetwork(env.observation_space.shape, len(RIGHT_ONLY))
# net = nn.NeuralNetwork.fromFileH5("saves/mario/train5-1.h5")


agent = Agent(input_shape=env.observation_space.shape, num_actions=len(RIGHT_ONLY), network=net, learning_rate=0.0001, batch_size=128, gamma=0.99)


print("Commence")
max_ = 0
def main():
    global max_
    for i in range(NUM_OF_EPISODES):
        print(f"Épisode {i}: {max_}")
        done = False
        state, _ = env.reset()
        frame = 0
        total_reward = 0
        max_ = 0

        while not done:
            # env.render()
            # print("frame: ", frame, end="\r")
            action = agent.choose_action(state)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_in_memory(state, action, reward, new_state, done)
            total_reward += reward # type: ignore
            

            agent.env_step_counter += 1
            if agent.env_step_counter % agent.learn_every == 0:
                agent.learn()
            max_ = info["x_pos"]

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
    agent.online_network.saveH5("saves/mario/train5-1.h5")
    print("All datas have been saved")




env.close()