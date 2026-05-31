from wrappers_mario import apply_wrappers, CropObservation
from agent import Agent
import neuralNetwork2 as nn
import mlx.core as mx
import gymnasium as gym
import ale_py


def app_wrappers(env):
    env = CropObservation(env, 35, 16, 0, 0)
    # return env
    return apply_wrappers(env)


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

gym.register_envs(ale_py)

env = gym.make('ALE/Pong-v5')#, render_mode="human")
env = app_wrappers(env)

print(f"env shape : {env.observation_space.shape}")
state, _ = env.reset()


net = getNetwork(env.observation_space.shape, 6)
# net = nn.NeuralNetwork.fromFileH5("saves/pong/train.h5")

n_actions = env.action_space.n

agent = Agent(
        input_dim=env.observation_space.shape,
        num_actions=n_actions,
        network=net,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999995,
        sync_network_rate=5_000,
        batch_size=128,
        min_replay_size=10_000,
    )

print("Commence")
total_reward = 0
def main():
    global total_reward
    for i in range(NUM_OF_EPISODES):
        print(f"Épisode {i}: {total_reward}")
        done = False
        state, _ = env.reset()
        frame = 0
        total_reward = 0

        while not done:
            # env.render()
            # print("frame: ", frame, end="\r")
            action = agent.choose_action(state)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_in_memory(state, action, reward, (new_state), done)
            total_reward += reward # type: ignore
            

            agent.env_step_counter += 1
            if agent.env_step_counter % agent.learn_every == 0:
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
    agent.online_network.saveH5("saves/pong/train.h5")
    print("All datas have been saved")




env.close()