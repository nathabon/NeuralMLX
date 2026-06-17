from src.sample_gymnasium import GymnasiumTraining
from trackmania.clientGame import *
from .sample_client import *
from neural.agent import *
from neural import neuralNetwork2 as nn
import numpy as np



save_name = "saves/trackmania/tracks/Straight1/centerline.csv"
center = np.loadtxt(save_name)
print(center.shape)

observations: list[ObservationType] = [ObservationType.DISPLAY_SPEED, ObservationType.DISTANCE_TO_CENTERLINE, ObservationType.DIR_TO_CENTERLINE]

def getReward(state: SimStateData, actions: np.ndarray):
    dist = get_dist_to_centerline(np.array(state.position), center)[0]

    speed_reward = state.display_speed / 500
    center_reward = np.exp(-dist / 3)

    return speed_reward + center_reward

game = TMNF()
game.connect_socket()
game.set_sim_speed(5)

BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.99
EPS_START = 1.0
EPS_DECAY = 0.995
MIN_REPLAY = 1_000
SYNC_RATE = 5_000
SAVE_EVERY = 10_000



def get_network(shape, n):
    print(shape)
    return nn.NeuralNetwork([
        nn.Layer.Linear(shape, 64, nn.ReLU),
        nn.Layer.Linear(64, 128, nn.ReLU),
        nn.Layer.Linear(128, n, nn.sigmoid)
    ])

def make_env():
    return GymnasiumTraining(env=game)


def train(resume_path: str | None = None):
    trainer = make_env()
    trainer.train(
        get_network=get_network,
        resume_path=resume_path,
        save_path="saves/trackmania/Corner1/follow_centerline",
        nb_episodes=50_000,
        learn_every=1,
        save_every=SAVE_EVERY,
        learning_rate=LR,
        gamma=GAMMA,
        epsilon=EPS_START,
        epsilon_decay=EPS_DECAY,
        sync_network_rate=SYNC_RATE,
        batch_size=BATCH_SIZE,
        min_replay_size=MIN_REPLAY,
    )


def evaluate(model_path: str, n_episodes: int = 10, render: bool = True):
    trainer = make_env()
    trainer.evaluate(model_path, n_episodes=n_episodes, render=render)


if __name__ == "__main__":
    train()