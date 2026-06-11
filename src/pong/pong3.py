import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from collections import deque
from PIL import Image

from sample_gymnasium import GymnasiumTraining
from neural.neuralNetwork2 import *
from mario.wrappers_mario import apply_wrappers

FRAME_H = 84
FRAME_W = 84
STACK_N = 4

BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.99
EPS_START = 1.0
EPS_DECAY = 0.99
MIN_REPLAY = 10_000
SYNC_RATE = 10_000
SAVE_EVERY = 100


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame).convert("L")
    img = img.crop((0, 34, 160, 205))
    img = img.resize((FRAME_W, FRAME_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float16) / 255.0




def build_dqn(num_actions: int) -> NeuralNetwork:
    FLAT_DIM = 7 * 7 * 64
    layers = [
        Layer.Conv2d(STACK_N, 32, 8, 8, ReLU, stride=4),
        Layer.Conv2d(32, 64, 4, 4, ReLU, stride=2),
        Layer.Conv2d(64, 64, 3, 3, ReLU, stride=1),
        Layer.Flatten(),
        Layer.Linear(FLAT_DIM, 512, ReLU),
        Layer.Linear(512, 512, ReLU),
        Layer.Linear(512, num_actions, fx),
    ]
    return NeuralNetwork(layers)


def make_trainer(resume_path: str | None = None) -> GymnasiumTraining:
    def env_wrapper(env):
        return apply_wrappers(env)

    return GymnasiumTraining(envName="PongNoFrameskip-v4", envWrapper=env_wrapper, actions=[2, 3])


def train(resume_path: str | None = None):
    trainer = make_trainer(resume_path)
    trainer.train(
        get_network=build_dqn,
        resume_path=resume_path,
        save_path="saves/pong/pong3",
        nb_episodes=5_000,
        learn_every=4,
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
    trainer = make_trainer(None)
    trainer.evaluate(model_path, n_episodes=n_episodes, render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong DQN with GymnasiumTraining")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Lancer l'entraînement")
    p_train.add_argument("--resume", type=str, default=None, metavar="FILE.h5",
                         help="Reprendre depuis un checkpoint")

    p_eval = sub.add_parser("eval", help="Évaluer un modèle sauvegardé")
    p_eval.add_argument("model", type=str, help="Chemin vers le fichier .h5")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--no-render", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        train(resume_path=args.resume)
    elif args.cmd == "eval":
        evaluate(args.model, n_episodes=args.episodes, render=not args.no_render)
    else:
        # train(None)
        evaluate("saves/pong/pong3_final.h5")
