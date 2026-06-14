import argparse
from sample_gymnasium import GymnasiumTraining
from neural import neuralNetwork2 as nn


BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.99
EPS_START = 1.0
EPS_DECAY = 0.995
MIN_REPLAY = 1_000
SYNC_RATE = 5_000
SAVE_EVERY = 10_000


def get_network(n):
    return nn.NeuralNetwork([
        nn.Layer.Linear(4, 64, nn.ReLU),
        nn.Layer.Linear(64, 64, nn.ReLU),
        nn.Layer.Linear(64, n, nn.sigmoid)
    ])


def make_trainer(resume_path: str | None = None) -> GymnasiumTraining:
    return GymnasiumTraining(envName="CartPole-v1")


def train(resume_path: str | None = None):
    trainer = make_trainer(resume_path)
    trainer.train(
        get_network=get_network,
        resume_path=resume_path,
        save_path="saves/cartpole/cartpole2",
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
        train(None)
        # evaluate("saves/cartpole/cartpole_ep04000.h5")
