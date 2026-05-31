"""
Pong DQN — NeuralCPy + MLX, sans librairie d'apprentissage.

Architecture : CNN standard DQN (Mnih et al., 2015)
  (B, 84, 84, 4) → Conv(8×8, s4) → Conv(4×4, s2) → Conv(3×3, s1)
                 → Flatten → Dense(512) → Dense(num_actions)

Installation :
    pip install "gymnasium[atari]" ale-py pillow
    pip install autorom[accept-rom-license]   # accepte les ROMs Atari
"""

import time
import argparse
import numpy as np
import mlx.core as mx
import gymnasium as gym
from PIL import Image
from collections import deque

from neuralNetwork2 import NeuralNetwork
from neuralLayers import Layer
from other import ReLU, fx
from agent import Agent
import ale_py


# ─── Hyperparamètres ─────────────────────────────────────────────────────────

FRAME_H       = 84
FRAME_W       = 84
STACK_N       = 4           # frames empilées formant l'état

BATCH_SIZE    = 64
LR            = 0.00025
GAMMA         = 0.99

EPS_START     = 1.0
EPS_DECAY     = 0.99     # décroissance par step d'apprentissage
EPS_MIN       = 0.05

SYNC_RATE     = 10_000      # steps entre sync online → target
MIN_REPLAY    = 10_000      # steps avant le 1er apprentissage
REPLAY_CAP    = 50_000

LEARN_EVERY   = 4           # apprendre tous les N env-steps
MAX_STEPS     = 5_000_000
SAVE_EVERY    = 100         # sauvegarder tous les N épisodes


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    RGB (210, 160, 3) → float32 (84, 84), niveaux de gris, normalisé.
    On coupe le bandeau de score (lignes 0-34 et 194-210).
    """
    img = Image.fromarray(frame).convert("L")       # niveaux de gris
    img = img.crop((0, 34, 160, 205))               # recadrage carré 160×160
    img = img.resize((FRAME_W, FRAME_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class FrameStack:
    """
    Maintient un buffer glissant de N frames grises.
    Sortie : np.ndarray (84, 84, N) — empilé sur la dim des canaux.
    """

    def __init__(self, n: int = STACK_N):
        self.n   = n
        self._buf: deque[np.ndarray] = deque(maxlen=n)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Initialise le buffer en répétant la première frame N fois."""
        f = preprocess_frame(frame)
        self._buf.clear()
        for _ in range(self.n):
            self._buf.append(f)
        return self._state()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Ajoute une frame et renvoie le nouvel état."""
        self._buf.append(preprocess_frame(frame))
        return self._state()

    def _state(self) -> np.ndarray:
        return np.stack(list(self._buf), axis=-1)   # (84, 84, 4)


# ─── Architecture ─────────────────────────────────────────────────────────────

def build_dqn(num_actions: int) -> NeuralNetwork:
    """
    DQN Atari standard.

    Calcul des dimensions de sortie des Conv :
        (84, 84) ──Conv(8×8, s4)──► (20, 20)
                 ──Conv(4×4, s2)──► (9,  9)
                 ──Conv(3×3, s1)──► (7,  7)
        Flatten ► 7 × 7 × 64 = 3 136
    """
    FLAT_DIM = 7 * 7 * 64  # 3 136

    layers = [
        Layer.Conv2d(STACK_N, 32, 8, 8, ReLU, stride=4),
        Layer.Conv2d(32,      64, 4, 4, ReLU, stride=2),
        Layer.Conv2d(64,      64, 3, 3, ReLU, stride=1),
        Layer.Flatten(),
        Layer.Linear(FLAT_DIM, 512,         ReLU),
        Layer.Linear(512,      num_actions,  fx),
    ]
    return NeuralNetwork(layers)


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train(resume_path: str | None = None):
    env = gym.make("ALE/Pong-v5", render_mode=None)
    num_actions = env.action_space.n    # 6 pour Pong
    print(f"Environnement  : ALE/Pong-v5  —  {num_actions} actions")

    # Réseau
    if resume_path:
        print(f"Reprise depuis : {resume_path}")
        network = NeuralNetwork.fromFileH5(resume_path)
    else:
        network = build_dqn(num_actions)

    print(f"Paramètres     : {network.getNpParameters():,}")
    print(f"Shape de sortie: {network.get_output_shape((1, FRAME_H, FRAME_W, STACK_N))}")

    agent = Agent(
        input_dim         = (FRAME_H, FRAME_W, STACK_N),
        num_actions       = num_actions,
        network           = network,
        learning_rate     = LR,
        gamma             = GAMMA,
        epsilon           = EPS_START,
        epsilon_decay     = EPS_DECAY,
        sync_network_rate = SYNC_RATE,
        batch_size        = BATCH_SIZE,
        min_replay_size   = MIN_REPLAY,
    )
    # On remplace le ReplayBuffer par la capacité voulue
    from agent import ReplayBuffer
    agent.replay_buffer = ReplayBuffer(capacity=REPLAY_CAP, state_shape=(FRAME_H, FRAME_W, STACK_N))


    stacker      = FrameStack(STACK_N)
    total_steps  = 0
    episode      = 0
    reward_hist  = deque(maxlen=50)     # pour la moyenne glissante

    print("\n─── Début de l'entraînement ──────────────────────────────────────")
    print(f"{'Ep':>6} {'Steps':>9} {'R':>6} {'avg50':>7} "
          f"{'ε':>7} {'buf':>7} {'temps':>6}")
    print("─" * 60)

    try:
        while total_steps < MAX_STEPS:
            obs, _  = env.reset()
            state   = stacker.reset(obs)

            # Pong nécessite un FIRE (action=1) pour lancer la balle
            obs, _, _, _, _ = env.step(1)
            state = stacker.push(obs)

            episode_reward = 0.0
            done           = False
            t0             = time.time()

            while not done:
                # ── Action ────────────────────────────────────────────────
                action = agent.choose_action(state)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state = stacker.push(next_obs)

                # Reward clipping DQN standard : signe du score
                clipped_reward = float(np.clip(reward, -1.0, 1.0))

                agent.store_in_memory(state, action, clipped_reward, next_state, done)
                state           = next_state
                episode_reward += reward        # log du vrai score
                total_steps    += 1

                # ── Apprentissage ─────────────────────────────────────────
                if total_steps % LEARN_EVERY == 0:
                    agent.learn()

            agent.decay_epsilon()

            # ── Fin d'épisode ─────────────────────────────────────────────
            episode += 1
            reward_hist.append(episode_reward)
            avg = np.mean(reward_hist)

            print(
                f"{episode:6d} "
                f"{total_steps:9d} "
                f"{episode_reward:+6.1f} "
                f"{avg:+7.2f} "
                f"{agent.epsilon:7.4f} "
                f"{len(agent.replay_buffer):7d} "
                f"{time.time() - t0:5.1f}s"
            )

            if episode % SAVE_EVERY == 0:
                fname = f"pong_dqn_ep{episode:05d}.h5"
                network.saveH5(fname)
                print(f"  ✓ sauvegardé → {fname}")

    except KeyboardInterrupt:
        print("\nInterruption clavier — sauvegarde en cours…")

    env.close()
    network.saveH5("pong_dqn_final.h5")
    print("Entraînement terminé. Modèle sauvegardé → pong_dqn_final.h5")


# ─── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(model_path: str, n_episodes: int = 10, render: bool = True):
    """
    Charge un modèle sauvegardé et joue N épisodes en mode greedy (ε=0).
    """
    render_mode = "human" if render else None
    env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    num_actions = env.action_space.n

    network = NeuralNetwork.fromFileH5(model_path)
    network.freeze()    # désactive le mode training → pas d'accumulation de grads

    stacker = FrameStack(STACK_N)
    scores  = []

    print(f"\nÉvaluation de {model_path} sur {n_episodes} épisodes…")

    for ep in range(1, n_episodes + 1):
        obs, _  = env.reset()
        state   = stacker.reset(obs)
        obs, _, _, _, _ = env.step(1)       # FIRE
        state   = stacker.push(obs)

        total_reward = 0.0
        done         = False

        while not done:
            # Greedy : argmax sur Q(s, ·)
            obs_mx = mx.array(state[np.newaxis], dtype=mx.float32)  # (1, 84, 84, 4)
            q_vals  = network(obs_mx)                                 # (1, num_actions)
            action  = int(mx.argmax(q_vals).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done          = terminated or truncated
            state         = stacker.push(next_obs)
            total_reward += reward

        scores.append(total_reward)
        print(f"  Épisode {ep:3d} | score : {total_reward:+.1f}")

    print(f"\nScore moyen : {np.mean(scores):.2f}  (min {np.min(scores):.1f}, max {np.max(scores):.1f})")
    env.close()


# ─── Entrée ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Pong — NeuralCPy + MLX")
    sub = parser.add_subparsers(dest="cmd")

    # Entraînement
    p_train = sub.add_parser("train", help="Lancer l'entraînement")
    p_train.add_argument("--resume", type=str, default=None,
                         metavar="FILE.h5", help="Reprendre depuis un checkpoint")

    # Évaluation
    p_eval = sub.add_parser("eval", help="Évaluer un modèle sauvegardé")
    p_eval.add_argument("model", type=str, help="Chemin vers le fichier .h5")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--no-render", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        train(resume_path="pong_dqn_final.h5")
    elif args.cmd == "eval":
        evaluate(args.model, n_episodes=args.episodes, render=not args.no_render)
    else:
        # Par défaut : entraîner
        train("pong_dqn_final.h5")