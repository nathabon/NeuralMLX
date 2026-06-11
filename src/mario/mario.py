"""
Mario DQN — NeuralCPy + MLX, sans librairie d'apprentissage.

Architecture : CNN standard DQN (Mnih et al., 2015) — identique à Pong
  (B, 84, 84, 4) → Conv(8×8, s4) → Conv(4×4, s2) → Conv(3×3, s1)
                 → Flatten → Dense(512) → Dense(num_actions)

Différences clés vs Pong :
  - Espace d'action réduit : SIMPLE_MOVEMENT (7 actions)
  - Reward = delta de position x (progression dans le niveau)
  - Perte d'une vie = done pour l'entraînement (pas pour l'épisode réel)
  - Preprocessing : crop du HUD NES (32px en bas, 16px en haut)

Installation :
    pip install gym-super-mario-bros
    pip install shimmy   # compatibilité gym → gymnasium
    pip install pillow
"""

import time
import argparse
import numpy as np
from neural import mx
from PIL import Image
from collections import deque

# ── Compatibilité gym-super-mario-bros (utilise l'ancienne API gym) ───────────
import gym_super_mario_bros
import gymnasium as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from neural.neuralNetwork2 import NeuralNetwork
from neural.neuralLayers import Layer
from neural.other import ReLU, fx
from neural.agent import Agent, ReplayBuffer


# ─── Hyperparamètres ─────────────────────────────────────────────────────────

FRAME_H       = 84
FRAME_W       = 84
STACK_N       = 4

BATCH_SIZE    = 64
LR            = 0.00025
GAMMA         = 0.9       # plus court que Pong : récompenses plus fréquentes

EPS_START     = 1.0
EPS_DECAY     = 0.99999   # Mario est plus long à apprendre → decay plus lent
EPS_MIN       = 0.05

SYNC_RATE     = 10_000
MIN_REPLAY    = 10_000
REPLAY_CAP    = 50_000

LEARN_EVERY   = 4
MAX_STEPS     = 50_000_000
SAVE_EVERY    = 50


# ─── Environnement ────────────────────────────────────────────────────────────

def make_env(world: int = 1, stage: int = 1):
    """
    Crée l'environnement Mario avec l'API gym (pas gymnasium).
    gym-super-mario-bros n'a pas encore de wrapper gymnasium natif stable.
    
    SIMPLE_MOVEMENT = 7 actions :
        0: NOOP
        1: right
        2: right + A (saut)
        3: right + B (sprint)
        4: right + A + B (saut + sprint)
        5: A (saut sur place)
        6: left
    """
    env_id = f"SuperMarioBros-1-1-v0'"
    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    RGB NES (240, 256, 3) → float32 (84, 84), niveaux de gris, normalisé.

    Le HUD NES de Mario occupe :
      - 16px en haut  (score, monde, temps)
      - 32px en bas   (vie, pièces)
    On garde la zone de jeu : lignes 16 à 208 → 192×256 → resize 84×84.
    """
    img = Image.fromarray(frame).convert("L")
    img = img.crop((0, 16, 256, 208))                    # zone de jeu
    img = img.resize((FRAME_W, FRAME_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class FrameStack:
    def __init__(self, n: int = STACK_N):
        self.n   = n
        self._buf: deque[np.ndarray] = deque(maxlen=n)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        f = preprocess_frame(frame)
        self._buf.clear()
        for _ in range(self.n):
            self._buf.append(f)
        return self._state()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._buf.append(preprocess_frame(frame))
        return self._state()

    def _state(self) -> np.ndarray:
        return np.stack(list(self._buf), axis=-1)        # (84, 84, 4)


# ─── Reward shaping ───────────────────────────────────────────────────────────

class RewardShaper:
    """
    Transforme le reward brut de Mario en signal d'apprentissage dense.

    Reward brut gym-super-mario-bros :
        +/- delta_x   : progression horizontale (peut être négatif si on recule)
        -50           : mort
        +50           : fin de niveau (flagpole)
        +1            : par tranche de 100 pts de score (pièces, ennemis)

    Problème : le delta_x est déjà un reward dense mais bruité.
    On le normalise pour qu'il reste dans [-1, 1], tout en gardant
    les signaux de mort/victoire forts.
    """

    def __init__(self):
        self.last_x    = 0
        self.last_life = 2

    def reset(self, info: dict):
        self.last_x    = info.get("x_pos",   0)
        self.last_life = info.get("life",     2)

    def shape(self, reward: float, info: dict, done: bool) -> tuple[float, bool]:
        """
        Retourne (reward_shapé, life_done).
        life_done = True si Mario vient de mourir (pour marquer done dans le buffer
        même si l'épisode continue avec une autre vie).
        """
        life_done = False

        current_life = info.get("life", self.last_life)

        # Perte de vie : signal fort négatif + marquer done pour le buffer
        if current_life < self.last_life:
            shaped = -15.0
            life_done = True
        # Fin de niveau : signal fort positif
        elif info.get("flag_get", False):
            shaped = 15.0
        else:
            # Delta de position normalisé dans [-1, 1]
            # Max delta_x par step ≈ 3-4 pixels → normalisation /3
            dx = info.get("x_pos", self.last_x) - self.last_x
            shaped = np.clip(dx / 3.0, -1.0, 1.0)

        self.last_x    = info.get("x_pos",   self.last_x)
        self.last_life = current_life

        return shaped, life_done


# ─── Architecture ─────────────────────────────────────────────────────────────

def build_dqn(num_actions: int) -> NeuralNetwork:
    """
    Identique à Pong — DQN Atari standard.
    (84, 84, 4) → 3 Conv → Flatten(3136) → Dense(512) → Dense(num_actions)
    """
    FLAT_DIM = 7 * 7 * 64

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

def train(resume_path: str | None = None, world: int = 1, stage: int = 1):
    env         = make_env(world, stage)
    num_actions = env.action_space.n
    print(f"Environnement  : SuperMarioBros-{world}-{stage}  —  {num_actions} actions")
    print(f"SIMPLE_MOVEMENT: {SIMPLE_MOVEMENT}")

    if resume_path:
        print(f"Reprise depuis : {resume_path}")
        network = NeuralNetwork.fromFileH5(resume_path)
    else:
        network = build_dqn(num_actions)

    print(f"Paramètres     : {network.getNpParameters():,}")

    agent = Agent(
        input_shape         = (FRAME_H, FRAME_W, STACK_N),
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
    agent.replay_buffer = ReplayBuffer(
        capacity    = REPLAY_CAP,
        state_shape = (FRAME_H, FRAME_W, STACK_N)
    )

    stacker      = FrameStack(STACK_N)
    shaper       = RewardShaper()
    total_steps  = 0
    episode      = 0
    reward_hist  = deque(maxlen=50)
    x_pos_hist   = deque(maxlen=50)   # suivi de la progression dans le niveau

    print("\n─── Début de l'entraînement ──────────────────────────────────────────")
    print(f"{'Ep':>6} {'Steps':>10} {'R':>7} {'avg50':>7} "
          f"{'xPos':>6} {'avgX':>6} {'ε':>7} {'buf':>7} {'t':>5}")
    print("─" * 75)

    try:
        while total_steps < MAX_STEPS:
            obs, _   = env.reset()
            state    = stacker.reset(obs)
            shaper.reset({"x_pos": 0, "life": 2})

            episode_reward = 0.0
            done           = False
            t0             = time.time()
            max_x          = 0

            while not done:
                action = agent.choose_action(state)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                shaped_reward, life_done = shaper.shape(reward, info, done)

                next_state = stacker.push(next_obs)

                # done pour le buffer = vrai done OU mort
                buffer_done = done or life_done

                agent.store_in_memory(state, action, shaped_reward, next_state, buffer_done)

                state           = next_state
                episode_reward += reward          # reward brut pour le log
                total_steps    += 1
                max_x           = max(max_x, info.get("x_pos", 0))

                if total_steps % LEARN_EVERY == 0:
                    agent.learn()

                agent.decay_epsilon()

            episode += 1
            reward_hist.append(episode_reward)
            x_pos_hist.append(max_x)

            print(
                f"{episode:6d} "
                f"{total_steps:10d} "
                f"{episode_reward:+7.1f} "
                f"{np.mean(reward_hist):+7.2f} "
                f"{max_x:6d} "
                f"{np.mean(x_pos_hist):6.0f} "
                f"{agent.epsilon:7.4f} "
                f"{len(agent.replay_buffer):7d} "
                f"{time.time() - t0:4.1f}s"
            )

            if episode % SAVE_EVERY == 0:
                fname = f"mario_{world}-{stage}_ep{episode:05d}.h5"
                network.saveH5(fname)
                print(f"  ✓ sauvegardé → {fname}")

    except KeyboardInterrupt:
        print("\nInterruption clavier — sauvegarde en cours…")

    env.close()
    network.saveH5(f"mario_{world}-{stage}_final.h5")
    print(f"Entraînement terminé. Modèle sauvegardé → mario_{world}-{stage}_final.h5")


# ─── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(model_path: str, n_episodes: int = 5, world: int = 1, stage: int = 1, render: bool = True):
    if render:
        import gym_super_mario_bros
        env = gym_super_mario_bros.make(
            f"SuperMarioBros-{world}-{stage}-v0",
            render_mode="human"
        )
    else:
        env = make_env(world, stage)

    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    network = NeuralNetwork.fromFileH5(model_path)
    network.freeze()

    stacker = FrameStack(STACK_N)
    scores  = []
    x_poss  = []

    print(f"\nÉvaluation de {model_path} — monde {world}-{stage}")

    for ep in range(1, n_episodes + 1):
        obs, _       = env.reset()
        state        = stacker.reset(obs)
        total_reward = 0.0
        done         = False
        max_x        = 0

        while not done:
            if render:
                env.render()
            obs_mx = mx.array(state[np.newaxis], dtype=mx.float32)
            q_vals = network(obs_mx)
            action = int(mx.argmax(q_vals).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state         = stacker.push(next_obs)
            total_reward += reward
            max_x         = max(max_x, info.get("x_pos", 0))

        scores.append(total_reward)
        x_poss.append(max_x)
        flag = "🚩" if max_x >= 3150 else "  "
        print(f"  Épisode {ep:3d} | score : {total_reward:+7.1f} | x_pos : {max_x:4d} {flag}")

    print(f"\nScore moyen  : {np.mean(scores):.1f}")
    print(f"x_pos moyen  : {np.mean(x_poss):.0f} / ~3186 (fin du niveau)")
    print(f"Niveaux finis: {sum(x >= 3150 for x in x_poss)}/{n_episodes}")
    env.close()


# ─── Entrée ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Mario — NeuralCPy + MLX")
    parser.add_argument("--world", type=int, default=1, help="Monde (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Stage (1-4)")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--resume", type=str, default=None, metavar="FILE.h5")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("model", type=str)
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--no-render", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        train(resume_path=args.resume, world=args.world, stage=args.stage)
    elif args.cmd == "eval":
        evaluate(args.model, n_episodes=args.episodes,
                 world=args.world, stage=args.stage, render=not args.no_render)
    else:
        train(world=args.world, stage=args.stage)