import time
import argparse
import numpy as np

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete

import ale_py

from collections import deque
from collections.abc import Callable

import neural as mx
from neural.neuralNetwork2 import *
from neural.agent import Agent

def none(param, default):
    return param if param is not None else default 

# MARK: GymnasiumTraining
class GymnasiumTraining:       
    def __init__(self, envName: str | None = None, env: Env | None = None, envWrapper: Callable[[Env], Env] | None = None, actions: list | None = None) -> None:
        if envName is None and env is None:
            raise ValueError("envName ou env doivent être donnés pour créer un environement")
        self.env = none(env, gym.make(envName)) # type: ignore
        if envWrapper is not None:
            self.env = envWrapper(self.env)
        self.envWrapper = envWrapper
        
        print(self.env)

        if actions is not None:
            self.actions = actions
            self.n_actions = len(self.actions)
        else:
            print(self.env.action_space)
            if isinstance(self.action_space, Box):
                raise ValueError("This Agent cannot work on continius action space. Please use a discrete action space")
            if not isinstance(self.env.action_space, Discrete):
                print(self.action_space)
                raise NotImplementedError

            self.n_actions = int(self.env.action_space.n)
            self.actions = [i for i in range(self.n_actions)]

    
    @property
    def observation_space(self):
        return self.env.observation_space.shape
    
    @property
    def action_space(self):
        return self.env.action_space.shape

    #MARK: train
    def train(self, get_network: Callable[[tuple | None, int], NeuralNetwork] | None = None, resume_path: str | None = None, save_path: str | None = None, 
              nb_episodes: int = 5_000, learn_every: int = 4, save_every: int = 1_000,
              learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.99,
              sync_network_rate: int = 10_000, batch_size: int = 64, min_replay_size: int = 5_000):
        
        if resume_path is not None:
            print(f"Reprise depuis : {resume_path}")
            network = NeuralNetwork.fromFileH5(resume_path)
        elif get_network is not None:
            network = get_network(self.env.observation_space.shape, self.n_actions)
        else:
            raise ValueError("Il faut donner un moyen de créer un réseau de neurone")
        
        if save_path is None:
            save_path = f"saves/{self.env.unwrapped.spec.id}" # type: ignore

        print(f"Paramètres     : {network.getNpParameters():,}")
        print(f"Shape d'entrée: {self.observation_space}")
        print(f"Shape de sortie: {network.get_output_shape(self.observation_space)}")

        agent = Agent(
            input_shape       = self.observation_space,
            num_actions       = self.n_actions,
            network           = network,
            learning_rate     = learning_rate,
            gamma             = gamma,
            epsilon           = epsilon,
            epsilon_decay     = epsilon_decay,
            sync_network_rate = sync_network_rate,
            batch_size        = batch_size,
            min_replay_size   = min_replay_size,
        )

        total_steps  = 0
        episode      = 0
        reward_hist_20  = deque(maxlen=20)
        reward_hist_100  = deque(maxlen=100)

        print("─── Début de l'entraînement ───")
        save = True
        try:
            for episode in range(nb_episodes):
                state, _  = self.env.reset()

                episode_reward = 0.0
                done           = False
                t0             = time.time()

                while not done:
                    action_idx = agent.choose_action(state)
                    action = self.actions[action_idx]

                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    agent.store_in_memory(state, action_idx, reward, next_state, done)
                    state           = next_state
                    episode_reward += float(reward)
                    total_steps    += 1

                    if total_steps % learn_every == 0:
                        agent.learn()

                agent.decay_epsilon()

                episode += 1
                reward_hist_20.append(episode_reward)
                reward_hist_100.append(episode_reward)
                avg20 = np.mean(reward_hist_20)
                avg100 = np.mean(reward_hist_100)

                print(
                    f"Épisode : {episode:6d} | "
                    f"Steps : {total_steps:9d} | "
                    f"Reward : {episode_reward:+6.1f} | "
                    f"Avg20 : {avg20:+7.2f} | "
                    f"Avg100 : {avg100:+7.2f} | "
                    f"Epsilon: {agent.epsilon:7.4f} | "
                    f"Buffer : {len(agent.replay_buffer):7d} | "
                    f"Time : {time.time() - t0:5.1f}s"
                )

                if save_every != False and episode % save_every == 0:
                    fname = f"{save_path}_ep{episode:05d}.h5"
                    network.saveH5(fname)
                    print(f"Sauvegardé → {fname}")

        except KeyboardInterrupt:
            print()
            save = True if input("Voulez vous sauvegarder les paramètres (O, n) ? ").lower() == "o" else False

        self.env.close()
        if save:
            network.saveH5(f"{save_path}_final.h5")
            print(f"Entraînement terminé. Modèle sauvegardé → {save_path}_final.h5")


    # MARK: evaluate
    def evaluate(self, model_path: str, n_episodes: int = 10, render: bool = True):
        """
        Charge un modèle sauvegardé et joue N épisodes en mode greedy (ε=0).
        """
        render_mode = "human" if render else None
        # self.env.render_mode = render_mode
        self.env = gym.make(self.env.spec.id, render_mode=render_mode)
        if self.envWrapper is not None:
            self.env = self.envWrapper(self.env)

        network = NeuralNetwork.fromFileH5(model_path)
        network.freeze()
        scores  = []

        print(f"\nÉvaluation de {model_path} sur {n_episodes} épisodes…")

        for ep in range(1, n_episodes + 1):
            state, _  = self.env.reset()

            total_reward = 0.0
            done         = False

            while not done:
                if render:
                    self.env.render()
                obs = mx.array(state[np.newaxis], dtype=mx.float32)
                q_vals  = network(obs)
                action_idx  = int(mx.argmax(q_vals).item())
                action = self.actions[action_idx]

                state, reward, terminated, truncated, _ = self.env.step(action)
                done          = terminated or truncated
                total_reward += reward # type: ignore

            scores.append(total_reward)
            print(f"  Épisode {ep:3d} | score : {total_reward:+.1f}")

        print(f"\nScore moyen : {np.mean(scores):.2f}  (min {np.min(scores):.1f}, max {np.max(scores):.1f})")
        self.env.close()
