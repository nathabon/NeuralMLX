import os
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx

import neuralNetwork2 as nn
from pong import app_wrappers


gym.register_envs(ale_py)

SAVE_PATH = "saves/pong/pong_dqn.h5"
OUT_DIR = "debug_pong_view"

ACTION_MAP = [2, 3]  # 2 = monter, 3 = descendre selon tes tests


# def app_wrappers(env):
#     env = CropObservation(env, 35, 16, 0, 0)
#     return apply_wrappers(env)


def make_env(render=False):
    render_mode = "human" if render else "rgb_array"

    env = gym.make(
        "ALE/Pong-v5",
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0
    )

    return app_wrappers(env)


def pong_preprocess(x):
    x = mx.array(x, dtype=mx.float32)

    if len(x.shape) == 3:
        return x.reshape(1, *x.shape)

    return x


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_frame_stack(state, prefix):
    """
    Sauvegarde les 4 frames vues par le réseau.
    state: (84, 84, 4)
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for i in range(4):
        axes[i].imshow(state[:, :, i], cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"frame {i}")
        axes[i].axis("off")

    plt.tight_layout()
    path = f"{OUT_DIR}/{prefix}_frames.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)


def save_motion_maps(state, prefix):
    """
    Sauvegarde les différences entre frames.
    Si la balle bouge, elle doit apparaître ici.
    """
    diffs = [
        np.abs(state[:, :, 1] - state[:, :, 0]),
        np.abs(state[:, :, 2] - state[:, :, 1]),
        np.abs(state[:, :, 3] - state[:, :, 2]),
        np.abs(state[:, :, 3] - state[:, :, 0]),
    ]

    titles = [
        "diff 1-0",
        "diff 2-1",
        "diff 3-2",
        "diff 3-0",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, diff, title in zip(axes, diffs, titles):
        ax.imshow(diff, cmap="gray")
        ax.set_title(f"{title}\nsum={diff.sum():.2f}")
        ax.axis("off")

    plt.tight_layout()
    path = f"{OUT_DIR}/{prefix}_motion.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    print(
        prefix,
        "diff sums:",
        [float(d.sum()) for d in diffs],
        "diff max:",
        [float(d.max()) for d in diffs]
    )


def detect_right_paddle(state):
    img = state[:, :, -1]
    mask = img > 0.35

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    right_mask = xs > 66

    if not np.any(right_mask):
        return None

    return float(np.mean(ys[right_mask]))


def detect_ball_by_motion(state):
    f0 = state[:, :, 0]
    f3 = state[:, :, 3]

    motion = np.abs(f3 - f0)

    # On retire les zones paddles et la ligne centrale approximative.
    motion[:, :18] = 0
    motion[:, 66:] = 0
    motion[:, 39:45] = 0

    ys, xs = np.where(motion > 0.12)

    if len(xs) == 0:
        return None, None

    return float(np.mean(xs)), float(np.mean(ys))


def save_detection_overlay(state, prefix):
    img = state[:, :, -1]
    motion = np.abs(state[:, :, 3] - state[:, :, 0])

    motion[:, :18] = 0
    motion[:, 66:] = 0
    motion[:, 39:45] = 0

    paddle_y = detect_right_paddle(state)
    ball_x, ball_y = detect_ball_by_motion(state)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Dernière frame + détection")
    axes[0].axis("off")

    if paddle_y is not None:
        axes[0].axhline(paddle_y, color="yellow")

    if ball_x is not None and ball_y is not None:
        axes[0].scatter([ball_x], [ball_y], s=40, color="red")

    axes[1].imshow(motion, cmap="gray")
    axes[1].set_title("Mouvement utilisé pour balle")
    axes[1].axis("off")

    if ball_x is not None and ball_y is not None:
        axes[1].scatter([ball_x], [ball_y], s=40, color="red")

    plt.tight_layout()
    path = f"{OUT_DIR}/{prefix}_detection.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    print(
        prefix,
        "paddle_y:", paddle_y,
        "ball_x:", ball_x,
        "ball_y:", ball_y
    )


def load_network():
    if not os.path.exists(SAVE_PATH):
        print("Aucun réseau sauvegardé trouvé :", SAVE_PATH)
        return None

    print("Chargement réseau :", SAVE_PATH)
    return nn.NeuralNetwork.fromFileH5(SAVE_PATH)


def q_values(network, state):
    x = pong_preprocess(state)
    q = network(x)
    mx.eval(q)
    return np.array(q).reshape(-1)


def print_q_values(network, state, prefix):
    q = q_values(network, state)
    print(f"{prefix} Q-values:", q)

    if len(q) == 2:
        print(f"{prefix} action choisie:", int(np.argmax(q)), "env action:", ACTION_MAP[int(np.argmax(q))])


def save_activation_maps(network, state, prefix):
    """
    Sauvegarde les activations moyennes des couches convolutionnelles.
    Ça permet de voir si les convs réagissent à quelque chose.
    """
    x = pong_preprocess(state)

    conv_index = 0

    for i, layer in enumerate(network.layers):
        x = layer(x)
        mx.eval(x)

        arr = np.array(x)

        # activation conv : (1, H, W, C)
        if len(arr.shape) == 4:
            # moyenne sur les canaux
            mean_map = arr[0].mean(axis=2)

            plt.figure(figsize=(4, 4))
            plt.imshow(mean_map, cmap="gray")
            plt.title(f"Activation conv {conv_index} layer {i}\nshape={arr.shape}")
            plt.axis("off")

            path = f"{OUT_DIR}/{prefix}_activation_conv{conv_index}.png"
            plt.savefig(path, dpi=160)
            plt.close()
            print("saved:", path)

            print(
                f"{prefix} conv {conv_index}",
                "shape:", arr.shape,
                "min:", float(arr.min()),
                "max:", float(arr.max()),
                "mean:", float(arr.mean()),
                "nonzero ratio:", float(np.mean(arr != 0))
            )

            conv_index += 1


def save_occlusion_map(network, state, prefix, action_index=None, patch=8):
    """
    Test très utile :
    on masque une zone de l'image, et on regarde si la Q-value change.
    Si la carte est noire partout, le réseau n'utilise pas vraiment l'image.
    Si certaines zones changent beaucoup la Q-value, le réseau y est sensible.
    """
    base_q = q_values(network, state)

    if action_index is None:
        action_index = int(np.argmax(base_q))

    base_value = base_q[action_index]

    H, W, C = state.shape
    heat = np.zeros((H // patch, W // patch), dtype=np.float32)

    for iy, y in enumerate(range(0, H - patch + 1, patch)):
        for ix, x0 in enumerate(range(0, W - patch + 1, patch)):
            masked = np.array(state, copy=True)

            # On masque la zone dans les 4 frames.
            masked[y:y + patch, x0:x0 + patch, :] = 0.0

            q = q_values(network, masked)
            heat[iy, ix] = abs(base_value - q[action_index])

    plt.figure(figsize=(5, 5))
    plt.imshow(heat, cmap="hot")
    plt.colorbar()
    plt.title(
        f"Occlusion map | action={action_index} env={ACTION_MAP[action_index]}\n"
        f"base Q={base_value:.3f}"
    )
    plt.axis("off")

    path = f"{OUT_DIR}/{prefix}_occlusion_action{action_index}.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)

    print(
        prefix,
        "occlusion action:", action_index,
        "base q:", float(base_value),
        "heat max:", float(heat.max()),
        "heat mean:", float(heat.mean())
    )


def collect_states(env, mode="alternating", steps=200):
    """
    Récupère quelques états représentatifs.
    """
    state, _ = env.reset()

    collected = []

    for t in range(steps):
        if mode == "noop":
            action = 0
        elif mode == "up":
            action = 2
        elif mode == "down":
            action = 3
        elif mode == "alternating":
            action = 2 if (t // 20) % 2 == 0 else 3
        else:
            action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if t in [10, 30, 60, 100, 150, 199]:
            collected.append((t, np.array(state, copy=True), reward))

        if done:
            break

    return collected


def main():
    ensure_dir()

    env = make_env(render=False)

    print("Action meanings:", env.unwrapped.get_action_meanings())
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    state, _ = env.reset()
    print("Initial state:")
    print("  shape:", state.shape)
    print("  dtype:", state.dtype)
    print("  min/max:", state.min(), state.max())
    print("  channel sums:", [float(state[:, :, i].sum()) for i in range(state.shape[2])])

    network = load_network()

    modes = ["noop", "up", "down", "alternating"]

    for mode in modes:
        print()
        print("=== MODE", mode, "===")

        states = collect_states(env, mode=mode, steps=220)

        for t, s, reward in states:
            prefix = f"{mode}_t{t}"

            print()
            print("---", prefix, "reward:", reward, "---")
            print("shape:", s.shape, "dtype:", s.dtype, "min/max:", s.min(), s.max())
            print("channel sums:", [float(s[:, :, i].sum()) for i in range(s.shape[2])])

            save_frame_stack(s, prefix)
            save_motion_maps(s, prefix)
            save_detection_overlay(s, prefix)

            if network is not None:
                print_q_values(network, s, prefix)
                save_activation_maps(network, s, prefix)
                save_occlusion_map(network, s, prefix, action_index=None, patch=8)

    env.close()
    print()
    print("Fichiers générés dans :", OUT_DIR)


if __name__ == "__main__":
    main()