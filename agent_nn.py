import mlx.core as mx
import numpy as np
import neuralNetwork2 as nn

from collections import deque

def reshape_shape(shape):
    # (4, 84, 84, 3) → (84, 84, 12)
    frames, h, w, c = shape
    return (h, w, frames * c)

def reshape_im(im: np.ndarray) -> np.ndarray:
    # Grayscale : (4, H, W) → (H, W, 4)
    return im.transpose(1, 2, 0)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)  # supprime automatiquement les plus vieux
    
    def add(self, experience: dict):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> dict:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            "state":      mx.array(np.stack([e["state"]      for e in batch]), dtype=mx.float32),
            "next_state": mx.array(np.stack([e["next_state"] for e in batch]), dtype=mx.float32),
            "action":     mx.array(np.array([e["action"] for e in batch])),
            "reward":     mx.array(np.array([e["reward"] for e in batch], dtype=np.float32)),
            "done":       mx.array(np.array([e["done"]   for e in batch], dtype=np.float32)),
        }
    
    def __len__(self):
        return len(self.buffer)

class AgentNN:
    def __init__(self, input_shape, n_actions, freeze=False):
        # Détecte grayscale vs RGB
        if len(input_shape) == 3:
            # Grayscale : (H, W, 4) → C_in=4, H=84, W=84
            H, W, frames = input_shape
            C_in = frames
        else:
            # RGB : (4, H, W, 3) → C_in=12
            frames, H, W, C = input_shape
            C_in = frames * C

        # conv_layers = nn.NeuralNetwork([
        #     nn.Layer.Conv2d(C_in, 16, 8, 8, nn.ReLU),
        #     nn.Layer.MaxPooling((2, 2)),

        #     nn.Layer.Conv2d(16, 32, 4, 4, nn.ReLU),
        #     nn.Layer.MaxPooling((2, 2)),

        #     nn.Layer.Conv2d(32, 32, 3, 3, nn.ReLU),
        # ])

        conv_layers = nn.NeuralNetwork([
            nn.Layer.Conv2d(C_in, 32, 8, 8, nn.ReLU, stride=4),
            nn.Layer.Conv2d(32, 64, 4, 4, nn.ReLU, stride=2),
            nn.Layer.Conv2d(64, 64, 3, 3, nn.ReLU, stride=1),
        ])

        # Dummy en (N, H, W, C_in) — format MLX conv2d
        dummy = mx.zeros((1, H, W, C_in))
        dummy_out = conv_layers(dummy)  # (1, oH, oW, 64)
        flat_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

        self.network = nn.NeuralNetwork(conv_layers.layers + [
            nn.Layer.Flatten(),
            nn.Layer.Linear(flat_dim, 128, nn.ReLU),
            nn.Layer.Linear(128, n_actions, nn.softmax)
        ])

        if freeze:
            self._freeze()
    
    def _freeze(self):
        for layer in self.network.layers:
            layer.training = False
    
    def __call__(self, X: mx.array) -> mx.array:
        return self.network(X)


class Agent:
    def __init__(self, input_dims, num_actions):
        self.num_actions = num_actions
        self.learn_every = 4
        self.env_step_counter = 0
        self.learn_step_counter = 0
        self.min_replay_size = 2000

        # Hyperparameters
        self.alpha = 0.00025
        self.gamma = 0.9
        self.epsilon = 1.0
        self.eps_decay = 0.99999975
        self.eps_min = 0.1
        self.batch_size = 32
        self.sync_network_rate = 10_000

        # Networks
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)
        self.sync_networks(force=True)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        obs = mx.array(observation, dtype=mx.float32).reshape(1, 84, 84, 4)
        output = self.online_network(obs)
        return int(mx.argmax(output).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add({
            "state":      np.array(state,      dtype=np.float32),  # (84, 84, 4)
            "action":     int(action),
            "reward":     float(reward),
            "next_state": np.array(next_state, dtype=np.float32),
            "done":       bool(done)
        })

    def sync_networks(self, force=False):
        if force or (self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0):
            for layer_on, layer_tg in zip(
                self.online_network.network.layers,
                self.target_network.network.layers
            ):
                layer_tg.copy_from(layer_on)

            self.target_network.network.eval_mlx()
    
    @nn.no_grad
    def get_prediction(self, next_states):
        next_q = self.target_network(next_states)           # (B, num_actions)
        return mx.max(next_q, axis=1)                 # (B,)

    def learn(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        import time
        def t(label, start):
            mx.eval()  # force tout ce qui est en attente
            print(f"  {label}: {(time.time() - start)*1000:.1f}ms")
            return time.time()

        # print("Learn")
        t0 = time.time()

        self.sync_networks()
        # t0 = t("sync_networks", t0)

        samples     = self.replay_buffer.sample(self.batch_size)
        states      = samples["state"]
        actions     = samples["action"]
        rewards     = samples["reward"]
        next_states = samples["next_state"]
        dones       = samples["done"]
        # t0 = t("sample replay buffer", t0)

        predicted_q = self.online_network(states)
        predicted_q = predicted_q[mx.arange(self.batch_size), actions]
        mx.eval(predicted_q)
        # t0 = t("forward online network", t0)

        max_next_q = self.get_prediction(next_states)
        mx.eval(max_next_q)
        # t0 = t("forward target network", t0)

        target_q = rewards + self.gamma * max_next_q * (1 - dones)
        grad     = 2 * (predicted_q - target_q) / self.batch_size
        # t0 = t("bellman + grad", t0)

        actions = actions.astype(mx.int32)
        one_hot = mx.eye(self.num_actions)[actions]
        delta_full = one_hot * grad[:, None]
        # t0 = t("build delta", t0)

        self.online_network.network.getDelta(delta_full, None)
        # t0 = t("getDelta (backprop)", t0)

        self.online_network.network.updateWeights(self.alpha)
        mx.eval()
        # t0 = t("eval et update (backprop)", t0)

        self.learn_step_counter += 1
        self.decay_epsilon()