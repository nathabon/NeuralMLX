from neural import mx
import numpy as np
import neural.neuralNetwork2 as nn
import h5py
import time

def to_numpy(a, dtype):
    if isinstance(a, np.ndarray):
        return a
    return np.array(a, dtype=dtype)

def get_tot(shape: tuple):
    if len(shape) == 1:
        return shape[0]
    return shape[0] * get_tot(shape[1:])

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple, state_dtype: type = np.float16, action_dtype: type = np.uint8):
        self.capacity = capacity
        self.size     = 0
        self.ptr      = 0

        self.state_dtype = state_dtype
        self.action_dtype = action_dtype


        # Pré-allocation : un array numpy par champ
        self.states      = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.next_states = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.actions     = np.zeros((capacity,),              dtype=action_dtype)
        self.rewards     = np.zeros((capacity,),              dtype=np.float16)
        self.dones       = np.zeros((capacity,),              dtype=np.bool)

        size = self.states.nbytes + self.next_states.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        print(size / 1024**3)


    def add(self, state, action, reward, next_state, done):
        self.states     [self.ptr] = to_numpy(state, self.state_dtype)
        self.next_states[self.ptr] = to_numpy(next_state, self.state_dtype)
        self.actions    [self.ptr] = to_numpy(action, self.action_dtype)
        self.rewards    [self.ptr] = float(reward)
        self.dones      [self.ptr] = bool(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "state":      mx.array(self.states     [indices], dtype=mx.float16),
            "next_state": mx.array(self.next_states[indices], dtype=mx.float16),
            "action":     mx.array(self.actions    [indices], dtype=mx.uint8),
            "reward":     mx.array(self.rewards    [indices], dtype=mx.float16),
            "done":       mx.array(self.dones      [indices], dtype=mx.bool_),
        }

    def __len__(self):
        return self.size
    
class Agent:
    def __init__(self, 
                 input_shape, num_actions: int, network: nn.NeuralNetwork, learning_rate: float = 0.001, gamma: float = 0.9, epsilon: float = 1.0, epsilon_decay: float = 0.9991, sync_network_rate: int = 10_000, batch_size: int = 32, min_replay_size: int = 3000, buffer_capacity: int = 50_000, state_preprocess = None, state_buffer_shape: tuple | None = None, state_dtype: type = np.float16, action_dtype: type = np.uint8):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learn_every = 4
        self.env_step_counter = 0
        self.learn_step_counter = 0
        self.min_replay_size = min_replay_size

        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = epsilon_decay
        self.eps_min = 0.05
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        self.state_preprocess = state_preprocess or (lambda x: x)
        self.state_buffer_shape = self.input_shape if state_buffer_shape is None else state_buffer_shape

        # Networks
        self.online_network = network
        self.target_network = network.copy()
        self.target_network.eval_mlx()
        self.target_network.freeze()
        self.sync_networks(force=True)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_shape=self.state_buffer_shape, state_dtype=state_dtype, action_dtype=action_dtype)

    # @classmethod 
    # def fromH5(cls, filename)
        
    # def saveH5(self, filename)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        obs = mx.array(observation, dtype=mx.float16)
        obs = self.state_preprocess(obs)
        output = self.online_network(obs)
        return int(mx.argmax(output).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        #     np.asarray(state,      dtype=np.float16),
        #     int(action),
        #     float(reward),
        #     np.asarray(next_state, dtype=np.float16),
        #     bool(done)
        # )

    def sync_networks(self, force=False):
        if force or (self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0):
            for layer_on, layer_tg in zip(
                self.online_network.layers,
                self.target_network.layers
            ):
                layer_tg.copy_from(layer_on)

            self.target_network.eval_mlx()

    def get_prediction(self, next_states):
        # Désactiver le stockage last_X/last_Z pour ce forward
        for layer in self.online_network.layers:
            layer.training = False

        next_q_online = self.online_network(next_states)
        next_actions  = mx.argmax(next_q_online, axis=1)

        for layer in self.online_network.layers:
            layer.training = True

        next_q_target = self.target_network(next_states)
        return next_q_target[mx.arange(self.batch_size), next_actions]
    
    def learn(self, optimizer: str = "adam"):
        if len(self.replay_buffer) < self.min_replay_size:
            return

        # t0 = time.perf_counter()
        # print(f"input: {self.input_dim}")
        self.sync_networks()

        samples     = self.replay_buffer.sample(self.batch_size)
        states      = self.state_preprocess(samples["state"])
        actions     = samples["action"]
        actions     = actions.astype(mx.int32)
        rewards     = samples["reward"]
        next_states = self.state_preprocess(samples["next_state"])
        dones       = samples["done"]
        # t_sample = time.perf_counter() - t0
        

        # t0 = time.perf_counter()
        # ── max_next_q EN PREMIER (n'a pas besoin de last_X correct) ──────────
        max_next_q  = self.get_prediction(next_states)    # training=False → ne touche pas last_X

        # ── predicted_q EN DERNIER → last_X/last_Z correctement stockés ───────
        predicted_q = self.online_network(states)[mx.arange(self.batch_size), actions]
        target_q = rewards + self.gamma * max_next_q * (1 - dones)
        grad     = predicted_q - target_q
 
        one_hot = mx.eye(self.num_actions)[actions]
        delta_full = one_hot * grad[:, None]

        debug = self.learn_step_counter % 500 == 0 and False

        if debug:
            loss_before = mx.mean((predicted_q - target_q) ** 2)
            print("loss before:", float(loss_before.item()))

        self.online_network.getDelta(delta_full, None) # type: ignore
        self.online_network.updateWeights(self.alpha, optimizer=optimizer)
        # t_graph = time.perf_counter() - t0

        # t0 = time.perf_counter()
        mx.eval()
        # mx.synchronize()
        # t_eval = time.perf_counter() - t0

        # print(f"sample: {t_sample} - raph: {t_graph} - eval: {t_eval}")

        if debug:
            q_after_all = self.online_network(states)
            q_after = q_after_all[mx.arange(self.batch_size), actions]
            loss_after = mx.mean((q_after - target_q) ** 2)
            print("loss after :", float(loss_after.item()))

        self.learn_step_counter += 1
        # self.decay_epsilon()

        if self.learn_step_counter % 1000 == 0 and False:
            loss = mx.mean((predicted_q - target_q) ** 2)
            print(
                "learn:", self.learn_step_counter,
                "epsilon:", float(self.epsilon),
                "loss:", float(loss.item()),
                "q_mean:", float(mx.mean(predicted_q).item()),
                "target_mean:", float(mx.mean(target_q).item()),
                "grad_mean:", float(mx.mean(mx.abs(grad)).item())
            )