import socket
import struct
import time
import signal
from tminterface.structs import SimStateData, CheckpointData
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from enum import IntEnum
import numpy as np
import gymnasium as gym


class SocketMessageType(IntEnum):
    SC_RUN_STEP_SYNC  = 0
    C_SET_SPEED       = 1
    C_REWIND_TO_STATE = 2
    C_SET_INPUT_STATE = 3
    C_SHUTDOWN        = 4
    C_SET_TRACK       = 5


class ObservationType(IntEnum):
    POSITION                    = 0
    DISPLAY_SPEED               = 1
    VELOCITY                    = 2
    INPUTS                      = 3

    DISTANCE_TO_CENTERLINE      = 4
    PROGRESS_ON_TRACK           = 5
    DIRECTION_VERSUS_CENTERINE  = 6
    DISTANCE_TO_NEXT_CHECKPOINT = 7


def getObservationTypeBox(obs: ObservationType):
    boxes = [
        ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), # Position
        ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), # Display speed
        ([0], [1000]),                                           # Velocity
        ([0, 0, -65536], [65536, 65536, 65536]),                 # Inputs
        ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), # Distance to centerline
        ([0], [1]),                                              # Progress on track
        ([-1, -1, -1], [1, 1, 1]),                               # Direction versus centerline
        ([0], [np.inf]),                                         # Distance no next chekpoint
    ]

    return boxes[obs]



#MARK: TMNF
class TMNF(gym.Env):
    """
    Main class of the game. 
    """

    host: str
    port: int
    sock: socket.socket

    state: SimStateData

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # signal.signal(signal.SIGINT, self.signal_handler)

        self.first_state = None
        self.state = None # type: ignore



    # -----------------------
    # MARK: Signal and socket
    # -----------------------

    def connect_socket(self, host: str = "127.0.0.1", port: int = 8477):
        """Connect socket to (host, port)"""
        self.sock.connect((host, port))
        print("Connected to socket")

    def close_socket(self):
        """Shutdown and close socket"""
        self.sock.close()

    def sendall(self, data):
        """Send to socket"""
        self.sock.sendall(data)

    def send_data(self, fmt: str | bytes, /, *v):
        """Send data with type"""
        self.sendall(struct.pack(fmt, v[0]))
    
    def send_signal(self, signal: int):
        """Send signal"""
        self.send_data('i', int(signal))

    def signal_handler(self, sig, frame):
        """Send shutdown signal"""
        print('Shutting down...')
        self.send_signal(SocketMessageType.C_SHUTDOWN)
        self.sock.close()

    def set_track(self, track_name: str):
        self.send_signal(SocketMessageType.C_SET_TRACK)
        self.send_data('s', track_name)

    def rewind_to_state(self, state):
        """Sent state to rewind"""
        self.send_signal(SocketMessageType.C_REWIND_TO_STATE)
        self.send_data('i', len(state.data))
        self.sendall(state.data)

    def set_input_state(self, up: int | bool = False, down: int | bool = False, steer: int | float = 0x0):
        """Send car control"""
        if isinstance(steer, float) and int(steer) >= -1 and int(steer) <= 1:
            steer = int(steer * 0x10000)

        if isinstance(up, int):
            up = up >= 19661
        if isinstance(down, int):
            down = down >= 19661
        
        self.send_signal(SocketMessageType.C_SET_INPUT_STATE)
        self.send_data('b', up)
        self.send_data('b', down)
        self.send_data('i', steer)

    def set_sim_speed(self, speed: float = 5.0):
        self.send_signal(SocketMessageType.C_SET_SPEED)
        self.send_data('f', speed)

    def _recv(self, bufsize: int, flags: int = 0):
        """Receive the signal sent by the game"""
        return self.sock.recv(bufsize, flags)
    
    def recv(self, bufsize: int):
        """Receive and unpack the signal sent by the game"""
        return struct.unpack('i', self._recv(bufsize))
    
    def recv_state(self):
        """Receive the current signal"""
        state_length = self.recv(4)[0]
        state = SimStateData(self._recv(state_length))
        state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length) # type: ignore
        state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length) # type: ignore
        self.state = state
        return state
    
    
    #
    # MARK: Properties of the car
    #

    @property
    def player_info(self):
        return self.state.player_info
    
    # @property
    # def in_race(self):
    #     return self

    @property
    def car(self):
        return self.state.scene_mobil
    
    @property
    def position(self):
        return self.state.position

    @property
    def display_speed(self):
        return self.state.display_speed
    
    @property
    def velocity(self):
        return self.state.velocity


    @property
    def is_running(self):
        return True
    
    #
    # MARK: Env
    #
    def get_state(self, max_try: int = 1_000):
        i = 0
        while i < max_try:
            msg_type = self.recv(4)[0]
            if msg_type == SocketMessageType.SC_RUN_STEP_SYNC:
                state = self.recv_state()
                return state
            i += 1

        raise ValueError("Socket seems to be disconedted")


    def _get_obs(self, observations: list[ObservationType] = [ObservationType.POSITION]):
        pass

    def _get_info(self) -> dict[str, Any]: # type: ignore
        pass

    def make_env(self, track_name: str, reward: Callable[[SimStateData, np.ndarray], float], steering_values = [0, 1], observations: list[ObservationType] = [ObservationType.POSITION]):
        self.action_space = gym.spaces.Box(low=np.array([0, 0, -65536]), high=np.array([65536, 65536, 65536]), dtype=np.int32)

        low = []
        high = []
        for obs in observations:
            l, h = getObservationTypeBox(obs)
            low += l
            high += h
        self.observation_space = gym.spaces.Box(low=np.array(low), high=np.array(high))

        self.compute_reward = reward
        self.observations = observations

    def reset_env(self, track_name: str, observations: list[ObservationType] = [ObservationType.POSITION], seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.set_track(track_name)

        obs = self._get_obs(observations)
        info = self._get_info()

        return obs, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Cette action n'est pas possible. Il faut gaz € [0, 65526], brake € [0, 65536] et steer € [-65526, 65526]")
        
        gaz, brake, steer = action
        self.set_input_state(gaz, brake, steer)

        self.state = self.get_state()

        reward = self.compute_reward(self.state, action)
        terminated = not self.player_info.finish_not_passed # type: ignore
        truncated = terminated

        obs = self._get_obs(self.observations)
        info = self._get_info()

        return obs, reward, terminated, truncated, info




def get_dist_to_centerline(P: np.ndarray, points: np.ndarray) -> tuple[float, np.ndarray]:
    """
    P      : position voiture (3,)
    points : centerline (N, 3)
    Retourne (dist_center, closest_point)
    """
    A  = points[:-1]                                       # (N-1, 3)
    B  = points[1:]                                        # (N-1, 3)
    AB = B - A                                             # (N-1, 3)

    t  = np.einsum('ij,ij->i', P - A, AB)                 # (N-1,)
    t /= np.einsum('ij,ij->i', AB, AB) + 1e-8             # (N-1,)
    t  = np.clip(t, 0.0, 1.0)                             # (N-1,)

    T     = A + t[:, None] * AB                           # (N-1, 3)
    dists = np.linalg.norm(T - P, axis=1)                 # (N-1,)
    idx   = np.argmin(dists)

    return float(dists[idx]), T[idx]