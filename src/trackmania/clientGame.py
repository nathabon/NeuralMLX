import socket
import struct
import time
import signal
from tminterface.structs import SimStateData, CheckpointData, PlayerInfoStruct
from enum import IntEnum
import numpy as np

class GameSignal(IntEnum):
    SC_RUN_STEP_SYNC = 0
    C_SET_SPEED = 1
    C_REWIND_TO_STATE = 2
    C_SET_INPUT_STATE = 3
    C_SHUTDOWN = 4

#MARK: TMNF
class TMNF:
    """
    Main class of the game. 
    """

    host: str
    port: int
    sock: socket.socket

    state: SimStateData

    def __init__(self, host: str = "127.0.0.1", port: int = 8477) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.state = None # type: ignore

    # -----------------------
    # MARK: Signal and socket
    # -----------------------

    def connect_socket(self):
        """Connect socket to (host, port)"""
        self.sock.connect((self.host, self.port))
        print("Connected")

    def sendall(self, data):
        """Send to socket"""
        self.sock.sendall(data)

    def send_data(self, fmt: str | bytes, /, *v):
        """Send data with type"""
        self.sendall(struct.pack(fmt, v))
    
    def send_signal(self, signal: int):
        """Send signal"""
        self.send_data("i", signal)

    def signal_handler(self, sig, frame):
        """Send shutdown signal"""
        print('Shutting down...')
        self.send_signal(GameSignal.C_SHUTDOWN)
        self.sock.close()

    def rewind_to_state(self, state):
        """Sent state to rewind"""
        self.send_signal(GameSignal.C_REWIND_TO_STATE)
        self.send_data('i', len(state.data))
        self.sendall(state.data)

    def set_input_state(self, up: int | bool = -1, down: int | bool = -1, steer: int | float = 0x7FFFFFFF):
        """Send car control"""
        if isinstance(steer, float):
            steer = int(steer * 0x7FFFFFFF)
        
        self.send_signal(GameSignal.C_SET_INPUT_STATE)
        self.send_data('b', up)
        self.send_data('b', down)
        self.send_data('i', steer)

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
    
    @property
    def car(self):
        return self.state.scene_mobil
    
    @property
    def position(self):
        return self.state.position

    @property
    def display_speed(self):
        return self.state.display_speed
    
def dist(p):
    return np.sqrt(np.sum(p**2))



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