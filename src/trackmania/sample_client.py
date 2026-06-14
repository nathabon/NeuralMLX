import socket
import struct
import time
import signal
from tminterface.structs import SimStateData, CheckpointData

HOST = "127.0.0.1"
PORT = 8477

SC_RUN_STEP_SYNC = 0
C_SET_SPEED = 1
C_REWIND_TO_STATE = 2
C_SET_INPUT_STATE = 3
C_SHUTDOWN = 4

sock = None

def signal_handler(sig, frame):
    global sock

    print('Shutting down...')
    sock.sendall(struct.pack('i', C_SHUTDOWN))
    sock.close()


def rewind_to_state(sock, state):
    sock.sendall(struct.pack('i', C_REWIND_TO_STATE))
    sock.sendall(struct.pack('i', len(state.data)))
    sock.sendall(state.data)

def set_input_state(sock, up=-1, down=-1, steer=0x7FFFFFFF):
    sock.sendall(struct.pack('i', C_SET_INPUT_STATE))
    sock.sendall(struct.pack('b', up))
    sock.sendall(struct.pack('b', down))
    sock.sendall(struct.pack('i', steer))

def respond(sock, typ):
    sock.sendall(struct.pack('i', typ))

def main():
    global sock

    first_state = 0

    ticks_per_second = 0
    now = time.time()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    signal.signal(signal.SIGINT, signal_handler)

    sock.connect((HOST, PORT))
    print('Connected')
    while True:
        message_type = struct.unpack('i', sock.recv(4))[0]
        if message_type == SC_RUN_STEP_SYNC:
            state_length = struct.unpack('i', sock.recv(4))[0]
            state = SimStateData(sock.recv(state_length))
            state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
            state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)

            race_time = state.player_info.race_time
            if race_time == 0:
                first_state = state
                set_input_state(sock, up=True)

            if race_time == 3000:
                set_input_state(sock, steer=-65536)

            if race_time > 0 and race_time % 5000 == 0 and first_state:
                rewind_to_state(sock, first_state)
                set_input_state(sock, up=True, steer=65536)

            respond(sock, SC_RUN_STEP_SYNC)

            if time.time() - now > 1:
                print(f'Effective speed: {ticks_per_second / 100}x')
                now = time.time()
                ticks_per_second = 0

            ticks_per_second += 1

if __name__ == "__main__":
    main()