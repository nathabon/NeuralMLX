from neural.agent import Agent
from trackmania.clientGame import *
import time

game = TMNF()
game.connect_socket()

first_state = 0

ticks_per_second = 0
now = time.time()



while True:
    message_type = game.recv(4)[0]
    if message_type == SocketMessageType.SC_RUN_STEP_SYNC:
        state = game.recv_state()

        race_time = state.player_info.race_time # type: ignore
        if race_time == 0:
            first_state = state
            game.set_input_state(up=True)

        # if race_time == 3000:
        #     game.set_input_state(steer=-65536)

        # if race_time > 0 and race_time % 5000 == 0 and first_state:
        #     game.rewind_to_state(first_state)
        #     game.set_input_state(up=True, steer=65536)

        game.send_signal(SocketMessageType.SC_RUN_STEP_SYNC)

        if time.time() - now > 1:
            print(f'Effective speed: {ticks_per_second / 100}x')
            now = time.time()
            ticks_per_second = 0

        ticks_per_second += 1

game.close_socket()