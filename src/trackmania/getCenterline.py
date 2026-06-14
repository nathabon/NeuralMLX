from trackmania.clientGame import *
import time
import numpy as np

game = TMNF()
game.connect_socket()

first_state = 0

ticks_per_second = 0
now = time.time()

done = False
save_name = "saves/trackmania/tracks/Straight1/centerline.csv"
track_name = "A011-Race.Challenge.Gbx"
positions = []
# game.set_track(track_name)
game.set_sim_speed(0.5)

# center = np.loadtxt(save_name)
# print(center.shape)

try:
    while not done:
        msg = game.recv(4)
        message_type = msg[0]
        print(msg)
        if message_type == SocketMessageType.SC_RUN_STEP_SYNC:
            state = game.recv_state()

            race_time = state.player_info.race_time # type: ignore
            if race_time == 0:
                first_state = state
                print("------------------------------- RESET -------------------------------")
                positions = []

            
            p = np.array(state.position, dtype=np.float16)
            # print(get_dist_to_centerline(p, center))
            positions.append(p)

            game.send_signal(SocketMessageType.SC_RUN_STEP_SYNC)

            if time.time() - now > 1:
                print(f'Effective speed: {ticks_per_second / 100}x')
                now = time.time()
                ticks_per_second = 0

            ticks_per_second += 1
            done = state.player_info.race_finished
except KeyboardInterrupt:
    print("interrupt")


pos = np.array(positions, dtype=np.float16)
print(pos.shape)   # (n, 3)


save = input(f"Sauvegarder le fichier sous {save_name} ? (O, n)").lower() == "o"
if save:
    np.savetxt(save_name, pos)
game.close_socket()