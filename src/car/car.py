import pygame
import math
import numpy as np
from neural.agent import *
import neural.neuralNetwork2 as nn

WIDTH = 800
HEIGHT = 800

TRACK_CENTER = (WIDTH // 2, HEIGHT // 2)

OUTER_RADIUS = 300
INNER_RADIUS = 180

CAR_SPEED = 4
TURN_SPEED = 4

SENSOR_ANGLES = [-60, -30, 0, 30, 60]
MAX_SENSOR_DISTANCE = 2000


class Car:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2 - 240

        self.angle = 0

    def update(self, action):

        if action == 0:
            self.angle += TURN_SPEED

        elif action == 2:
            self.angle -= TURN_SPEED
        
        elif action == 3:
            return

        rad = math.radians(self.angle)

        self.x += math.cos(rad) * CAR_SPEED
        self.y -= math.sin(rad) * CAR_SPEED

    def get_sensors(self, track):

        distances = []

        for sensor_angle in SENSOR_ANGLES:

            angle = math.radians(self.angle + sensor_angle)

            distance = MAX_SENSOR_DISTANCE

            for d in range(MAX_SENSOR_DISTANCE):

                x = self.x + math.cos(angle) * d
                y = self.y - math.sin(angle) * d

                if not track.is_on_track(x, y):
                    distance = d
                    break

            distances.append(distance / MAX_SENSOR_DISTANCE)

        return np.array(distances, dtype=np.float32)

    def draw(self, screen, track):

        pygame.draw.circle(
            screen,
            (255, 0, 0),
            (int(self.x), int(self.y)),
            8
        )

        for sensor_angle in SENSOR_ANGLES:

            angle = math.radians(self.angle + sensor_angle)

            distance = MAX_SENSOR_DISTANCE

            for d in range(MAX_SENSOR_DISTANCE):

                x = self.x + math.cos(angle) * d
                y = self.y - math.sin(angle) * d

                if not track.is_on_track(x, y):
                    distance = d
                    break

            end_x = self.x + math.cos(angle) * distance
            end_y = self.y - math.sin(angle) * distance

            pygame.draw.line(
                screen,
                (0, 255, 0),
                (self.x, self.y),
                (end_x, end_y),
                2
            )


class Track:

    def is_on_track(self, x, y):

        dx = x - TRACK_CENTER[0]
        dy = y - TRACK_CENTER[1]

        dist = math.sqrt(dx * dx + dy * dy)

        return INNER_RADIUS <= dist <= OUTER_RADIUS

    def draw(self, screen):

        screen.fill((20, 20, 20))

        pygame.draw.circle(
            screen,
            (120, 120, 120),
            TRACK_CENTER,
            OUTER_RADIUS
        )

        pygame.draw.circle(
            screen,
            (20, 20, 20),
            TRACK_CENTER,
            INNER_RADIUS
        )


class RacingEnv:

    def __init__(self):

        self.track = Track()

        self.reset()

    def reset(self):

        self.car = Car()

        return self.car.get_sensors(self.track)

    def step(self, action):

        self.car.update(action)

        alive = self.track.is_on_track(
            self.car.x,
            self.car.y
        )

        reward = 1.0

        if not alive:
            reward = -100.0

        done = not alive

        state = self.car.get_sensors(self.track)

        return state, reward, done

    def render(self, screen):

        self.track.draw(screen)
        self.car.draw(screen, self.track)


# Agent

def get_network(n_actions):
    return nn.NeuralNetwork([
        nn.Layer.Linear(5, 256, nn.ReLU),
        nn.Layer.Linear(256, 256, nn.ReLU),
        nn.Layer.Linear(256, n_actions, nn.fx)
    ])

network = get_network(4)
# network = nn.NeuralNetwork.fromFileH5("car/train.h5")

agent = Agent(
    input_shape=(5,),
    num_actions=4,
    network=network,
    learning_rate=0.0005,
    gamma=0.99,
    epsilon=1,
    epsilon_decay=0.99,
    sync_network_rate=1_000,
    batch_size=64,
    min_replay_size=5000,
    state_preprocess=lambda e: e
)

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

env = RacingEnv()

done = False

for i in range(20_000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render(screen)

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)
        total_reward += reward

        agent.store_in_memory(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.learn()

        state = next_state

        if done:
            state = env.reset()
        clock.tick(120)

        pygame.display.flip()
    agent.decay_epsilon()
    
    print(f"episode: {i}, reward: {total_reward}, eps: {agent.epsilon}")

pygame.quit()