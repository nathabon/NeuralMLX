import numpy as np 
from gymnasium import Env, Wrapper, ObservationWrapper
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation, ResizeObservation
from gymnasium import spaces

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        for _ in range(self.skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward  # type: ignore
            if terminated or truncated:
                break
        return state, total_reward, terminated, truncated, info # type: ignore

class TransposeObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        shape = env.observation_space.shape

        if len(shape) == 3 and shape[0] == 4:
            frames, H, W = shape
            new_shape = (H, W, frames)
            self.need_transpose = True
        elif len(shape) == 3 and shape[-1] == 4:
            new_shape = shape
            self.need_transpose = False
        else:
            raise ValueError(f"Shape inattendue après FrameStackObservation: {shape}")

        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        observation = np.array(observation)

        if self.need_transpose:
            return observation.transpose(1, 2, 0)

        return observation
    

class CropObservation(ObservationWrapper):
    """
    Crop une image RGB au format (H, W, C).
    Exemple Mario NES brut : souvent environ (240, 256, 3).
    """
    def __init__(self, env, top=0, bottom=0, left=0, right=0):
        super().__init__(env)

        h, w, c = env.observation_space.shape
        print(h, c, w)

        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

        new_h = h - top - bottom
        new_w = w - left - right

        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(new_h, new_w, c),
            dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        h, w, c = observation.shape

        y1 = self.top
        y2 = h - self.bottom if self.bottom > 0 else h
        x1 = self.left
        x2 = w - self.right if self.right > 0 else w

        return observation[y1:y2, x1:x2, :]
    

class ReduceHighsObservation(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=self.observation_space.shape,
            dtype=np.float16
        )
    
    def observation(self, observation):
        return (observation.astype(np.float32) / 255.0)
    
    
class BinaryPongObservation(ObservationWrapper):
    def __init__(self, env, threshold=0.5):
        super().__init__(env)
        self.threshold = threshold
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return (observation > self.threshold).astype(np.float32)

def apply_wrappers(env):
    env = SkipFrame(env, 4)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    env = TransposeObservation(env)
    env = ReduceHighsObservation(env)
    
    return env