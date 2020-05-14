import os
import argparse
import gym
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np


dir_path = os.path.dirname(os.path.abspath(__file__))
model_paths = ['normal.h5', 'cutter.h5', 'drifter.h5']
model_paths = [os.path.join('models', m) for m in model_paths]
model_paths = [os.path.join(dir_path, m) for m in model_paths]


parser = argparse.ArgumentParser(description='AI racer')
parser.add_argument("-s", "--style", type=int, default=2,
                    help="Driving style: 0 - normal, 1 - cutter, 2 - drifter")
parser.add_argument("-i", "--iterations", type=int, default=3000,
                    help="Frames to proccess, when shut down")
args = parser.parse_args()
driving_style = args.style
iter_count = args.iterations


if driving_style not in range(len(model_paths)):
    raise Exception('Unsupported driving style, read --help.')
model_path = model_paths[driving_style]
model = load_model(model_path)


speed_limit = 0.4 if driving_style == 2 else 0.6
break_power = 0.3 if driving_style == 2 else 0.4


ACTIONS = np.array(
    [
        np.array([0, speed_limit, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, break_power]),
        np.array([-1, 0, 0]),
        np.array([1, 0, 0]),
        np.array([-1, 0, break_power]),
        np.array([1, 0, break_power])
    ], dtype=np.float16
)


def main():
    state_shape = (1, 96, 96, 3)
    env = gym.make('CarRacing-v0')
    state = env.reset()
    states = state.reshape(state_shape)

    # Initial acceleration
    for _ in range(50):
        action = np.array([0, 1, 0])
        next_state, reward, done, _ = env.step(action)
        env.render()
        states = next_state.reshape(state_shape)

    # Model predictions
    for _ in range(iter_count):
        states = tf.cast(states, tf.float16)
        action_dist = model.predict(states)
        action_num = np.argmax(action_dist)
        action = ACTIONS[action_num]
        next_state, reward, done, _ = env.step(action)
        env.render()
        states = next_state.reshape(state_shape)

        if done:
            print('Track successfully completed')
            break

    env.close()


if __name__ == "__main__":
    main()
