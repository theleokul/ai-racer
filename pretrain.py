import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import json
from collections import OrderedDict
import copy


clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


ACTIONS = OrderedDict({
    'ACC': np.array([0, 1, 0]),
    'IDLE': np.array([0, 0, 0]),
    'BR': np.array([0, 0, 0.4]),
    'LEFT': np.array([-1, 0, 0]),
    'RIGHT': np.array([1, 0, 0]),
    'LEFT_BR': np.array([-1, 0, 0.4]),
    'RIGHT_BR': np.array([1, 0, 0.4]),
})
ACTIONS_LIST = list(ACTIONS.values())


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    std_adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    return returns, std_adv


def key_press(k, mod):
    global restart
    # if k == 0xff0d: restart = True
    if k == key.ESCAPE:
        restart = True
    if k == key.UP:
        a[3] = +1.0
        if a[0] == 0.0:
            a[1] = +1.0
    if k == key.LEFT:
        a[0] = -1.0
        a[1] = 0.0  # Cut gas while turning
    if k == key.RIGHT:
        a[0] = +1.0
        a[1] = 0.0  # Cut gas while turning
    if k == key.DOWN:
        a[2] = +0.4  # stronger brakes


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0.0
        if a[3] == 1.0:
            a[1] = 1.0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0.0
        if a[3] == 1.0:
            a[1] = 1.0
    if k == key.UP:
        a[1] = 0.0
        a[3] = 0.0
    if k == key.DOWN:
        a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    os.makedirs(datasets_dir, exist_ok=True)
    data_file = os.path.join(datasets_dir, "data-%s.pkl" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)


def save_results(episode_reward, results_dir="./results"):
    os.makedirs(results_dir, exist_ok=True)
    results = dict()
    results["episode_reward"] = episode_reward
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(fname, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-data", action='store_true', default=False, help="Collect the data in a pickle file.")
    args = parser.parse_args()
    should_collect = args.collect_data

    good_samples = {
        'states': [],
        'actions': [],
        'values': [],
        'masks': [],
        'rewards': [],
        'actions_probs': [],
        'actions_onehot': []
    }

    env = gym.make('CarRacing-v0').unwrapped
    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    n_actions = len(ACTIONS)
    a = np.zeros(4, dtype=np.float32)
    episode_rewards = []

    # Episode loop
    while True:
        episode_reward = 0
        episode_samples = copy.deepcopy(good_samples)
        state = env.reset()
        restart = False
        steps_count = 0
        # State loop
        while True:
            next_state, r, done, info = env.step(a[:3])
            episode_reward += r

            episode_samples["states"].append(state)  # state has shape (96, 96, 3)
            episode_samples["actions"].append(np.array(a[:3]))  # action has shape (1, 3)
            episode_samples["values"].append(1 if r > 0 else -1)
            episode_samples["masks"].append(not done)
            episode_samples["rewards"].append(r)
            actions_probs = np.zeros(n_actions)
            action_ind = list(filter(lambda x: np.allclose(x[1], a[:3]), enumerate(ACTIONS_LIST)))[0][0]
            actions_probs[action_ind] = 1
            episode_samples["actions_probs"].append(actions_probs)
            episode_samples["actions_onehot"].append(actions_probs)

            state = next_state
            steps_count += 1

            env.render()
            if done or restart:
                break

        if done and should_collect:
            # We done if we are here
            good_samples = copy.deepcopy(episode_samples)
            print('Steps gone: ', steps_count)
            print('... saving data')
            store_data(good_samples, "./data")
            save_results(episode_reward, "./results")
            print('... finished')
            break

    env.close()
