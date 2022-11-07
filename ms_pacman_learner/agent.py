import argparse
import ffmpeg
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

from datetime import datetime as dt
from PIL import Image
from stable_baselines3 import PPO, DQN


def parse() -> dict:
    """Parses CLI arguments for script parameters"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_steps', type=int, default=10000,
        help="Max number of steps the agent will take throughout each rollout")
    parser.add_argument('--n_rollout', type=int, default=1,
        help="Number of rollouts to generate")
    parser.add_argument('--n_traj', type=int, default=20,
        help="Number of trajectories per rollout")
    parser.add_argument('--logging_base_path', type=str, default='/project/logs/',
        help="Directory in which to create logging subfolders")
    parser.add_argument('--policy_type', action=str, default='random',
        choices=('random','dqn','ppo'),
        help="Type of policy to use")
    parser.add_argument('--policy_file', type=str, default='',
        help="Saved model .zip file to use for determining the next action")

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    return params

def setup_dirs(logging_base_path: str, include_subfolders: bool) -> str:
    """Creates directories for logging, metrics, and image/video outputs"""

    # Create folder to store trajectories
    timestamp = dt.strftime(dt.now(), '%Y%m%d_%H%M%S')
    logging_path = os.path.join(logging_base_path, timestamp)
    os.mkdir(logging_path)

    if include_subfolders:
        os.mkdir(os.path.join(logging_path, 'images'))

    return logging_path


def initialize_traj(logging_base_path: str, env) -> None:
    """Helper function to set-up trajectory logging and capture first observation"""

    print('\n Starting trajectory {}! \n'.format(traj+1))

    # Initialize individual trajectory with data to track for metrics
    logging_dir = setup_dirs(logging_base_path=logging_dir_parent, include_subfolders=True)

    # Reset environment and record initial state
    obs = env.reset()
    frame_image = Image.fromarray(obs[0]).resize((500,500))
    frame_image.save(os.path.join(logging_dir, 'images', 'frame_000000.jpg'))

    return obs

def save_image(frame_number: int, env, logging_dir: str) -> None:
    """Saves the environment's current image in the specified logging directory"""

    frame_num = str(frame_number).zfill(6)

    frame_image = env.render(mode = 'rgb_array')
    frame_image = Image.fromarray(frame_image).resize((500,500))
    frame_image.save(os.path.join(logging_dir, 'images', 'frame_{}.jpg'.format(frame_num)))

def main():
    """Runs the main simulation. Captures metrics, images, and video of the agent"""

    params = parse()
    policy_type = params['policy_type']
    policy_file = params['policy_file']

    # Initialize the environment and retrieve pre-trained policy
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', frameskip=1)

    if policy_type == 'dqn':
        policy = DQN.load(policy_file)
    elif policy_type == 'ppo':
        policy = PPO.load(policy_file)
    else:
        policy = None
        print('Using random agent policy!')

    # Create parent directory and initialize list of trajectories
    logging_dir_parent = setup_dirs(logging_base_path=params['logging_base_path'],
        include_subfolders=False)
    trajectories = []

    for traj in range(params['n_traj']):

        # Initialize trajectory and get first observation
        obs = initialize_traj(logging_base_path=logging_dir_parent, env=env)
        trajectory = []

        # Run the simulation
        for move in range(params['max_steps']):

            if not policy:
                action = env.action_space.sample()
            else:
                action = int(policy.predict(obs)[0])

            new_obs, reward, done, status = env.step(random_action)
            lives = status['lives']
            episode_frame_number = status['episode_frame_number']
            status.pop('rgb')

            new_metrics = (reward, done, status)
            trajectory.append(new_metrics)
            # reward = trajectory[move][0]
            # done = trajectory[move][1]
            # lives = trajectory[move][2]

            # print('Move {move}: {action}, reward: {reward}, done: {done}, {status}' \
            #       .format(move=move+1, action=ACTION_MAP[random_action], reward=reward, done=done, status=status))

            print('''Move {move}: {action}, reward: {reward}, done: {done},
                lives: {lives}, trajectory_frame: {traj_frame_num},
                rollout_frame: {rollout_frame_num}''' \
                .format(move=move+1,
                    action=ACTION_MAP[random_action],
                    reward=reward,
                    done=done,
                    lives=status['lives'],
                    traj_frame_num=status['episode_frame_number'],
                    rollout_frame_num=status['frame_number']
                )
            )

            save_image(frame_number=status['episode_frame_number'], env=env,
                logging_dir=logging_dir)

            obs = new_obs

            # Stop simulation if terminal state has been reached
            if done:
                print('Simulation has ended!')
                break

        trajectories.append(trajectory)

        print('Creating video for trajectory {}!'.format(traj+1))
        ffmpeg.input(os.path.join(logging_dir, 'images', '*.jpg'), pattern_type='glob') \
            .output(os.path.join(logging_dir, 'pacman.mp4')) \
            .run()

    # Save all metrics to pickle file
    assert len(trajectories) == n_traj
    print('Pickling metrics!')
    with open(os.path.join(logging_dir_parent, 'metrics.pickle'), 'wb') as pkl:
        pickle.dump(trajectories, pkl)
        pkl.close()

if __name__ == '__main__':
    main()
