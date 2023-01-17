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

# Global mapping of action ID to Ms. Pac-Man arcade joystick movements
ACTION_MAP = {
    0: "NOOP",
    1: "UP",
    2: "RIGHT",
    3: "LEFT",
    4: "DOWN",
    5: "UPRIGHT",
    6: "UPLEFT",
    7: "DOWNRIGHT",
    8: "DOWNLEFT"
}


def parse() -> dict:
    """Parses CLI arguments for script parameters"""

    # Set-up top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Indicator to include verbose logging")

    # Set-up subparsers
    subparser = parser.add_subparsers(dest='command')
    subparser.required = True

    # Subparser to train a model using any (one) of the possible modifications
    train = subparser.add_parser('train')
    train.add_argument('--model', type=str, default='DQN',
        choices=('DQN','PPO'),
        help='''Model algorithm to use''')
    train.add_argument('--experiment', type=str, default='baseline', nargs='+',
        choices=('baseline','grayscale','longevity_rewards','bonus_scaling',
        'noop','action_space_reduction'),
        help='''Specify the experiment(s) to run. Note that some combinations
        of experiments may not make sense, i.e. longevity with bonus scaling
        would result in just longevity being implemented, because bonuses are
        irrelevant under that scenario''')
    train.add_argument('--bonus-scalar', type=int, default=1,
        help='''Scalar to be applied to bonus rewards during training. Includes
        power orbs, fruits, and ghosts, but does not include standard orbs.
        Requires experiment "bonus_scaling" to be specified.''')
    train.add_argument('--noop-frames', type=int, default=0,
        help='''Number of frames to execute only NOOP at the beginning of
        training. Due to the jingle at the start of each simulation, this
        value is added to 265 to produce NOOP commands during playable
        frames. Requires experiment "noop" to be specified.''')

    # Subparser used to evaluate an already trained and saved model. The
    # exception is the random action model, which does not require training
    eval = subparser.add_parser('eval')
    eval.add_argument('--max_steps', type=int, default=10000,
        help="Max number of steps the agent will take throughout each rollout")
    eval.add_argument('--n_rollout', type=int, default=1,
        help="Number of rollouts to generate")
    eval.add_argument('--n_traj', type=int, default=20,
        help="Number of trajectories per rollout")
    eval.add_argument('--logging_base_path', type=str, default='../output',
        help="Directory in which to create logging subfolders")
    eval.add_argument('--policy_type', type=str, default='random',
        choices=('random','dqn','ppo'),
        help="Type of policy to use")
    eval.add_argument('--policy_file', type=str, default='',
        help="Saved model .zip file to use for determining the next action")

    # Subparser to generate plots of already evaluated models
    subparser.add_parser('plot')

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    return params

def setup_dirs(logging_base_path: str, include_subfolders: bool) -> str:
    """Creates directories for logging, metrics, and image/video outputs"""

    # Create base path if it doesn't already exist
    if not os.path.exists(logging_base_path):
        os.makedirs(logging_base_path, mode = 0o777, exist_ok = False)

    # Create folder to store trajectories
    timestamp = dt.strftime(dt.now(), '%Y%m%d_%H%M%S')
    logging_path = os.path.join(logging_base_path, timestamp)
    os.mkdir(logging_path)

    if include_subfolders:
        os.mkdir(os.path.join(logging_path, 'images'))

    return logging_path


def initialize_traj(logging_base_path: str, env) -> None:
    """Helper function to set-up trajectory logging and capture first observation"""

    # Initialize individual trajectory with data to track for metrics
    logging_dir = setup_dirs(logging_base_path=logging_base_path, include_subfolders=True)

    # Reset environment and record initial state
    obs = env.reset()
    frame_image = Image.fromarray(obs[0]).resize((500,500))
    frame_image.save(os.path.join(logging_dir, 'images', 'frame_000000.jpg'))

    return logging_dir, obs

def save_image(frame_number: int, env, logging_dir: str) -> None:
    """Saves the environment's current image in the specified logging directory"""

    frame_num = str(frame_number).zfill(6)

    frame_image = env.render(mode = 'rgb_array')
    frame_image = Image.fromarray(frame_image).resize((500,500))
    frame_image.save(os.path.join(logging_dir, 'images', 'frame_{}.jpg'.format(frame_num)))

def main():
    """Runs the main simulation. Captures metrics, images, and video of the agent"""

    params = parse()

    import ipdb; ipdb.set_trace()

    if params['command'] == 'train':
        pass

    elif params['command'] == 'eval':
        pass

    elif params['command'] == 'plot':
        pass

    else:
        raise "Invalid command entered!"

    policy_type = params['policy_type']
    policy_file = params['policy_file']
    n_traj = params['n_traj']

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

    for traj in range(n_traj):

        # Initialize trajectory and get first observation
        print('\n Starting trajectory {}! \n'.format(traj+1))
        logging_dir, obs = initialize_traj(logging_base_path=logging_dir_parent, env=env)
        trajectory = []
        total_reward = 0

        # Run the simulation
        for move in range(params['max_steps']):

            if not policy:
                action = env.action_space.sample()
            else:
                action = int(policy.predict(obs)[0])

            new_obs, reward, done, status = env.step(action)
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
                    action=ACTION_MAP[action],
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
