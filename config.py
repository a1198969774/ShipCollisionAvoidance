import argparse


class Config:
    parser = argparse.ArgumentParser(description='Run DQN ')
    parser.add_argument('--env', default='SpaceInvaders-v0')
    parser.add_argument('--seed', default=10703, type=int)
    parser.add_argument('--input_shape', default=(80, 80))
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--max_step', default=700)
    parser.add_argument('--max_episode', default=500)
    parser.add_argument('--epsilon', default=0.1)
    parser.add_argument('--learning_rate', default=0.00025)
    parser.add_argument('--window_size', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--num_process', default=1, type = int)
    parser.add_argument('--num_iteration', default=20000000, type=int)
    parser.add_argument('--eval_every', default=0.001, type=float)
    parser.add_argument('--is_duel', default=1, type=int)
    parser.add_argument('--is_double', default=1, type=int)
    parser.add_argument('--is_per', default=0, type=int)
    parser.add_argument('--is_distributional', default=0, type=int)
    parser.add_argument('--num_step', default=1, type=int)
    parser.add_argument('--is_noisy', default=0, type=int)

    parser.add_argument('--self_speed', default=18.52)
    parser.add_argument('--self_heading', default=0)
    parser.add_argument('--target_speed', default=18.52)
