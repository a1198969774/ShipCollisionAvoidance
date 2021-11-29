import argparse
import gym

import numpy as np
import random
import tensorflow as tf
import PIL

from env import envModel
from replayMemory import ReplayMemory, PriorityExperienceReplay
from model import create_deep_q_network, create_duel_q_network, create_model, create_distributional_model
from agent import DQNAgent


NUM_FRAME_PER_ACTION = 4
UPDATE_FREQUENCY = 4 # do one batch update when UPDATE_FREQUENCY number of new samples come
TARGET_UPDATE_FREQENCY = 10000 # target-net更新频率
REPLAYMEMORY_SIZE = 500000 # 经验池的大小
MAX_EPISODE_LENGTH = 100000 # 最大的序列长度
RMSP_EPSILON = 0.01
RMSP_DECAY = 0.95
RMSP_MOMENTUM =0.95
NUM_FIXED_SAMPLES = 10000
NUM_BURN_IN = 5000
LINEAR_DECAY_LENGTH = 4000000
NUM_EVALUATE_EPSIODE = 20

def get_fixed_samples(env,num_actions,num_samples):
    ##fixed_samples = []
    ##num_environment = env.num_process
    env.reset()
    old_state, action, reward, new_state, is_terminal = env.get_state()
    action = np.random.randint(0, num_actions)
    env.step(action)
    return np.array(new_state)
    #     env.take_action(action)
    # for _ in range(0,num_samples,num_environment):
    #     old_state,action,reward,new_state,is_terminal = env.get_state()
    #     action = np.random.randint(0,num_actions,size = (num_environment,))
    #     env.take_action(action)
    #     for state in new_state:
    #         fixed_samples.append(state)
    #return np.array(fixed_samples)



def main():
    parser = argparse.ArgumentParser(description='Run DQN ')
    parser.add_argument('--env', default='SpaceInvaders-v0')
    parser.add_argument('--seed', default=10703, type=int)
    parser.add_argument('--input_shape', default=(80,80))
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--epsilon', default=0.1)
    parser.add_argument('--learning_rate', default=0.00025)
    parser.add_argument('--window_size', default=4, type = int)
    parser.add_argument('--batch_size', default=32, type = int)
    #parser.add_argument('--num_process', default=1, type = int)
    parser.add_argument('--num_iteration', default=20000000, type = int)
    parser.add_argument('--eval_every', default=0.001, type = float)
    parser.add_argument('--is_duel', default=1, type = int )
    parser.add_argument('--is_double', default=1, type = int )
    parser.add_argument('--is_per', default=0, type = int)
    parser.add_argument('--is_distributional', default=0, type = int)
    parser.add_argument('--num_step', default=1, type = int)
    parser.add_argument('--is_noisy', default=1, type = int)


    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)
    print('Environment : %s.' % (args.env,))

    ##env = gym.make(args.env)
    env = envModel()
    num_actions = env.action_space_n
    print('number actions %d' % (num_actions,))

    ##env.close()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    ##batch_environment = BatchEnvironment(args.env, args.num_process,
                                         ##args.window_size, args.input_shape, NUM_FRAME_PER_ACTION, MAX_EPISODE_LENGTH)

    # 是否使用优先经验回放
    if args.is_per == 1:
        replay_memory = PriorityExperienceReplay(REPLAYMEMORY_SIZE,args.window_size,args.input_shape)
    else:
        replay_memory = ReplayMemory(REPLAYMEMORY_SIZE,args.window_size,args.input_shape)

    create_network_fn = create_deep_q_network if args.is_duel == 0 else create_duel_q_network

    create_model_fn = create_model if args.is_distributional == 0 else create_distributional_model

    noisy = True if args.is_noisy == 1 else False

    eval_model,eval_params = create_model_fn(args.window_size,args.input_shape,num_actions,
                                             'eval_model',create_network_fn,trainable=True,noisy=noisy)
    target_model,target_params = create_model_fn(args.window_size,args.input_shape,num_actions,
                                                 'target_model',create_network_fn,trainable=False,noisy=noisy)

    update_target_params_ops = [t.assign(s) for s,t in zip(eval_params,target_params)]

    agent = DQNAgent(eval_model,
                     target_model,
                     replay_memory,
                     num_actions,
                     args.gamma,
                     UPDATE_FREQUENCY,
                     TARGET_UPDATE_FREQENCY,
                     update_target_params_ops,
                     args.batch_size,
                     args.is_double,
                     args.is_per,
                     args.is_distributional,
                     args.num_step,
                     args.is_noisy,
                     args.learning_rate,
                     RMSP_DECAY,
                     RMSP_MOMENTUM,
                     RMSP_EPSILON)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(update_target_params_ops)

        print('prepare fixed samples for mean max q')
        ##get state and action
        fixed_samples = get_fixed_samples(env, num_actions, NUM_FIXED_SAMPLES)

        ##agent.fit(sess,batch_environment,NUM_BURN_IN,do_train=False)
        agent.fit(sess, env, NUM_BURN_IN,do_train=False)

        # Begin to train:
        fit_iteration = int(args.num_iteration * args.eval_every)

        for i in range(0, args.num_iteration, fit_iteration):
            # Evaluate:
            reward_mean, reward_var = agent.evaluate(sess, env, NUM_EVALUATE_EPSIODE)
            mean_max_Q = agent.get_mean_max_Q(sess, fixed_samples)
            print("%d, %f, %f, %f" % (i, mean_max_Q, reward_mean, reward_var))
            # Train:
            agent.fit(sess, env, fit_iteration, do_train=True)
            break

    env.close()







if __name__ == '__main__':
    main()

