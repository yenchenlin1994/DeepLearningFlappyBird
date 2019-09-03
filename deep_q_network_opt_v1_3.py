#!/usr/bin/env python

# distinction between dqn (vanilla) and this:
# we want to formulate an addition to state, s, called "delta s", ds, which will cause the Q value of s+ds and a
# target action, a^t, to be higher than the q value for s+ds and any other action. The implication is that adding ds
# to our data will result in an action to be drawn. In cases where ds is not added/nothing is changed, we will
# behave normally.
# In order to retain the natural behaviour of a DQN controller, we will not be changing the weights or controller
# for that matter, we will instead minimize the loss between Q(s+ds, a^t) and Q(s+ds, a), where a^t is the target a,
# and a is any non-target a. Thee loss function will be (some) hinge loss: l(a,b) = max(b - (a + eps), 0), which
# will essentially enforce the condition: a >= b + eps
# An informal proof associated with the possibility this will work depends on the fact that our controller learns how
# to behave well within a certain set of input states from the possible set of states it's been trained on,
# called (here) the game-possible pixel space. This is the set of frames the game can generate under any scenario
# within the game. The game-possible pixel space is small, relative to the pixel space, which is a space containing
# all possible combinations of pixel intensities that a screen can generate. Given that ds blongs in the pixel space,
# and s belongs in the game-possible pixel space, we can say that s+ds belongs in the pixel space, which the controller
# may not know how to handle, which, if abused properly may result in a simple adversarial attack.

from __future__ import print_function
import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import tensorflow.contrib.slim as slim
from scipy.optimize import minimize

# const for advesarial optimization:
# flap = [0, 1]
# noaction = [1, 0]
action_target = [0, 1]
LR = 0.01 # learning rate for optimizing ds
# number of time steps in which to calculate ds, first 10 frames of game
INTERVAL = 10
OPTIMIZATION_EPSILON = 2


GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # new vars for optimization
    delta_s = tf.Variable(
        name="added_perturbation",
        initial_value=tf.random.normal([INTERVAL, 80, 80, 4], mean=(255.0 / 2), stddev=(255 * (0.01 ** 0.5))),
        trainable=True)

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])
    optimization = tf.placeholder("bool")
    s_opt = tf.cond(optimization,
            lambda: s + delta_s, #T
            lambda: s)  #F
    tf.summary.histogram("s_opt", s_opt)

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s_opt, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, optimization, delta_s, readout, h_fc1


def trainNetwork(s, optimization, delta_s, readout, h_fc1, sess):

    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks\
    variables_to_restore = slim.get_variables_to_restore(exclude=["added_perturbation"])
    saver = tf.train.Saver(var_list=variables_to_restore)
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # # new vars for optimization
    # delta_s = tf.Variable(
    #     name="added_perturbation",
    #     initial_value=tf.random.normal([INTERVAL, 80, 80, 4], mean=(255.0 / 2), stddev=(255 * (0.01 ** 0.5))),
    #     trainable=True)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t],
                                            optimization : False})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()


        # attack first few frames
        if t > INTERVAL:
            # generate a ds
            #delta_s.initializer # since we restore the network variables, we need to init this one separately to
                                  # save us some struggle with the saver

            # sample a minibatch to optimize on, the entire sequence of frames thus far in the program
            opt_batch = random.sample(list(D), INTERVAL)

            # get the batch variables
            s_opt_batch = [d[0] for d in opt_batch] # only take stats form opt batch

            #delta_s.assign_add(s_opt_batch) # s+ds

            # taking params into attack operation:
            # intake batch of s_t, a_t, r_t, s_t1 resulted from a normal controller devised optimization, across batch by modulating ds,
            # reduce the sum (expected) loss between Q(s+ds, a^t) and Q(s+ds, a), where a^t != a.
            # subjects: batch of s_t, a_t, r_t, s_t1, size BATCH.
            # Note: the training is complete, so in theory we shouldn't be using this as a SGD batch anymore.
            # Note: in this formulation, we are letting s + ds form by optimizing s to meet our objective. There is no
            # explicit ds added, it is formed.
            #
            # theory/idea:
            # RL is less suceptible to a stationary attack since reproducing a setting is challenging. We can go through an
            # interaction, record it, and then produce an optimization which will abuse that origional interaction, but when
            # will that come in handy? In CV, usually repeated inputs are easy to produce, but in an RL setting there isn't
            # opportunity for repeated input.
            # if we find that one adversarial input transfers well to other similar images, maybe we can make a case here.

            # Q values for both actions at state s + ds for entire batch

            # you just need to feed into s
            # ds generates automatically
            # then you use s+ds as your new input
            # talk after meeting have Q's

            init_perturbation = np.random.normal(loc=(255.0 / 2), scale=(255 * (0.01 ** 0.5)), size=([INTERVAL, 80, 80, 4]))

            # scipy objective function to minimize
            def adv_objective(s_snippet):
                q_vals = readout.eval(feed_dict={s: [s_snippet][0],
                                                       optimization: False})
                # no_flap = readout[0]  # target action, a
                # flap = readout[1]  # b
                return q_vals

            def constraint1(s_snippet):
                q_vals = readout.eval(feed_dict={s: [s_snippet][0],
                                                 optimization: False})
                a = q_vals[0] # no flap, target action
                b = q_vals[1] # flap
                loss = b - a + OPTIMIZATION_EPSILON
                return loss

            # # readout(s) = [Q(no flap), Q(flap)]
            # # a = readout[target_action]
            # a = tf.placeholder("float", shape=[INTERVAL,], name="a") # readout_s_ds[1]
            # a_val = readout_s_ds[:,0]
            #
            # # b = readout[!target_action]
            # b = tf.placeholder("float", shape=[INTERVAL,], name="b") # readout_s_ds[0]
            # b_val = readout_s_ds[:,1]

            # opt = tf.train.GradientDescentOptimizer(LR, name="GRADDESC").minimize(loss, var_list=[delta_s])
            # opt.run()

            res = minimize(adv_objective, s_opt_batch, method='BFGS', constraints=({'type':'ineq', 'fun': constraint1}))
            print(res)


        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    s, optimization, delta_s, readout, h_fc1 = createNetwork()
    trainNetwork(s, optimization, delta_s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()

# after digging in the tf src i think i need to better specify the subject of the optimization. maybe implict isnt the way to go.
# comment prev. src so they are up to date.