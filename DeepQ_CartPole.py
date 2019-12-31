# Implementation of Deep-Q network agent for CarePole-v0 environment compatible with TensorFlow 2.0
#

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np           # Handle matrices
import gym                # Retro Environment
from collections import deque

env = gym.make('CartPole-v0')
print("The state size is: ", env.observation_space)
print("The action size is : ", env.action_space.n)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
print("Possible actions: ", possible_actions)

# hyper-parameters
action_size = env.action_space.n  # 2 possible actions
total_episodes = 10000           # Total episodes for training
max_steps = 200              # Max possible steps in an episode
batch_size = 32               # Batch size
explore_start = 1.0           # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 1e-4              # exponential decay rate for exploration prob
lr_start = 1e-1                 # learning rate at start
lr_stop = 1e-3                 # learning rate at stop
lr_decay = 1e-4               # learning rate decay rate
gamma = 1.0                    # Discounting rate
memory_size = 5000          # memory size

# training or test
training = False  # if false, test the saved network


# network definition
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='tanh', input_shape=(None, 4)))
model.add(tf.keras.layers.Dense(32, activation='tanh'))
model.add(tf.keras.layers.Dense(2))


# class definition
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size_):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size_,
                                 replace=False)
        return [self.buffer[ii] for ii in index]


memory = Memory(max_size=memory_size)


# function definition
def predict_action(explore_start_, explore_stop_, decay_rate_, decay_step_, state_, actions_):
    rand_val = np.random.rand()
    explore_prob = explore_stop_ + (explore_start_ - explore_stop_) * np.exp(-decay_rate_ * decay_step_)

    if explore_prob > rand_val:
        choice_ = np.random.randint(env.action_space.n)  # 0, 1, 3... integer actions
        action_ = actions_[choice_]  # one-hot encoding actions

    else:
        qs = model(state_.reshape((1, 4)))
        choice_ = np.argmax(qs)
        action_ = actions_[choice_]

    return action_, choice_, explore_prob


def backprop(st, tar, act, lr_start_, lr_stop_, lr_decay_, lr_decay_step):
    with tf.GradientTape() as tape:
        output = model(st)
        q = tf.reduce_sum(tf.multiply(output, act))
        loss_ = tf.reduce_mean(tf.square(tar - q))
    grad = tape.gradient(loss_, model.trainable_variables)
    learning_rate = lr_stop_ + (lr_start_ - lr_stop_) * np.exp(-lr_decay_ * lr_decay_step)
    tf.keras.optimizers.Adam(learning_rate).apply_gradients(zip(grad, model.trainable_variables))


# start training
decay_step = 0
if training:
    rewards_list = []

    for episode in range(total_episodes):
        step = 0

        episode_rewards = []

        state = env.reset()

        while step < max_steps:

            step += 1

            decay_step += 1

            action, choice, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                                 state, possible_actions)
            next_state, reward, done, _ = env.step(choice)

            episode_rewards.append(reward)

            # If the game is finished
            if done:
                next_state = np.zeros((4,), dtype=np.int)

                # if step < 195:
                #     reward -= 200
                #     episode_rewards.append(-200)

                step = max_steps

                total_reward = np.sum(episode_rewards)

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Explore P: {:.4f}'.format(explore_probability))

                rewards_list.append((episode, total_reward))

                # Store transition <st,at,rt+1,st+1> in memory D
                memory.add((state, action, reward, next_state, done))

            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # st+1 is now our current state
                state = next_state

        if len(memory.buffer) > batch_size:
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=2)
            actions_mb = np.array([each[1] for each in batch], dtype='float32')
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=2)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            Qs_next_state = model(next_states_mb)

            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i, :])
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch], dtype='float32')
            backprop(states_mb, targets_mb, actions_mb, lr_start, lr_stop, lr_decay, decay_step)

        if episode % 100 == 0:
            model.save_weights("./check/model_weights.h5")
            print("Model Saved")

else:
    model.load_weights('./check/model_weights.h5')
    print("Model Loaded")

    step = 0

    state = env.reset()

    while step < max_steps:

        step += 1
        env.render()
        qs_test = model(state.reshape((1, 4)))
        choice = np.argmax(qs_test)
        next_state, reward, done, _ = env.step(choice)
        state = next_state
