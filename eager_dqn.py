import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import gym
import time

tfe.enable_eager_execution()
tf.set_random_seed(1)

env = gym.make('CartPole-v0')


class Model(object):
    """
    defining the architecture of the neural net
    """

    def __init__(self):
        self._input_shape = [-1, env.observation_space.shape[0] + env.action_space.n]
        self.dense1 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.layers.Dense(256, activation=tf.nn.relu)
        self.dense3 = tf.layers.Dense(128, activation=tf.nn.relu)
        self.dense4 = tf.layers.Dense(1)
        self.dropout = tf.layers.Dropout(0.1)

    def predict(self, inputs):
        # input is the state and the action (converted to a one hot vector)
        x = tf.reshape(inputs, self._input_shape)
        x = self.dropout(self.dense1(x))
        x = self.dropout(self.dense2(x))
        x = self.dense3(x)
        x = self.dense4(x)
        return x


def loss(model, inputs, targets):
    """
    loss is the mean squared error
    """
    return tf.reduce_mean(tf.square(model.predict(inputs) - targets))


def train(model, n_iterations):
    gamma = 0.99  # how much to discount future rewards
    epsilon_final = 0.1  # how often we want to choose a random action at the end of training
    training_batch_size = 2000  # size of the batch of training data

    memory = []  # past states, actions, rewards to learn from
    for i in range(1, n_iterations + 1):
        epsilon = 1 - (1 - epsilon_final) * (i / n_iterations)  # linearly go from epsilon=1 to epsilon=epsilon_final
        print("iteration", i, "out of", n_iterations)
        print("epsilon:", epsilon)
        state = env.reset()
        this_reward = 0
        for _ in range(env._max_episode_steps):
            env.render()
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # take a random action with probability epsilon
            else:
                action = int(argmax_a(model, state))  # otherwise, take the action that maximizes the estimated Q value

            state_, reward, done, _ = env.step(action)
            state = tf.reshape(state, [-1, env.observation_space.shape[0]])
            action = tf.reshape(tf.cast(tf.equal(tf.range(env.action_space.n), action), tf.double),
                                [-1, env.action_space.n])
            memory.append([tf.concat([state, action], 1), reward, state_, done])  # append data to memory to learn it
            this_reward += reward
            state = state_
            if done:
                break

        print("reward:", this_reward)

        np.random.shuffle(memory)  # shuffle the memory so as to take a random subset
        if len(memory) < training_batch_size:
            training_data = memory
        else:
            training_data = memory[:training_batch_size]

        training_x = []
        training_y = []

        for data in training_data:
            state_action = data[0]
            reward = data[1]
            state_ = data[2]
            done = data[3]

            training_x.append(state_action)
            if done:
                # if state is terminal, Q should output the reward at that step
                training_y.append(reward)
            else:
                # if not terminal, Q should output the reward + gamma * estimated future rewards
                training_y.append(reward + gamma * tf.reduce_max(Q(model, tf.tile(state_, [env.action_space.n]),
                                                                   tf.reshape(tf.range(env.action_space.n), [-1, 1]))))

        training_x = tf.reshape(training_x, [-1, env.observation_space.shape[0] + env.action_space.n])
        training_y = tf.reshape(training_y, [-1, 1])

        gradient = tfe.implicit_gradients(loss)  # find the gradient so we can minimize the loss
        optimizer = tf.train.AdamOptimizer()  # define the optimizer being used
        optimizer.apply_gradients(gradient(model, training_x, training_y))  # update the network's weights
        this_loss = loss(model, training_x, training_y).numpy()
        print("training set size:", len(training_data))
        print("loss:", this_loss)
        print("-----------------------------------")

        training_data = []
        training_x = []
        training_y = []


def argmax_a(model, state):
    return tf.argmax(Q(model, tf.tile(state, [env.action_space.n]), tf.reshape(tf.range(env.action_space.n), [-1, 1])))


def Q(model, state, action):
    state = tf.reshape(state, [-1, env.observation_space.shape[0]])
    action = tf.reshape(tf.cast(tf.equal(tf.range(env.action_space.n), action), tf.double), [-1, env.action_space.n])
    return model.predict(tf.concat([state, action], 1))


if __name__ == "__main__":
    my_model = Model()
    t0 = time.time()
    train(my_model, 250)
    print("training time:", time.time() - t0)

    rewards = []
    for i in range(100):
        state = env.reset()
        episode_reward = 0
        for _ in range(env._max_episode_steps):
            env.render()
            action = int(argmax_a(my_model, state))
            state_, reward, done, _ = env.step(action)
            episode_reward += reward
            state = state_
            if done:
                break
        rewards.append(episode_reward)
        print(i, episode_reward)

    print(np.array(rewards).mean())
