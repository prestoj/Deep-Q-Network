import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import gym
import time

np.random.seed(1)

env = gym.make('CartPole-v0')

model = Sequential()
# input is the state and the action (converted to a one hot vector)
model.add(Dense(units=128, input_dim=env.observation_space.shape[0] + env.action_space.n,
                kernel_initializer='uniform', bias_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=256, kernel_initializer='uniform', bias_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=128, kernel_initializer='uniform', bias_initializer='uniform', activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mse')


def train(n_iterations):
    gamma = 0.99  # how much to discount future rewards
    epsilon_final = 0.1  # how often we want to choose a random action at the end of training
    training_batch_size = 1000  # size of the batch of training data

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
                action = argmax_a(state)  # otherwise, take the action that maximizes the estimated Q value

            state_, reward, done, _ = env.step(action)
            memory.append([state, action, reward, state_, done])  # append data to memory to learn it
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
            state = data[0]
            action = data[1]
            reward = data[2]
            state_ = data[3]
            done = data[4]

            training_x.append(reshape_input(state, action))

            if done:
                # if state is terminal, Q should output the reward at that step
                training_y.append(reward)
            else:
                # if not terminal, Q should output the reward + gamma * estimated future rewards
                training_y.append(reward + gamma * Q(state_, argmax_a(state_)))

        training_x = np.array(training_x).reshape((len(training_data),
                                                   env.observation_space.shape[0] + env.action_space.n))

        model.fit(np.array(training_x), np.array(training_y), epochs=1)  # learn from the data

        training_data = []
        training_x = []
        training_y = []


def argmax_a(state):
    best_action = None
    max_Q = float("-inf")
    for i in range(env.action_space.n):
        this_Q = Q(state, i)
        if this_Q > max_Q:
            best_action = i
            max_Q = this_Q

    return best_action


def Q(state, action):
    return model.predict(reshape_input(state, action))[0, 0]


def reshape_input(state, action):
    action_vec = np.array(np.arange(env.action_space.n) == action, dtype=int)
    return np.append(state, action_vec).reshape((1, env.observation_space.shape[0] + env.action_space.n))


if __name__ == "__main__":
    t0 = time.time()
    train(250)
    print("training time: ", time.time() - t0)

    rewards = []
    for i in range(100):
        this_reward = 0
        state = env.reset()
        for _ in range(env._max_episode_steps):
            env.render()
            action = argmax_a(state)
            state_, reward, done, _ = env.step(action)
            this_reward += reward
            state = state_
            if done:
                break
        print(i, this_reward)
        rewards.append(this_reward)

    print(np.array(rewards).mean())
