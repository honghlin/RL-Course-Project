import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import random


class Agent_Q:
    def __init__(self, env):

        self.env = env
        self.env_action_space = env.action_space
        self.num_episodes = 1500
        self.q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))
        self.max_number_of_steps = 200

    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):

        cart_pos, cart_v, pole_angle, pole_v = observation

        digitized = [np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, 4)),
                     np.digitize(cart_v, bins=self.bins(-3.0, 3.0, 4)),
                     np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, 4)),
                     np.digitize(pole_v, bins=self.bins(-2.0, 2.0, 4))]

        return sum([x * (4 ** i) for i, x in enumerate(digitized)])

    def get_action(self, state, action, observation, reward, episode):

        next_state = self.digitize_state(observation)

        epsilon = 0.5 * (0.99 ** episode)

        if epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(self.q_table[next_state])
        else:
            next_action = np.random.choice([0, 1])

        alpha = 0.2
        gamma = 0.99

        self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] + alpha * (
                reward + gamma * self.q_table[next_state, next_action])

        return next_action, next_state

    def act(self, s):
        next_state = self.digitize_state(s)
        next_action = np.argmax(self.q_table[next_state])
        return next_action

    # def act_egreedy(self, s, env_action_space, epsilon):
    #
    #     randsed = float(np.random.rand(1))
    #     if randsed < epsilon:
    #         a = env_action_space.sample()
    #     else:
    #         sess = tf.get_default_session()
    #         feed_dict = {self.s_input: [s]}
    #         Q_value = sess.run([self.Q_value], feed_dict=feed_dict)[0]
    #         a = np.argmax(Q_value)
    #
    #     return a

    def init_all_var(self):
        self.q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, self.env.action_space.n))

    def clear_say_buffer(self):
        self.q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, self.env.action_space.n))

    def train(self, env):

        for episode in range(self.num_episodes):
            observation = env.reset()
            state = self.digitize_state(observation)
            action = np.argmax(self.q_table[state])
            episode_reward = 0

            for t in range(2 * self.max_number_of_steps):

                observation, reward, done = env.step(action)  # _with_real_done , info
                #if done:
                    # reward = -200
                    #assert reward == -200
                action, state = self.get_action(state, action, observation, reward, episode)

                episode_reward += reward

                #print(episode, reward)

                if done:
                    # if episode_reward < 3 :
                    #     episode_reward += 200
                    #print(episode, episode_reward)
                    break

    def explore(self, env_action_space):
        a = self.env_action_space.sample()
        return a


class Agent_MC:
    def __init__(self, s_dim, a_num, fcs, lr, batch_size):
        self.s_dim = s_dim
        self.a_num = a_num
        self.fcs = fcs
        self.lr = lr
        self.batch_size = batch_size

        self.make_graph()

        self.say_buffer = []

    def make_graph(self):
        s_dim = self.s_dim
        a_num = self.a_num
        lr = self.lr  # 0.0001
        fcs = self.fcs

        with tf.variable_scope('Agent_MC', reuse=tf.AUTO_REUSE):
            # None means batch

            # placeholders
            self.s_input = tf.placeholder(tf.float32, [None, s_dim], name='batch_s')
            self.a_input = tf.placeholder(tf.int32, [None], name='batch_a')
            self.y_input = tf.placeholder(tf.float32, [None], name='batch_y')

            a_input_one_hot = tf.one_hot(self.a_input, a_num)

            self.Q_value = self.make_Q_network(self.s_input, fcs, a_num)
            Q_action = tf.reduce_sum(tf.multiply(self.Q_value, a_input_one_hot), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.y_input - Q_action))
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def make_Q_network(self, input, fcs, a_num):
        out = input
        h_initializer = tf.random_normal_initializer(stddev=np.sqrt(1 / (self.s_dim + 1)))
        o_initializer = tf.orthogonal_initializer(1.0)
        with tf.variable_scope('Q_networks', reuse=tf.AUTO_REUSE):
            for hidden in fcs:
                out = layers.fully_connected(out, hidden, activation_fn=tf.nn.relu, weights_initializer=h_initializer)
            Q_value = layers.fully_connected(out, a_num, activation_fn=None, weights_initializer=o_initializer)

        return Q_value

    def extend_say_buffer(self, s, a, y):
        self.say_buffer.append([s, a, y])

    def shuffle_say_buffer(self):
        random.shuffle(self.say_buffer)

    def clear_say_buffer(self):
        self.say_buffer = []

    def len_say_buffer(self):
        return len(self.say_buffer)

    def len_train_say_buffer(self):
        return int(len(self.say_buffer) * 0.7)

    def len_test_say_buffer(self):
        return int(len(self.say_buffer) * 0.3)

    def get_batch_size(self):
        return self.batch_size

    def train_Q_network(self):
        len_say_buffer = int(len(self.say_buffer) * 0.7)
        batch_size = self.batch_size

        if len_say_buffer < batch_size:
            return

        sess = tf.get_default_session()
        # batch_size = min(max(batch_size, int(len_say_buffer/3)),100000)
        loss = 0
        n_batch = int(len_say_buffer / batch_size)
        # print('len_say_buffer', len_say_buffer)
        # print('batch_size',batch_size)
        for i in range(n_batch):
            left = i * batch_size
            right = (i + 1) * batch_size
            minibatch = self.say_buffer[left:right]
            batch_s = [data[0] for data in minibatch]
            batch_a = [data[1] for data in minibatch]
            batch_y = [data[2] for data in minibatch]
            feed_dict = {
                self.s_input: batch_s,
                self.a_input: batch_a,
                self.y_input: batch_y
            }
            batch_loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            loss += batch_loss
        loss /= n_batch

        return loss

    def test_Q_network(self):
        len_say_buffer = int(len(self.say_buffer) * 0.3)
        batch_size = self.batch_size

        if len_say_buffer < batch_size:
            return

        sess = tf.get_default_session()

        loss = 0
        n_batch = int(len_say_buffer / batch_size)
        for i in range(n_batch):
            left = int(len(self.say_buffer) * 0.7) + i * batch_size
            right = int(len(self.say_buffer) * 0.7) + (i + 1) * batch_size
            minibatch = self.say_buffer[left:right]
            batch_s = [data[0] for data in minibatch]
            batch_a = [data[1] for data in minibatch]
            batch_y = [data[2] for data in minibatch]
            feed_dict = {
                self.s_input: batch_s,
                self.a_input: batch_a,
                self.y_input: batch_y
            }
            batch_loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            loss += batch_loss
        loss /= n_batch

        return loss

    def act(self, s):
        sess = tf.get_default_session()
        feed_dict = {self.s_input: [s]}
        Q_value = sess.run([self.Q_value], feed_dict=feed_dict)[0]
        a = np.argmax(Q_value)
        return a

    def act_egreedy(self, s, env_action_space, epsilon):
        randsed = float(np.random.rand(1))
        if randsed < epsilon:
            a = env_action_space.sample()
        else:
            sess = tf.get_default_session()
            feed_dict = {self.s_input: [s]}
            Q_value = sess.run([self.Q_value], feed_dict=feed_dict)[0]
            a = np.argmax(Q_value)

        return a

    def explore(self, env_action_space):
        a = env_action_space.sample()
        return a

    def rollout(self, env_action_space):
        a = env_action_space.sample()
        return a

    def init_all_var(self):
        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Agent_MC')
        sess = tf.get_default_session()
        sess.run(tf.variables_initializer(var))

    def mpc(self, s, env):
        sess = tf.get_default_session()

        reward_list = [0, 0]
        state_list = [s, s]
        Q_list = [0, 0, 0, 0]
        for i in range(2):
            for j in range(4):
                a1 = env.action_space.sample()
                a2 = env.action_space.sample()

                s1 = state_list[1]
                s2 = state_list[2]
                next_s1 = env.single_transition(s1, a1)
                next_s2 = env.single_transition(s2, a2)

                s = state_list[j]
                feed_dict = {self.s_input: [s]}
                Q_value = sess.run([self.Q_value], feed_dict=feed_dict)[0]
                Q_list.append(Q_value)
                next_s, r, done = env.sigle_transition(s, a)