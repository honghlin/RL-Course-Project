import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import math
from gym import spaces

# This simulator simulates the environment with continuous state space and discrete action space.
class Simu_Env:
    def __init__(self, s_dim, a_num, fcs, grad_clip, lr):
        self.s_dim = s_dim
        self.a_num = a_num
        self.fcs = fcs
        self.grad_clip = grad_clip
        self.lr = lr

        self.make_graph()

    def make_graph(self):
        s_dim = self.s_dim
        a_num = self.a_num
        fcs = self.fcs
        grad_clip = self.grad_clip # 0.5
        lr = self.lr # 0.0001

        with tf.variable_scope('Simu_Env', reuse=tf.AUTO_REUSE):
            # None and -1 means batch size

            # placeholders
            self.s_input = tf.placeholder(tf.float32, [None, s_dim], name='batch_s')
            self.a_input = tf.placeholder(tf.int32, [None], name='batch_a')
            self.next_s_input = tf.placeholder(tf.float32, [None, s_dim], name='batch_next_s')

            a_input_one_hot = tf.one_hot(self.a_input, a_num)
            input = tf.concat([self.s_input, a_input_one_hot], axis=1)

            # variables
            self.next_s = self.make_transition_network(input, fcs, s_dim)
            self.env_loss = tf.reduce_mean(tf.square(self.next_s - self.next_s_input)) # calculate the loss
            simu_env_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Simu_Env')  # get vars
            gradients = tf.gradients(self.env_loss, simu_env_vars)  # get gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)  # clip gradients
            grads_and_vars = zip(clipped_gradients, simu_env_vars)  # make a zip
            optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-5)  # get optimizer
            self.optimizer = optimizer.apply_gradients(grads_and_vars)  # apply gradients to optimizer

            # cosine loss of transition
            average_next_s_input = tf.reduce_mean(self.next_s_input, axis=0)
            bn_next_s_input = self.next_s_input - average_next_s_input
            next_s_input_norm = tf.sqrt(tf.reduce_sum(tf.square(bn_next_s_input), axis=1))

            average_next_s = tf.reduce_mean(self.next_s, axis=0)
            bn_next_s = self.next_s - average_next_s
            next_s_norm = tf.sqrt(tf.reduce_sum(tf.square(bn_next_s), axis=1))

            dot_multi = tf.multiply(bn_next_s_input, bn_next_s)
            dot_sum = tf.reduce_sum(dot_multi, axis=1)
            div = next_s_input_norm * next_s_norm
            self.cos_loss = tf.reduce_mean(dot_sum / div )

    def make_transition_network(self, input, fcs, s_dim):
        out = input
        h_initializer = tf.random_normal_initializer(stddev=np.sqrt(1 / (self.s_dim + 1)))
        o_initializer = tf.orthogonal_initializer(1.0)
        with tf.variable_scope('hiddens', reuse=tf.AUTO_REUSE):
            for hidden in fcs:
                out = layers.fully_connected(out, hidden, activation_fn=tf.nn.relu, weights_initializer=h_initializer)
            next_s = layers.fully_connected(out, s_dim, activation_fn=None, weights_initializer=o_initializer)

        return next_s

    def train(self, batch_s, batch_a, batch_next_s):
        feed_dict = {
            self.s_input: batch_s,
            self.a_input: batch_a,
            self.next_s_input: batch_next_s
        }
        sess = tf.get_default_session()
        env_loss, _ = sess.run([self.env_loss, self.optimizer], feed_dict=feed_dict)
        return env_loss

    def test(self, batch_s, bathc_a, batch_next_s):
        feed_dict = {
            self.s_input: batch_s,
            self.a_input: bathc_a,
            self.next_s_input: batch_next_s
        }
        sess = tf.get_default_session()
        env_loss= sess.run(self.env_loss, feed_dict=feed_dict)
        return env_loss

    def batch_transition(self, batch_s, bathc_a):
        feed_dict = {
            self.s_input: batch_s,
            self.a_input: bathc_a,
        }
        sess = tf.get_default_session()
        next_s = sess.run(self.next_s, feed_dict=feed_dict)
        return next_s

    def single_transition(self, s, a):
        feed_dict = {
            self.s_input: [s],
            self.a_input: [a],
        }
        sess = tf.get_default_session()
        next_s = sess.run(self.next_s, feed_dict=feed_dict)[0]
        return next_s

    def get_cos_loss(self, batch_s, batch_a, batch_next_s):
        feed_dict = {
            self.s_input: batch_s,
            self.a_input: batch_a,
            self.next_s_input: batch_next_s
        }
        sess = tf.get_default_session()
        cos_loss = sess.run(self.cos_loss, feed_dict=feed_dict)
        return cos_loss

# This class gives the real reset state distribution
# and can give true done or false done.
class Simu_CartPole_v0(Simu_Env):
    def __init__(self, s_dim, a_num, fcs, grad_clip, lr):
        super().__init__(s_dim, a_num, fcs, grad_clip, lr)
        self.reset()

        self.time = 0
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.time = 0
        return np.array(self.state).tolist()

    def step(self, action):
        self.time += 1
        s = self.state
        a = action
        simulated_state = self.single_transition(s, a)
        self.state = simulated_state

        x, x_dot, theta, theta_dot = simulated_state
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        reward = 1.0

        if self.time >= 200:
            done = True

        if done:
            # print(self.time)
            self.reset()

        if done:
            reward = -200

        return np.array(self.state).tolist(), reward, done

    def step_with_real_done(self, action):
        assert 1 == 0
        s = self.state
        x, x_dot, theta, theta_dot = s

        self.time += 1
        a = action
        simulated_state = self.single_transition(s, a)
        self.state = simulated_state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        reward = 1.0

        if done:
            reward = -200

        if self.time >= 200 :
            done = True

        if done:
            #print(self.time)
            self.reset()

        return np.array(self.state).tolist(), reward, done

# This class gives the real reset state distribution
# and can give true done or false done.
# Besides, this class gives the real reward function of CartPole-v1.
class Simu_CartPole_v1(Simu_Env):
    def __init__(self, s_dim, a_num, fcs, grad_clip, lr):
        super().__init__(s_dim, a_num, fcs, grad_clip, lr)
        self.reset()

        self.time = 0
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.time = 0
        return np.array(self.state).tolist()

    def step(self, action):
        self.time += 1
        s = self.state
        a = action
        simulated_state = self.single_transition(s, a)
        self.state = simulated_state

        x, x_dot, theta, theta_dot = simulated_state
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if self.time >= 100:
            done = True

        reward = 1.0



        if done:
            self.reset()

        tmp_x = np.array(self.state).tolist()[0]
        if tmp_x >= -0.5 and tmp_x <= 0.5:
            reward = 2
        elif tmp_x >= -1 and tmp_x <= 1:
            reward = 1
        else:
            reward = 0

        return np.array(self.state).tolist(), reward, done

    def step_with_real_done(self, action):
        s = self.state
        x, x_dot, theta, theta_dot = s

        self.time += 1
        a = action
        simulated_state = self.single_transition(s, a)
        self.state = simulated_state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        # if done:
        #     reward = -200

        reward = 1.0

        if self.time >= 100 :
            done = True

        if done:
            self.reset()

        tmp_x = np.array(self.state).tolist()[0]
        if tmp_x >= -0.5 and tmp_x <= 0.5:
            reward = 2
        elif tmp_x >= -1 and tmp_x <= 1:
            reward = 1
        else:
            reward = 0

        return np.array(self.state).tolist(), reward, done

