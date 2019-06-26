import numpy as np
import pickle
import time
import gym
import random

def pickle_save(object, file_path):
    f = open(file_path, 'wb')
    pickle.dump(object, f)

class Trainer:
    def __init__(self, agent, real_env, simu_env, method, gamma, num_iterations, num_samples_per_iteration,
                 num_epoches_in_train_env, train_env_batch_size, num_episodes_in_train_policy,
                 num_epoches_in_train_policy, num_episodes_in_test_policy):
        self.agent = agent # agent is implicit
        self.real_env = real_env
        self.simu_env = simu_env
        self.method = method
        self.gamma = gamma

        # run
        self.num_iterations = num_iterations
        self.num_samples_per_iteration = num_samples_per_iteration

        # train env
        self.num_epoches_in_train_env = num_epoches_in_train_env
        self.train_env_batch_size = train_env_batch_size

        # train policy
        self.num_episodes_in_train_policy = num_episodes_in_train_policy
        self.num_epoches_in_train_policy = num_epoches_in_train_policy

        # test policy
        self.num_episodes_in_test_policy = num_episodes_in_test_policy

        # show and save
        self.cur_time = time.localtime(time.time())
        self.test_average_r_list = []

    def run(self):
        method = self.method
        real_env = self.real_env
        simu_env = self.simu_env
        num_iterations = self.num_iterations # 1000
        num_samples_per_iteration = self.num_samples_per_iteration # 10000
        sas_buffer = [] # dataset of (s, a, s')
        test_cos_loss = 1

        sas_buffer = []
        for n in range(num_iterations):

            for k in range(num_samples_per_iteration):
                s, a, next_s = self.interaction(method, real_env, simu_env, test_cos_loss)
                sas_buffer.append([s, a, next_s])
            simu_env, test_cos_loss = self.train_env(simu_env, sas_buffer)
            self.train_policy(simu_env)##1 real_env
            # self.train_policy(real_env)
            test_average_r = round(self.test_policy(real_env))

            self.show_and_save(method, n, test_average_r)

    def Policy(self, real_env):

        s = real_env.state
        a = self.agent.act(s)
        next_s, r, done = real_env.step(a)

        return s, a, next_s

    def D_Policy(self, real_env):

        #to get (s, a) ~ D_{poicy}
        gamma = self.gamma

        s = real_env.reset()
        accept_prob = 1
        while True:
            a = self.agent.act(s)
            next_s, r, done = real_env.step(a)

            accept_prob = gamma
            randsed = float(np.random.rand(1))
            if randsed < accept_prob or done: # accept
                break
            s = next_s

        return s, a, next_s

    def Batch(self, real_env):

        s = real_env.state
        a = np.random.choice([0, 1])
        next_s, r, done = real_env.step(a)
        return s, a, next_s

    def D_Batch(self, real_env):

        #to get (s, a) ~ v
        gamma = self.gamma

        s = real_env.reset()
        accept_prob = 1
        while True:
            a = np.random.choice([0, 1])
            next_s, r, done = real_env.step(a)

            accept_prob = gamma
            randsed = float(np.random.rand(1))
            if randsed < accept_prob or done: # accept
                break
            s = next_s
        return s, a, next_s


    def DAgger(self, real_env):
        # to get (s, a) ~ 0.5*v + 0.5*D_{poicy}

        randsed = float(np.random.rand(1))
        if randsed < 0.5:
            s, a, next_s = self.D_Policy(real_env)
        else:
            s, a, next_s = self.D_Batch(real_env)

        return s, a, next_s

    def DAgger_MC(self, real_env):
        # to get (s, a) ~ delta

        gamma = self.gamma

        randsed_init = float(np.random.rand(1))
        if randsed_init < 0.5:
            z, b, y = self.D_Policy(real_env)
        elif randsed_init >= 0.5 and randsed_init < 0.75:
            z, b, y = self.D_Batch(real_env)
        elif randsed_init >= 0.75 and randsed_init < (0.75 + 0.25 * (1 - gamma)):
            z = real_env.reset()
            b = self.agent.act(z)
            state, reward, done = real_env.step(b)
            y = state
        else:
            x, c, z = self.D_Batch(real_env)
            b = self.agent.act(z)
            state, reward, done = real_env.step(b)
            y = state

        s = y
        accpet_prob = 1
        while True:
            a = self.agent.rollout(real_env.action_space)
            next_s, r, done = real_env.step(a)

            accpet_prob = gamma
            randsed = float(np.random.rand(1))
            if randsed < accpet_prob or done:
                break
            s = next_s

        return s, a, next_s

    def H_DAgger_MC(self, real_env, simu_env): # do not use H_DAgger_MC now
        # to get (s, a) ~ h

        gamma = self.gamma
        T = 200 # rollout depth
        D = [] # D_{n,k,1:T-1}

        randsed_init = float(np.random.rand(1))
        if randsed_init < 0.5:
            x, b, next_s = self.Policy(real_env)
        elif randsed_init >= 0.5 and randsed_init < 0.75:
            x, b, next_s = self.Batch(real_env)
        elif randsed_init >= 0.75 and randsed_init < (0.75 + 0.25 * (1 - gamma)):
            x = real_env.reset()
            b = self.agent.act(z)
            state, reward, done, info = real_env.step(b)
            next_s = state.tolist()
        else:
            y, c, x = self.Batch(real_env)
            b = self.agent.act(x)
            state, reward, done, info = real_env.step(b)
            next_s = state.tolist()
        s = x
        z = x
        a = b
        for t in range(T):
            D.append([z, a, next_s])
            next_z = simu_env.single_transition(z, a)
            s = next_s
            z = next_z
            a = self.agent.explore(real_env)
            state, reward, done, info = real_env.step(a)
            next_s = state.tolist()
            if done: # there may be some problems
                next_s = real_env.reset()

        return D

    def Transit_Aware(self, real_env, test_cos_loss):
        # to get (s, a) ~ cos*D_{policy} + (1-cos)*v

        randsed = float(np.random.rand(1))
        if randsed < test_cos_loss:
            s, a, next_s = self.Policy(real_env)
        else:
            s, a, next_s = self.Batch(real_env)

        return s, a, next_s

    def interaction(self, method, real_env, simu_env, test_cos_loss):
        if method == 'Policy':
            s, a, next_s = self.Policy(real_env)
        elif method == 'Batch':
            s, a, next_s = self.Batch(real_env)
        elif method == 'DAgger':
            s, a, next_s = self.DAgger(real_env)
        elif method == 'DAgger_MC':
            s, a, next_s = self.DAgger_MC(real_env)
        elif method == 'H_DAgger_MC':
            D = self.H_DAgger_MC(real_env, simu_env)
        elif method == 'Transit_Aware':
            s, a, next_s = self.Transit_Aware(real_env, test_cos_loss)

        return s, [a], next_s

    def train_env(self, simu_env, sas_buffer):

        num_epoches = self.num_epoches_in_train_env  # 5000
        batch_size = self.train_env_batch_size  # 5000
        ratio = 0.7

        len_sas = int(len(sas_buffer))
        random.shuffle(sas_buffer)

        len_train = int(len_sas * ratio)
        train_sas_buffer = sas_buffer[0:len_train]
        test_sas_buffer = sas_buffer[len_train:]

        print("start train env \n",
              "len of sas_buffer:", len(sas_buffer), "\n",
              "len of train_sas_buffer:", len(train_sas_buffer), "\n",
              "len of test_sas_buffer:", len(test_sas_buffer), "\n",
              "train_env_batch_size:", batch_size, "\n",
              "max epoches:", num_epoches)

        num_overfit = 0  # 记录过拟合的次数，大于10次就停止
        # train env
        for epoch in range(num_epoches):
            times = int(len(train_sas_buffer) / batch_size)
            train_env_loss = 0
            for j in range(times):
                left = j * batch_size
                right = ((j + 1) * batch_size)
                if right < len(train_sas_buffer):
                    train_batch_data = train_sas_buffer[left:right]
                else:
                    train_batch_data = train_sas_buffer[left:] + train_sas_buffer[0: right - len(train_sas_buffer)]
                batch_s, batch_a, batch_next_s = list(map(list, zip(*train_batch_data)))
                batch_a = np.squeeze(batch_a)
                tmp_train_env_loss = simu_env.train(batch_s, batch_a, batch_next_s)
                train_env_loss += tmp_train_env_loss
            train_env_loss /= times

            # test env
            batch_s, batch_a, batch_next_s = list(map(list, zip(*test_sas_buffer)))
            batch_a = np.squeeze(batch_a)
            test_env_loss = simu_env.test(batch_s, batch_a, batch_next_s)
            test_cos_loss = simu_env.get_cos_loss(batch_s, batch_a, batch_next_s)

            print("epoch:", epoch, "train_env_loss", train_env_loss, "test_env_loss", test_env_loss, "test_cos_loss",
                  test_cos_loss)

            if num_overfit > 10:
                print("train env finished --- ", "real epoch:", epoch, "train_env_loss", train_env_loss,
                      "test_env_loss", test_env_loss, "test_cos_loss", test_cos_loss)
                break
            if train_env_loss < test_env_loss:
                num_overfit += 1

            # 重新打乱数据，设置训练和验证集
            random.shuffle(sas_buffer)
            len_train = int(len_sas * ratio)
            train_sas_buffer = sas_buffer[0:len_train]
            test_sas_buffer = sas_buffer[len_train:]

        return simu_env, test_cos_loss

        return simu_env, test_cos_loss

    def train_policy(self, simu_env):

        if(self.method == 'DAgger_MC'): #

            self.agent.clear_say_buffer()
            # 需要提高的地方：epoch不需要这么多
            # 在done为false的情况下，episode reward需要加上值函数
            # epsilon最好为decay的形式
            gamma = self.gamma
            num_episodes = self.num_episodes_in_train_policy  # 100000
            num_epoches = self.num_epoches_in_train_policy  # 5000

            for episode in range(num_episodes):
                sar_buffer = []
                init_s = simu_env.reset()
                init_a = self.agent.act(init_s)
                s = init_s
                a = init_a
                epsilon = 1  # epsilon初始值
                while True:
                    epsilon *= 1  # decay速率
                    next_s, r, done = simu_env.step(a)
                    # next_s, r, done = simu_env.step(a)

                    sar_buffer.append([s, a, r])
                    if done:
                        break
                    s = next_s
                    a = self.agent.act_egreedy(s, simu_env.action_space, epsilon)

                coef = 1
                episode_reward = 0
                len_sar = len(sar_buffer)
                for i in range(len_sar):
                    s, a, r = sar_buffer[len_sar - 1 - i]
                    # episode_reward = episode_reward * gamma + r
                    episode_reward += r
                    self.agent.extend_say_buffer(s, a, episode_reward)

            len_say_buffer = self.agent.len_say_buffer()
            len_train_say_buffer = self.agent.len_train_say_buffer()
            len_test_say_buffer = self.agent.len_test_say_buffer()
            batch_size = self.agent.get_batch_size()
            print("start train policy \n",
                  "num_episodes:", num_episodes, "\n",
                  "len of sas_buffer:", len_say_buffer, "\n",
                  "len of train_say_buffer:", len_train_say_buffer, "\n",
                  "len of valid_say_buffer:", len_test_say_buffer, "\n",
                  "batch size:", batch_size, "\n",
                  "max epoches:", num_epoches)

            num_overfit = 0  # 训练Q函数过拟合的次数，次数大于10就停止训练
            for epoch in range(num_epoches):
                self.agent.shuffle_say_buffer()
                train_loss = self.agent.train_Q_network()
                test_loss = self.agent.test_Q_network()

                print("epoch:", epoch, "train_Q_loss", train_loss, "test_Q_loss", test_loss)
                if num_overfit > 10:
                    print("train Q finished --- ", "real epoch:", epoch, "train_Q_loss", train_loss,
                          "test_Q_loss", test_loss)
                    break
                if train_loss < test_loss:
                    num_overfit += 1

        else :
            self.agent.train(simu_env)



    def test_policy(self, real_env):
        gamma = self.gamma
        gamma = 1
        num_episodes = self.num_episodes_in_test_policy # 1000

        total_reward = 0
        for episode in range(num_episodes):
            coef = 1
            episode_reward = 0
            init_s = real_env.reset()
            init_a = self.agent.act(init_s)
            s = init_s
            a = init_a
            while True:
                next_s, r, done = real_env.step(a, test=True)
                # episode_reward += coef * r
                episode_reward += r
                if done:
                    break
                coef *= gamma
                s = next_s
                a = self.agent.act(s)
            total_reward += episode_reward
            #print(episode, episode_reward)
        average_r = total_reward / num_episodes

        return average_r

    def show_and_save(self, method, n, test_average_r):
        self.test_average_r_list.append([n, test_average_r])
        cur_time = self.cur_time

        time_info = str(cur_time.tm_year) + str(cur_time.tm_mon) + str(cur_time.tm_mday) + \
                    str(cur_time.tm_hour) + str(cur_time.tm_min) + str(cur_time.tm_sec)

        pickle_save(self.test_average_r_list, "results/" + method + "_" + time_info + "_test_r")

        print("\n iterartion:", n, "\n method:", method, "\n test result: ", self.test_average_r_list)