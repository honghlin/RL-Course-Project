import numpy as np
import tensorflow as tf

import parameters
from trainer import Trainer
from real_env import CartPole_v1
from simu_env import Simu_CartPole_v1
from real_env import CartPole_v0
from simu_env import Simu_CartPole_v0
from agent import Agent_MC
from agent import Agent_Q

def main():
    params = parameters

    sess = tf.Session()
    sess.__enter__()

    real_env = CartPole_v0()
    s_dim = real_env.state_dim
    a_num = real_env.action_num

    if(params.method == 'DAgger_MC') : #

        agent = Agent_MC(
            s_dim=s_dim,
            a_num=a_num,
            fcs=params.train_agent_fcs,
            lr=params.train_agent_lr,
            batch_size=params.train_agent_batch_size)

    else :
        agent = Agent_Q(env = real_env)

    simu_env = Simu_CartPole_v0(
        s_dim=s_dim,
        a_num=a_num,
        fcs=params.train_env_fcs,
        grad_clip=params.train_env_grad_clip,
        lr=params.train_env_lr
    )

    # simu_env1 = CartPole_v0()

    sess.run(tf.global_variables_initializer())

    trainer = Trainer(
        agent=agent,
        real_env=real_env,
        simu_env=simu_env,
        method=params.method,
        gamma=params.gamma,
        num_iterations=params.num_iterations,
        num_samples_per_iteration=params.num_samples_per_iteration,
        num_epoches_in_train_env=params.num_epoches_in_train_env,
        train_env_batch_size=params.train_env_batch_size,
        num_episodes_in_train_policy=params.num_episodes_in_train_policy,
        num_epoches_in_train_policy=params.num_epoches_in_train_policy,
        num_episodes_in_test_policy=params.num_episodes_in_test_policy
    )
    trainer.run()

if __name__ == '__main__':
    main()