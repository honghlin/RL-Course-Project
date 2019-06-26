# This file contains the parameters.

# train env
train_env_fcs = [64, 64]
train_env_grad_clip = 50
train_env_lr = 0.001

# train agent
train_agent_fcs = [64, 64]
train_agent_lr = 0.001

# trainer --- run 1
gamma = 0.99
num_iterations = 1000
num_samples_per_iteration = 1000 # fixed, determine the number of training data for training environment
method = 'Policy' # method options: Policy, Batch, DAgger, DAgger_MC, H_DAgger_MC, Transit_Aware DAgger Batch Policy

# trainer --- train env
num_epoches_in_train_env = 10000 # 1000 is about 1min
train_env_batch_size = 500

# trainer -- train policy
num_episodes_in_train_policy = 1500 # the number of training data is average_len_episode × num_episodes_in_train_policy
num_epoches_in_train_policy = 50 # determine the times to train Q network
train_agent_batch_size = 300 # train_agent_batch_size < num_episodes_in_train_policy

# trainer --- test policy
num_episodes_in_test_policy = 100

