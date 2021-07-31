import random
import numpy as np
import gin.tf
from hanabi_learning_environment.rl_env import Agent
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import time
from itertools import count
import math
import csv
import pickle

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon parameter for the agent's e-greedy policy.

  Args:
    decay_period: float, the decay period for epsilon.
    step: Integer, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before training starts.
    epsilon: float, the epsilon value.

  Returns:
    A float, the linearly decaying epsilon value.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
  return epsilon + bonus

@gin.configurable
class MLP(nn.Module):
    def __init__(self, n_actions, observation_size, out_size=512):
        super(MLP, self).__init__()
        self.n_actions = n_actions
        self.observation_size = observation_size
        self.out_size = out_size
        # self.embedding_size = embedding_size
        # self.obs_layer1_size = obs_layer1_size
        # self.obs_layer2_size = obs_layer2_size
        # self.embedder = nn.Linear(self.n_actions, embedding_size)
        # self.obs_layer = nn.Linear(observation_size, self.obs_layer1_size)
        # self.obs_layer2 = nn.Linear(self.obs_layer1_size, self.obs_layer2_size)
        # self.lstm = nn.LSTM(input_size = self.obs_layer2_size + self.embedding_size,
        #                     hidden_size = self.hidden_size, batch_first = True)
        self.out_layer = nn.Linear(self.observation_size, self.n_actions)
    
    def forward(self, observation):
        #Takes observations with shape (batch_size, seq_len, obs_dim)
        #Takes one_hot actions with shape (batch_size, seq_len, n_actions)
        # action_embedded = self.embedder(action)
        # observation = F.relu(self.obs_layer(observation))
        # observation = F.relu(self.obs_layer2(observation))
        # lstm_input = torch.cat([observation, action_embedded], dim = -1)
        # if hidden is not None:
        #     lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        # else:
        #     lstm_out, hidden_out = self.lstm(lstm_input)
        q_values = self.out_layer(observation)
        return q_values
    
    def act(self, observation, legal_actions, epsilon):
        q_values_tensor = self.forward(observation)
        q_values_np = q_values_tensor[0][0].cpu().detach().numpy()
        q_values = []
        legal_action_indices = np.where(legal_actions == 0.0)[0]
        for idx in range(len(legal_actions)):
          if idx in legal_action_indices:
            q_values.append(q_values_np[idx])
          else:
            q_values.append(-np.Inf)
        # print("q_values: ", q_values)
        if np.random.uniform() > epsilon:
        #   print("max_q_value")
          action = np.nanargmax(q_values)
        else:
        #   print("random")
          action = np.random.choice(legal_action_indices)
        return action

@gin.configurable
class ExpBuffer():
    def __init__(self, max_storage, sample_length):
        self.max_storage = max_storage
        self.sample_length = sample_length
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, aoarod):
        if self.counter < self.max_storage-1:
            self.counter +=1
        if self.filled < self.max_storage:
            self.filled += 1
        else:
            self.counter = 0
        self.storage[self.counter] = aoarod
    
    def sample(self, batch_size):
        #Returns sizes of (batch_size, seq_len, *) depending on action/observation/return/done
        seq_len = self.sample_length
        last_actions = []
        last_observations = []
        actions = []
        rewards = []
        observations = []
        dones = []

        for i in range(batch_size):
            if self.filled - seq_len < 0 :
                raise Exception("Reduce seq_len or increase exploration at start.")
            start_idx = np.random.randint(self.filled-seq_len)
            # print("filled: ", self.filled)
            # print("start_idx: ", start_idx)
            last_act, last_obs, act, rew, obs, done = zip(*self.storage[start_idx:start_idx+seq_len])
            last_actions.append(list(last_act))
            last_observations.append(last_obs)
            actions.append(list(act))
            rewards.append(list(rew))
            observations.append((obs))
            dones.append(list(done))
            # print("act: ", act)
            # print("last_act: ", last_act)
            # if act == -1:
            #     print("yo")
            # if last_act == -1:
            #     print("yo")
            # print("rewards", rewards)
        # print("last_actions shape: ", len(last_actions), len(last_actions[0]))
        # print("last_observations: ", last_observations)
        # print("actions: ", actions)
        # print("rewards: ", rewards)
        # print("observations: ", observations)
        # print("last_observations[0]: ", last_observations[0])
        # print("observations[0]: ", observations[0])
        # print("dones: ", dones)
           
        return torch.tensor(last_actions).cuda(),\
               torch.tensor(last_observations, dtype = torch.float32).cuda(),\
               torch.tensor(actions).cuda(),\
               torch.tensor(rewards).float().cuda(),\
               torch.tensor(observations, dtype = torch.float32).cuda(),\
               torch.tensor(dones).cuda()

@gin.configurable
class MLPAgent(Agent):

  @gin.configurable
  def __init__(self,
               num_actions=None,
               observation_size=None,
               num_players=None,
               embedding_size = 8,
               replay_buffer_size=100000,
               sample_length=20,
               batch_size = 64,
               epsilon_train = 0.02,
               epsilon_eval = 0.001,
               epsilon_decay_period = 1000,
               gamma = 0.999,
               learning_rate = 0.01,
               explore = 300,
               last_action = {}, # debug
               last_observation = {}, # debug
               begin = 1, # debug
               i_episode = 0, # debug
               training_steps = 0, # debug
               update_period = 4, # debug
               target_update_period = 500, # debug
               loss = 0, # debug
               epsilon_fn=linearly_decaying_epsilon,
               tf_device='/cpu:*',):

    # Global variables.
    self.num_actions = num_actions
    self.observation_size = observation_size
    self.num_players = num_players
    self.embedding_size = embedding_size
    self.replay_buffer_size = replay_buffer_size
    self.sample_length = sample_length
    self.batch_size = batch_size
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.explore = explore

    self.last_action = last_action # debug
    self.last_observation = last_observation # debug
    self.begin = begin # debug
    self.i_episode = i_episode # debug
    self.training_steps = training_steps # debug
    self.update_period = update_period # debug
    self.target_update_period = target_update_period # debug
    self.loss = loss
    self.epsilon_fn = epsilon_fn
    self.eval_mode = False
    self.q_values_list = []
    self.target_values_list = []
    self.raw_data = [["training_step", "example_reward", "example_done", "avg_q_value", "avg_target_value", 
                      "loss", "out_layer_weight_NORM", "out_layer_bias_NORM"]]

    print("num_actions: ", num_actions)
    print("observation_size: ", observation_size)
    print("num_players: ", num_players)

    # MLP and ExpBuffer instances
    self.replay_buffer = ExpBuffer(self.replay_buffer_size, self.sample_length)
    self.mlp = MLP(self.num_actions, self.observation_size).cuda()
    self.mlp_target = MLP(self.num_actions, self.observation_size).cuda()
    self.mlp_target.load_state_dict(self.mlp.state_dict())

    # Optimizer
    self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr = learning_rate)

  def begin_episode(self, current_player, legal_actions, observation):

    # print("begin_episode")
    # print("current_player: ", current_player)
    # print("i_episode: ", self.i_episode)
    # print("observation: ", torch.tensor(observation).float().view(1,1,-1))
    self.begin = 1
    self.last_action[current_player] = 0

    # print("one_hot: ", F.one_hot(torch.tensor(self.last_action[current_player]), self.num_actions).view(1,1,-1).float())

    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.explore, self.epsilon_train)

    action = self.mlp.act(
      torch.tensor(observation).float().view(1,1,-1).cuda(),
      legal_actions,
      epsilon = epsilon)

    # print("action: ", action)
    self.last_action[current_player] = action
    self.last_observation[current_player] = observation

    return action

  def step(self, reward, current_player, legal_actions, observation):

    # print("step")
    # print("current_player: ", current_player)
    # print("i_episode: ", self.i_episode)

    done = False
    if self.begin == 1:
      self.replay_buffer.write_tuple((
        0, 
        self.last_observation[current_player], 
        self.last_action[current_player],
        reward,
        observation,
        done
        ))
      self.begin = 0
      if self.i_episode > self.explore:
        # print("here 1")
        self.loss = self._update_network()

    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.explore, self.epsilon_train)

    action = self.mlp.act(
      torch.tensor(observation).float().view(1,1,-1).cuda(),
      legal_actions,
      epsilon = epsilon)

    if self.begin == 0:
      self.replay_buffer.write_tuple((
        self.last_action[current_player], 
        self.last_observation[current_player], 
        action,
        reward,
        observation,
        done
        ))
      if self.i_episode > self.explore:
        # print("here 2")
        self.loss = self._update_network()

      self.last_action[current_player] = action
      self.last_observation[current_player] = observation

    return action, self.loss

  def end_episode(self, final_rewards, player):

    print("end_episode")
    print("i_episode: ", self.i_episode)
    print("training_steps:", self.training_steps)

    done = True
    for current_player in range(self.num_players):
      self.replay_buffer.write_tuple((
        self.last_action[current_player], 
        self.last_observation[current_player], 
        0,
        final_rewards[current_player],
        np.array([0]*len(self.last_observation[current_player])), # debug
        done
        ))
    self.i_episode += 1

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    return None

  def _update_network(self):

    if self.eval_mode:
      return 0

    if self.training_steps % self.update_period == 0:

      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                 self.explore, self.epsilon_train)

      last_actions, last_observations, actions, rewards, observations, dones = self.replay_buffer.sample(self.batch_size)
      q_values = self.mlp.forward(last_observations).float()

      q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
      # q_values = torch.gather(q_values.type(torch.LongTensor), -1, actions.type(torch.LongTensor).unsqueeze(-1)).squeeze(-1).type(torch.LongTensor)
      # print("here")
      predicted_q_values = self.mlp.forward(observations).float()
      target_values = rewards + (self.gamma * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

      #Update network parameters
      self.optimizer.zero_grad()
      loss = torch.nn.MSELoss()(q_values , target_values.detach())
      loss.backward()
      self.optimizer.step()

      self.loss = loss

      weight, bias = None, None
      for name, param in self.mlp.named_parameters():
        if name == "out_layer.weight":
          weight = param.grad.norm().item()
        if name == "out_layer.bias":
          bias = param.grad.norm().item()

    if self.training_steps % self.target_update_period == 0:

      self.q_values_list.append(q_values)
      pickle.dump(self.q_values_list, open('q_values_list.pkl', 'wb' ))
      self.target_values_list.append(target_values)
      pickle.dump(self.target_values_list, open('target_values_list.pkl', 'wb' ))

      lst = [self.training_steps, rewards[0][0].item(), dones[0][0].item(), 
             torch.mean(torch.mean(q_values, 1), 0).item(),torch.mean(torch.mean(target_values, 1), 0).item(),
             loss.item(), weight, bias]
      self.raw_data.append(lst)
      with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(self.raw_data)

      self.mlp.load_state_dict(self.mlp.state_dict())
      print("eps: ", epsilon)

    self.training_steps += 1

    return self.loss
