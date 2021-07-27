
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

@gin.configurable
class ADRQN(nn.Module):
    def __init__(self, num_actions, observation_size):
        super(ADRQN, self).__init__()
        self.num_actions = num_actions
        # self.embedding_size = embedding_size
        # self.embedder = nn.Linear(n_actions, embedding_size)
        # self.obs_layer = nn.Linear(state_size, 16)
        # self.obs_layer2 = nn.Linear(16,32)
        self.lstm = nn.LSTM(input_size = observation_size+num_actions, hidden_size = 512, batch_first = True)
        self.out_layer = nn.Linear(512, num_actions)
    
    def forward(self, observation, action, hidden = None):
        #Takes observations with shape (batch_size, seq_len, obs_dim)
        #Takes one_hot actions with shape (batch_size, seq_len, n_actions)
        # action_embedded = self.embedder(action)
        # observation = F.relu(self.obs_layer(observation))
        # observation = F.relu(self.obs_layer2(observation))
        lstm_input = torch.cat([observation, action], dim = -1)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_out = self.lstm(lstm_input)

        q_values = self.out_layer(lstm_out)
        return q_values, hidden_out
    
    def act(self, observation, last_action, legal_actions, epsilon, hidden = None):
        q_values_tensor, hidden_out = self.forward(observation, last_action, hidden)
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
        # print("legal_actions: ", legal_actions)
        # print("action: ", action)
        # print("action: ", action)
        # if action == -1:
        #     print("legal_actions: ", legal_actions)
        #     print("legal_action_indices: ", legal_action_indices)
        return action, hidden_out

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
class AdrqnAgent(Agent):

  @gin.configurable
  def __init__(self,
               num_actions=None,
               observation_size=None,
               num_players=None,
               replay_buffer_size=100000,
               sample_length=20,
               batch_size = 64,
               eps_start = 0.9,
               eps_end = 0.05,
               eps_decay = 10,
               gamma = 0.999,
               learning_rate = 0.01,
               explore = 300,
               hidden = None, # debug
               last_action = {}, # debug
               last_observation = {}, # debug
               begin = 1, # debug
               i_episode = 0, # debug
               training_steps = 0, # debug
               update_period = 4, # debug
               target_update_period = 500, # debug
               tf_device='/cpu:*',):

    # Global variables.
    self.num_actions = num_actions
    self.observation_size = observation_size
    self.num_players = num_players
    self.replay_buffer_size = replay_buffer_size
    self.sample_length = sample_length
    self.batch_size = batch_size
    self.eps_start = eps_start
    self.eps = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.explore = explore

    self.hidden = hidden # debug
    self.last_action = last_action # debug
    self.last_observation = last_observation # debug
    self.begin = begin # debug
    self.i_episode = i_episode # debug
    self.training_steps = training_steps # debug
    self.update_period = update_period # debug
    self.target_update_period = target_update_period # debug

    # ADRQN and ExpBuffer instances
    self.replay_buffer = ExpBuffer(self.replay_buffer_size, self.sample_length)
    self.adrqn = ADRQN(self.num_actions, observation_size).cuda()
    self.adrqn_target = ADRQN(self.num_actions, observation_size).cuda()
    self.adrqn_target.load_state_dict(self.adrqn.state_dict())

    # Optimizer
    self.optimizer = torch.optim.Adam(self.adrqn.parameters(), lr = learning_rate)

  def begin_episode(self, current_player, legal_actions, observation):

    # print("begin_episode")
    # print("current_player: ", current_player)
    # print("i_episode: ", self.i_episode)
    # print("observation: ", torch.tensor(observation).float().view(1,1,-1))
    self.begin = 1
    self.last_action[current_player] = 0

    # print("one_hot: ", F.one_hot(torch.tensor(self.last_action[current_player]), self.num_actions).view(1,1,-1).float())

    action, self.hidden = self.adrqn.act(
      torch.tensor(observation).float().view(1,1,-1).cuda(),
      F.one_hot(torch.tensor(self.last_action[current_player]), self.num_actions).view(1,1,-1).float().cuda(),
      legal_actions,
      hidden = self.hidden,
      epsilon = self.eps)

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
        self._update_network()

    action, self.hidden = self.adrqn.act(
      torch.tensor(observation).float().view(1,1,-1).cuda(),
      F.one_hot(torch.tensor(self.last_action[current_player]), self.num_actions).view(1,1,-1).float().cuda(),
      legal_actions,
      hidden = self.hidden,
      epsilon = self.eps)

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
        self._update_network()

      self.last_action[current_player] = action
      self.last_observation[current_player] = observation

    return action

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

    if self.training_steps % self.update_period == 0:

      self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp((-1*(self.i_episode-self.explore))/self.eps_decay)

      last_actions, last_observations, actions, rewards, observations, dones = self.replay_buffer.sample(self.batch_size)
      q_values, _ = self.adrqn.forward(last_observations, F.one_hot(last_actions, self.num_actions).float())
      # print("q_values: ", q_values)
      # print("q_values shape: ", q_values.shape)
      # print("last_observations shape: ", last_observations.shape)
      # print("last_actions: ", last_actions)
      # print("last_actions shape: ", last_actions.shape)
      # print("actions shape: ", actions.shape)
      # print("actions unsqueezed shape: ", actions.unsqueeze(-1).shape)
      # print("q_values data type: ", q_values.dtype)
      # print("q_values tensor type: ", q_values.type)
      # q_values = q_values.type(torch.LongTensor)
      # actions = actions.type(torch.LongTensor)
      # print("q_values data type 2: ", q_values.dtype)
      # print("q_values tensor type 2: ", q_values.type)
      # print("q_values data type: ", q_values.dtype)fl
      # q_values = q_values.long()
      # print("q_values data type: ", q_values.dtype)
      # print("torch version: ", torch.__version__) 
      # print("actions: ", actions)
      # print("actions shape: ", actions.shape)
      # print("actions[0]: ", actions[0])
      # print("actions[1]: ", actions[1])
      # print("actions[2]: ", actions[2])
      # mx = 0
      # for e in actions:
      #     if max(e) > mx:
      #         mx = max(e)
      # print("max: ", mx)
      q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
      # q_values = torch.gather(q_values.type(torch.LongTensor), -1, actions.type(torch.LongTensor).unsqueeze(-1)).squeeze(-1).type(torch.LongTensor)
      # print("here")
      predicted_q_values, _ = self.adrqn_target.forward(observations, F.one_hot(actions, self.num_actions).float())
      target_values = rewards + (self.gamma * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])

      #Update network parameters
      self.optimizer.zero_grad()
      loss = torch.nn.MSELoss()(q_values , target_values.detach())
      loss.backward()
      self.optimizer.step()

    if self.training_steps % self.target_update_period == 0:
      self.adrqn_target.load_state_dict(self.adrqn.state_dict())

    self.training_steps += 1

  # Convert index to one hot action
  def _idx_to_one_hot(self, idx):
    one_hot_action = []
    for i in range(20):
      if i == idx:
        one_hot_action.append(1)
      else:
        one_hot_action.append(0)
    return one_hot_action