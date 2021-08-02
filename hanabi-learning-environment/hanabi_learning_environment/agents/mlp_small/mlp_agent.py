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
import random
from collections import defaultdict

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):

    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus

@gin.configurable
class MLP(nn.Module):
    def __init__(self, n_actions, observation_size, hidden_size=512):
        super(MLP, self).__init__()
        self.n_actions = n_actions
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        # Create fully connected layers
        self.fc1 = nn.Linear(self.observation_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.n_actions)
    
    def forward(self, observation):
        # Build the MLP for a forward pass
        x = self.fc1(observation)
        x = F.relu(x)
        q_values = self.fc2(x)
        return q_values
    
    def act(self, observation, legal_actions, epsilon):
        q_values_tensor = self.forward(observation)
        # print("q_values_tensor_shape: ", q_values_tensor.shape)
        # debug - changed from q_values_tensor[0][0] to q_values_tensor[0]
        q_values_np = q_values_tensor[0].cpu().detach().numpy()
        # print("q_values_np_shape: ", q_values_np.shape)

        # Create list with q_values for legal actions only
        q_values = []
        legal_action_indices = np.where(legal_actions == 0.0)[0]
        for idx in range(len(legal_actions)):
          if idx in legal_action_indices:
            q_values.append(q_values_np[idx])
          else:
            q_values.append(-np.Inf)

        # Take epsilon-greedy action
        if np.random.uniform() > epsilon:
          action = np.nanargmax(q_values)
        else:
          action = np.random.choice(legal_action_indices)
        return action

@gin.configurable
class ExpBuffer():
    def __init__(self, max_storage):
        self.max_storage = max_storage
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, oarod):
        if self.counter < self.max_storage-1:
            self.counter +=1
        if self.filled < self.max_storage:
            self.filled += 1
        else:
            self.counter = 0
        self.storage[self.counter] = oarod
    
    def sample(self, batch_size):
        # Returns sizes of (batch_size, *) depending on action/observation/return/done

        # print("storage[0]: ", self.storage[0])

        sample = random.sample(self.storage[0: self.filled], batch_size)
        last_observations, actions, rewards, observations, dones = zip(*sample)
           
        return torch.tensor(last_observations, dtype = torch.float32).cuda(),\
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
               replay_buffer_size=100000,
               batch_size = 32,
               epsilon_train = 0.02,
               epsilon_eval = 0.001,
               epsilon_decay_period = 1000,
               gamma = 0.99,
               learning_rate = 0.0025,
               explore = 500,
               last_action = {},
               last_observation = {},
               begin = defaultdict(lambda: 1),
               i_episode = 0,
               training_steps = 0,
               update_period = 4,
               target_update_period = 500,
               loss = 0,
               epsilon_fn=linearly_decaying_epsilon,
               tf_device='/cpu:*',):

    # Global variables.
    self.num_actions = num_actions
    self.observation_size = observation_size
    self.num_players = num_players
    self.replay_buffer_size = replay_buffer_size
    self.batch_size = batch_size
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.explore = explore

    self.last_action = last_action
    self.last_observation = last_observation
    self.begin = begin
    self.i_episode = i_episode
    self.training_steps = training_steps
    self.update_period = update_period
    self.target_update_period = target_update_period
    self.loss = loss
    self.epsilon_fn = epsilon_fn
    self.eval_mode = False
    self.q_values_list = []
    self.target_values_list = []
    self.raw_data = [["training_step", "example_reward", "example_done", "avg_q_value", "avg_target_value", 
                      "loss", "fc1_weight_NORM", "fc1_bias_NORM", "fc2_weight_NORM", "fc2_bias_NORM"]]

    # MLP and ExpBuffer instances
    self.replay_buffer = ExpBuffer(self.replay_buffer_size)
    self.mlp = MLP(self.num_actions, self.observation_size).cuda()
    self.mlp_target = MLP(self.num_actions, self.observation_size).cuda()
    self.mlp_target.load_state_dict(self.mlp.state_dict())

    # Optimizer
    self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr = self.learning_rate) # , weight_decay=1e-5

  def begin_episode(self, current_player, legal_actions, observation):

    # print("begin_episode")
    # print("current_player: ", current_player)
    # print("current_player type: ", type(current_player))
    # print("i_episode: ", self.i_episode)

    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.explore, self.epsilon_train)

    # debug - changed shape from .view(1,1,-1) to .view(1,-1) as no seq_len here
    action = self.mlp.act(
      torch.tensor(observation).float().view(1,-1).cuda(),
      legal_actions,
      epsilon = epsilon)

    # print("observation torch: ", torch.tensor(observation).float().view(1,-1).cuda())
    # print("observation torch shape: ", torch.tensor(observation).float().view(1,-1).cuda().shape)
    # print("legal_actions: ", legal_actions)
    # print("epsilon: ", epsilon)

    self.last_action[current_player] = action.copy()
    self.last_observation[current_player] = observation.copy()

    return action

  def step(self, reward, current_player, legal_actions, observation):

    # print("step")
    # print("current_player: ", current_player)
    # print("i_episode: ", self.i_episode)

    observation_copy = observation.copy()

    done = False
    if self.begin[current_player] == 1:
      self.replay_buffer.write_tuple((
        self.last_observation[current_player],
        self.last_action[current_player],
        reward,
        observation_copy,
        done
        ))
      self.begin[current_player] = 0
      if self.i_episode > self.explore:
        # print("update network after begin")
        self.loss = self._update_network()

    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.explore, self.epsilon_train)

    action = self.mlp.act(
      torch.tensor(observation).float().view(1,-1).cuda(),
      legal_actions,
      epsilon = epsilon)

    if self.begin[current_player] == 0:
      self.replay_buffer.write_tuple((
        self.last_observation[current_player],
        action,
        reward,
        observation_copy,
        done
        ))
      if self.i_episode > self.explore:
        # print("update network after step")
        self.loss = self._update_network()

      self.last_action[current_player] = action.copy()
      self.last_observation[current_player] = observation.copy()

    return action, self.loss

  def end_episode(self, final_rewards, player):

    # print("end_episode")
    # print("i_episode: ", self.i_episode)
    print("training_steps:", self.training_steps)

    done = True
    for current_player in range(self.num_players):
      if current_player in self.last_observation:
        self.replay_buffer.write_tuple((
            self.last_observation[current_player],
            0,
            final_rewards[current_player],
            np.array([0.]*len(self.last_observation[current_player])),
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

      last_observations, actions, rewards, observations, dones = self.replay_buffer.sample(self.batch_size)
    #   print("last_observations: ", last_observations)
    #   print("last_observations shape: ", last_observations.shape)
    #   print("last_observations[0]: ", last_observations[0])
    #   print("last_observations[0] shape: ", last_observations[0].shape)

      q_values = self.mlp.forward(last_observations).float()
    #   print("q_values initial: ", q_values)
    #   print("q_values initial shape: ", q_values.shape)
    #   print("q_values initialtype: ", type(q_values))
    #   print("actions: ", actions)
    #   print("actions shape: ", actions.shape)
    #   print("actionstype: ", type(actions))
      q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
    #   print("q_values final: ", q_values)
    #   print("q_values final shape: ", q_values.shape)
    #   print("q_values final type: ", type(q_values))

      predicted_q_values = self.mlp_target.forward(observations).float()
    #   print("predicted_q_values: ", predicted_q_values)
    #   print("predicted_q_values shape: ", predicted_q_values.shape)
    #   print("predicted_q_values type: ", type(predicted_q_values))
      target_values = rewards + (self.gamma * (1 - dones.float()) * torch.max(predicted_q_values, dim = -1)[0])
    #   print("target_values: ", target_values)
    #   print("target_values shape: ", target_values.shape)
    #   print("target_values type: ", type(target_values))
      #Update network parameters
      self.optimizer.zero_grad()
      loss = nn.MSELoss()(q_values , target_values.detach())
      loss.backward()
    #   nn.utils.clip_grad_norm_(self.mlp.parameters(), 1)
      self.optimizer.step()

      self.loss = loss

      fc1_weight, fc1_bias, fc2_weight, fc2_bias = None, None, None, None
      for name, param in self.mlp.named_parameters():
        if name == "fc1.weight":
          fc1_weight = param.grad.norm().item()
        if name == "fc1.bias":
          fc1_bias = param.grad.norm().item()
        if name == "fc2.weight":
          fc2_weight = param.grad.norm().item()
        if name == "fc2.bias":
          fc2_bias = param.grad.norm().item()

    if self.training_steps % self.target_update_period == 0:

      self.q_values_list.append(q_values)
      pickle.dump(self.q_values_list, open('q_values_list.pkl', 'wb' ))
      self.target_values_list.append(target_values)
      pickle.dump(self.target_values_list, open('target_values_list.pkl', 'wb' ))

      lst = [self.training_steps, rewards[0].item(), dones[0].item(),
             torch.mean(q_values, 0).item(),torch.mean(target_values, 0).item(),
             loss.item(), fc1_weight, fc1_bias, fc2_weight, fc2_bias]
      self.raw_data.append(lst)
      with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(self.raw_data)

      self.mlp_target.load_state_dict(self.mlp.state_dict())
      print("eps: ", epsilon)

    self.training_steps += 1

    return self.loss
