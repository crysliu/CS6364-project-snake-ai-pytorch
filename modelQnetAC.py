import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.distributions import Categorical

class ActorNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, action_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim=-1)
        return x
        # distribution = Categorical(F.softmax(x, dim=-1))
        # return probability distribution of each action

    def save(self, file_name='actor.pth'):
        model_folder_path = './actor'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class CriticNet(nn.Module):
    def __init__(self, state_input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(state_input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        # returns the Q val for each possible action

    def save(self, file_name='critic.pth'):
        model_folder_path = './critic'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class ACTrainer:
    def __init__(self, actor_model, critic_model, lr, gamma, alpha, beta):
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.actor_optimizer = optim.Adam(actor_model.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(critic_model.parameters(), lr=self.lr)
        self.actor_criterion = nn.MSELoss()
        self.critic_criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        # action: [0, 0, 1] format
        # [1 x 3] array
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        log_pi = []
        values = []
        returns = []

        # 1: predicted Q values with current state
        # pred = self.model(state)

        # set predictions
        actor_pred = self.actor_model(state)
        critic_pred = self.critic_model(state)

        # target = pred.clone()
        # set targets
        actor_target = actor_pred.clone()
        critic_target = critic_pred.clone()

        for idx in range(len(done)):
            s = state[idx]
            a = action[idx]
            r = reward[idx]
            next_s = next_state[idx]
            d = done[idx]

            dist = Categorical(self.actor_model(s))
            act =torch.argmax(a)
            # action type : torch.LongTensor
            log_pi.append(dist.log_prob(act).unsqueeze(0))
           
            # log_pi.append(dist.log_prob(act2).unsqueeze(0))
            values.append(torch.max(self.critic_model(s)))
            
            # print("reward: {}".format(r))
            Q_curr = self.critic_model(s)[act]
            Q_new = r
            if not d:
                Q_new = r + self.gamma * torch.max(self.critic_model(next_s))
            
            # update targets 
            actor_target[idx][act] = dist.log_prob(act).unsqueeze(0) * Q_curr
            critic_target[idx][act] = Q_new
            returns.append(Q_new)

        log_pi = torch.cat(log_pi)        
        
        returns = torch.tensor(returns)
        values = torch.tensor(values)
        # values = torch.tensor(values)
        # values = torch.cat(values)
        # advantage = returns - values
        advantage = returns
        advantage.requires_grad_(True)
        
        actor_loss = -(log_pi * advantage.detach()).mean()
        critic_loss = self.critic_criterion(critic_target, critic_pred)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        
