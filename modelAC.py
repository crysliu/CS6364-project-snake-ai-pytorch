import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import math

class ActorNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='actor.pth'):
        model_folder_path = './actor'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='critic.pth'):
        model_folder_path = './critic'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Actor_Trainer:
    def __init__(self, actor_model, critic_model, lr, alpha):
        self.lr = lr
        self.alpha = alpha
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.optimizer = optim.Adam(actor_model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # def train_step(self, state, action, reward, next_state, done):
    def train_step(self, state, action, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        # reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            # reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted next action with current state
        pred = self.actor_model(state)
        target = pred.clone()
        print("target: {}".format(target.shape))
        for idx in range(len(done)):
            pi_new = 0
            if not done[idx]:
                # pi_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                print("torch max: {}".format(torch.max(self.actor_model(next_state[idx]))))
                print("critic model: {}".format(self.critic_model(state[idx])))
                pi_new = self.alpha * torch.max(self.actor_model(state[idx])) * torch.max(self.critic_model(state[idx]))
            
            #target[idx][torch.argmax(action[idx]).item()] = Q_new
            print("argmax: {}".format(torch.argmax(action[idx])))
            print("pi new: {}".format(pi_new))
            target[idx][torch.argmax(action[idx]).item()] = pi_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

class Critic_Trainer:
    def __init__(self, model, lr, beta):
        self.lr = lr
        self.beta = beta
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.beta * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
