import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module): #create a neural network (input: state, output:action)
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # x is tensor of states
        x = F.relu(self.linear1(x)) # apply linear layer 1 and relu activation function 
        x = self.linear2(x) # apply linear layer 2 and don't need activation  -> can just use raw numbers
        return x

    def save(self, file_name='model.pth'): # saves the model 
        model_folder_path = './model'
        if not os.path.exists(model_folder_path): # check if file exists
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # saves a state dictionary


class QTrainer: #training and optimization
    # initialize trainer
    def __init__(self, model, lr, gamma):
        # store variables
        self.lr = lr
        # gamma = discount rate -> smaller than one
        self.gamma = gamma
        self.model = model
        # pytorch optimization step
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # need model parameters and learning rate
        # loss function
        self.criterion = nn.MSELoss() # mean squared error loss

    # train step function
    # state, action, reward -> stored parameters from last time
    def train_step(self, state, action, reward, next_state, done):
        # convert input parameters to pytorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # if multiple values, then already in form (n, x)

        if len(state.shape) == 1:
            # convert x -> (1, x)
            # (number of batches, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)
        # pred returns Q values for each action
        # pred = [Q-val, Q-val, Q-val]

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # only want the maximum of the next predicted Q values
        # set the index of the action w highest Q-value to the new Q-value
        
        # pred.clone()
        target = pred.clone()
        # target = [[Q-val, Q-val, Q-val],
        #           [Q-val, Q-val, Q-val],
        #           [Q-val, Q-val, Q-val],
        #           [Q-val, Q-val, Q-val]]
        # iterate through each step in sequence - iterate through all tensors
        for idx in range(len(done)):
            # find new Q-vals for each state
            Q_new = reward[idx]
            print("reward: {}".format(reward[idx]))
            print("target: {}".format(target.shape))
            # if state is not terminal
            if not done[idx]:
                # preds[argmax(action)] = Q_new
                # index of action with highest Q value set to new Q value
                # next_state: next_state[idx] -> *
                # next predicted Q-values: self.model(*) -> *
                # maximum of next predicted Q-value: torch.max(*)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # target = target Q values
            # target of the current index and the argmax of current action
            # *******
            print("action: {}".format(action[idx]))
            print("argmax: {}".format(torch.argmax(action[idx]).item()))
            print("Q new: {}".format(Q_new))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # empty gradient
        self.optimizer.zero_grad()
        # calculate loss
        # target: Q_new
        # pred: Q
        loss = self.criterion(target, pred)
        # apply backpropagation to update gradients
        loss.backward()

        self.optimizer.step()



