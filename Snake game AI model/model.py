import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If single step (1D tensor), add batch dimension: shape (1, x)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Predicted Q-values with current state
        pred = self.model(state)

        # 2. Q_new = r + y * max(next_predicted Q value) -> only for the chosen action
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Calculate loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

if __name__ == '__main__':
    # Test instantiation
    net = Linear_QNet(11, 256, 3)
    trainer = QTrainer(net, lr=0.001, gamma=0.9)

    # Dummy transition: state=11, action=3, reward=scalar, next_state=11, done=bool
    dummy_state = [0] * 11
    dummy_action = [1, 0, 0]
    dummy_reward = 0
    dummy_next_state = [0] * 11
    dummy_done = False

    # Short memory train step check
    trainer.train_step(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
    print("model.py successfully instantiated and verified!")