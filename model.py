import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))


class QTrainer:
    
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def _ensure_numpy_float(self, x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.array(x, dtype=np.float32)
    
    def _ensure_numpy_bool(self, x):
        if isinstance(x, np.ndarray):
            return x.astype(bool)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(bool)
        return np.array(x, dtype=bool)
    
    def train_step(self, state, action, reward, next_state, done):
        state_np = self._ensure_numpy_float(state)
        next_state_np = self._ensure_numpy_float(next_state)
        action_np = self._ensure_numpy_float(action)
        reward_np = self._ensure_numpy_float(reward)
        done_np = self._ensure_numpy_bool(done)
        state_t = torch.from_numpy(state_np).float()
        next_state_t = torch.from_numpy(next_state_np).float()
        action_t = torch.from_numpy(action_np).float()
        reward_t = torch.from_numpy(reward_np).float()
        done_t = torch.from_numpy(done_np).to(torch.bool)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
            next_state_t = next_state_t.unsqueeze(0)
            action_t = action_t.unsqueeze(0)
            reward_t = reward_t.unsqueeze(0)
            done_t = done_t.unsqueeze(0)
        batch_size = state_t.size(0)
        pred = self.model(state_t)
        target = pred.clone().detach()
        with torch.no_grad():
            next_pred = self.model(next_state_t)
            next_max = torch.max(next_pred, dim=1)[0]
        reward_vec = reward_t.view(batch_size, -1).squeeze(1) if reward_t.dim() > 1 else reward_t.squeeze()
        done_vec = done_t.view(batch_size, -1).squeeze(1) if done_t.dim() > 1 else done_t.squeeze()
        not_done_mask = (~done_vec).to(dtype=torch.float)
        Q_new = reward_vec + not_done_mask * (self.gamma * next_max)
        action_indices = torch.argmax(action_t, dim=1).long()
        target[torch.arange(batch_size), action_indices] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
