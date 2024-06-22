import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Custom dataset for few-shot learning
class FewShotDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# MAML class
class MAML:
    def __init__(self, model, inner_lr, outer_lr, inner_steps, task_batch_size):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.task_batch_size = task_batch_size
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr)
        
    def inner_update(self, loss, model):
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        updated_model = model
        with torch.no_grad():
            for param, grad in zip(updated_model.parameters(), grads):
                param -= self.inner_lr * grad
        return updated_model
    
    def forward(self, x):
        return self.model(x)
    
    def meta_update(self, meta_loss):
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
    
    def train_step(self, task_batch):
        meta_loss = 0.0
        for task in task_batch:
            data, labels = task
            train_data, val_data = data
            train_labels, val_labels = labels
            
            # Inner loop
            adapted_model = self.model
            for _ in range(self.inner_steps):
                train_outputs = adapted_model(train_data)
                train_loss = nn.CrossEntropyLoss()(train_outputs, train_labels)
                adapted_model = self.inner_update(train_loss, adapted_model)
            
            # Outer loop
            val_outputs = adapted_model(val_data)
            val_loss = nn.CrossEntropyLoss()(val_outputs, val_labels)
            meta_loss += val_loss
        
        meta_loss /= len(task_batch)
        self.meta_update(meta_loss)

# Generate synthetic data for illustration
def generate_synthetic_data(num_samples, input_size, num_classes):
    data = torch.randn(num_samples, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return data, labels

# Hyperparameters
input_size = 10
hidden_size = 64
output_size = 5
inner_lr = 0.01
outer_lr = 0.001
inner_steps = 1
task_batch_size = 4
num_tasks = 10

# Initialize model and MAML
model = SimpleNN(input_size, hidden_size, output_size)
maml = MAML(model, inner_lr, outer_lr, inner_steps, task_batch_size)

# Generate tasks
tasks = []
for _ in range(num_tasks):
    train_data, train_labels = generate_synthetic_data(20, input_size, output_size)
    val_data, val_labels = generate_synthetic_data(20, input_size, output_size)
    tasks.append(((train_data, val_data), (train_labels, val_labels)))

# Train MAML
for epoch in range(100):
    task_batch = [tasks[i] for i in np.random.choice(len(tasks), task_batch_size)]
    maml.train_step(task_batch)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Meta Loss: {meta_loss.item()}")

