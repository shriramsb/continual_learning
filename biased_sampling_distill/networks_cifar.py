import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherNetwork(nn.Module):
    def __init__(self, input_shape, num_class):
        super(TeacherNetwork, self).__init__()
        self.input_shape = input_shape
        input_size = 1
        for i in range(len(input_shape)):
            input_size *= input_shape[i]
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden, training=self.is_training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_hidden, training=self.is_training)
        x = self.fc3(x)
        return x

class StudentNetwork(nn.Module):
    def __init__(self, input_shape, num_class):
        super(StudentNetwork, self).__init__()
        self.input_shape = input_shape
        input_size = 1
        for i in range(len(input_shape)):
            input_size *= input_shape[i]
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True
        self.all_layers = [self.fc1, self.fc2]
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden, training=self.is_training)
        x = self.fc2(x)
        return x

class StudentNetworkLarge(nn.Module):
    def __init__(self, input_shape, num_class):
        super(StudentNetworkLarge, self).__init__()
        self.input_shape = input_shape
        input_size = 1
        for i in range(len(input_shape)):
            input_size *= input_shape[i]
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True
        self.all_layers = [self.fc1, self.fc2, self.fc3]
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden, training=self.is_training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_hidden, training=self.is_training)
        x = self.fc3(x)
        return x