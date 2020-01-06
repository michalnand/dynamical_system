import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Create(torch.nn.Module):

    def __init__(self, inputs_count, outputs_count):
        super(Create, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm_units_count = 64
        self.lstm_layer = nn.LSTM(inputs_count, hidden_size = self.lstm_units_count, batch_first = True).to(self.device)
        self.fc_layer   = nn.Linear(self.lstm_units_count, outputs_count).to(self.device)
        
        

    def forward(self, x):


        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.lstm_units_count).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(1, x.size(0), self.lstm_units_count).requires_grad_().to(self.device)

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm_layer(x, (h0.detach(), c0.detach()))

        fc_input = out[:, -1, :]


        return self.fc_layer.forward(fc_input)
        
