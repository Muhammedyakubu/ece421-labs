import torch
import torch.nn as nn


# Create RNN Model
class VanillaRNN(nn.Module):
  def __init__(self,
                input_size,
                hidden_size,
                output_size,
                nonlinearity='tanh'):
    super(VanillaRNN, self).__init__()

    # Number of hidden dimensions
    self.hidden_size = hidden_size
    
    #### YOUR CODE HERE ####
    # implement self.rnn which should apply a single-layer Elman RNN with 
    # tanh or ReLU non-linearity to an input sequence.
    # HINT: It is just one line 0f code looking like `self.rnn = nn.RNN(...)`.
    self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=nonlinearity, batch_first=True)
    ## YOUR CODE ENDS HERE ##
    
    #### YOUR CODE HERE ####
    # implement self.fc which is the readout layer
    # HINT: It is just one line of code looking like `self.fc = nn.Linear(...).
    self.fc = nn.Linear(hidden_size, output_size)
    ## YOUR CODE ENDS HERE ##

  def forward(self, x):
    """
    Forward method:
      Input: x
        It is a batch input sequences. 
        It is a tensor of size (batch_size, sequence_length, input_dimention) 

      Output: out
        the predicted labels for the input batch.
        It is a tensor of size (batch_size, 1, 1).
    """
    h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
    out = None
    
    #### YOUR CODE HERE ####
    # complete the forward method. You should pass each entry of a sequence 
    # through self.rnn followed by self.fc and keep updating the hidden states. 
    # After passing the last element of the sequence through self.rnn followed
    # by self.fc, you can return the last output from self.fc.
    # Our implementation has 3 lines of code, but feel free to deviate from that
    
    out, _ = self.rnn(x, h0)  # Process the entire sequence
    out = out[:, -1, :]  # Get hidden state of last timestep
    out = self.fc(out).unsqueeze(-1)  # Pass through the fully connected layer
    
    ## YOUR CODE ENDS HERE ##
    
    return out


class SimpleLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = 1
    self.output_size = output_size
    
    ### YOUR CODE HERE ###
    # implement self.lstm which applies a single-layer lstm to an input sequence.
    # HINT: It is just one line 0f code looking like `self.lstm = nn.LSTM(...)`.
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    ## YOUR CODE ENDS HERE ##
    
    ### YOUR CODE HERE ###
    # implement self.fc which is the readout layer
    # HINT: It is just one line of code looking like `self.fc = nn.Linear(...).
    self.fc = nn.Linear(hidden_size, output_size)
    ## YOUR CODE ENDS HERE ##
  
  def forward(self, x):
    """
    Forward method:
      Input: x
        It is a batch input sequences. 
        It is a tensor of size (batch_size, sequence_length, input_dimention) 

      Output: out
        the predicted labels for the input batch.
        It is a tensor of size (batch_size, 1, 1).
    """
    h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
    out = None
    
    #### YOUR CODE HERE ####
    # complete the forward method. You should pass each entry of a sequence 
    # through self.lstm followed by self.fc and keep updating the hidden states  
    # and the cell states. After passing the last element of the sequence through
    # self.lstm followed by self.fc, you can return the last output from self.fc
    # Our implementation has 3 lines of code, but feel free to deviate from that
    
    out, (_, _) = self.lstm(x, (h0, c0))  # Process the entire sequence
    out = out[:, -1, :]  # Get hidden state of last timestep
    out = self.fc(out).unsqueeze(-1)  # Pass through the fully connected layer
    
    ## YOUR CODE ENDS HERE ##
    
    return out
