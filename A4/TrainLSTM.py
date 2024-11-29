import torch
import torch.optim as optim
import torch.nn as nn

from models import SimpleLSTM
from train import train


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train1LayerLSTM(
  train_set,
  val_set,
  LSTM_input_size,
  LSTM_output_size,
  LSTM_hidden_size,
  optimizer_name='sgd',
  lr=0.001,
  batch_size=64,
  max_epoch=100
):

  # Initialize model
  LSTM_model = SimpleLSTM(input_size=LSTM_input_size,
                          hidden_size=LSTM_hidden_size,
                          output_size=LSTM_output_size).to(DEVICE)

  # Initialize loss
  loss_function = nn.MSELoss()

  # Initialize optimizer
  if optimizer_name=='sgd':
    optimizer = optim.SGD(LSTM_model.parameters(),
                          lr=lr)
  elif optimizer_name=='adam':
    optimizer = optim.Adam(LSTM_model.parameters(),
                          lr=lr)

  train_loss, val_loss = train(train_set,
                               val_set,
                               LSTM_model,
                               loss_function,
                               optimizer,
                               max_epoch=max_epoch,
                               batch_size=batch_size)
  return LSTM_model, train_loss, val_loss