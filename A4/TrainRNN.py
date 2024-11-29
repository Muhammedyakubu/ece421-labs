import torch
import torch.optim as optim
import torch.nn as nn

from models import VanillaRNN
from train import train


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train1LayerVanillaRNN(
  train_set,
  val_set,
  RNN_input_size,
  RNN_output_size,
  RNN_hidden_size,
  optimizer_name='sgd',
  lr=0.001,
  batch_size=64,
  max_epoch=100
):

  # Initialize model
  VRNN_model = VanillaRNN(input_size=RNN_input_size,
                          hidden_size=RNN_hidden_size,
                          output_size=RNN_output_size).to(DEVICE)

  # Initialize loss
  loss_function = nn.MSELoss()

  # Initialize optimizer
  if optimizer_name=='sgd':
    optimizer = optim.SGD(VRNN_model.parameters(),
                          lr=lr)
  elif optimizer_name=='adam':
    optimizer = optim.Adam(VRNN_model.parameters(),
                          lr=lr)

  train_loss, val_loss = train(train_set,
                               val_set,
                               VRNN_model,
                               loss_function,
                               optimizer,
                               max_epoch=max_epoch,
                               batch_size=batch_size)
  return VRNN_model, train_loss, val_loss