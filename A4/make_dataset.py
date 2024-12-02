import numpy as np
from torch.utils.data import Dataset
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomAddingDataset(Dataset):
  def __init__(self, input_data, labels):
    self.dataset = input_data
    self.labels = labels

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx], self.labels[idx]


# Function to generate sequences for the "adding numbers" task
def generate_adding_sequence(sequence_length):
  """
  Generates an input data, which is a tensor of shape (`sequence_length`, 2),
  and a target value, which is a tensor of size (1, 1).
    - Input tensor:
      * The first column denotes a sequence of lenght `sequence_length`, with values 
      drawn from indepandantly from a uniform distribution between 0 and 1.

      * The second column, known as the mask, is a sequence of lenght
      `sequence_length`, with binary values. Randomly two enties in this
      sequence are set to 1, and the rest are zeros.
      
    - The target value of the generated sequence is the sum of the values in the 
    iput sequence with mask 1.

    - An example input tensor with sequence_length=5 is:
        tensor([[0.4485, 0.0000],
                [0.8026, 1.0000],
                [0.1788, 0.0000],
                [0.1535, 0.0000],
                [0.3918, 1.0000]])
      and its target/label would be tensor([[1.1945]]), 
      as 0.8026 + 0.3918 = 1.1945.

    - How should we feed the input sequence to our model?
      * As mentioned, input is a tensor of shape (`sequence_length`, 2).
      Each row in this sequence represent the input to your neural netwrok at a
      time step. So, we feed the rows of datapoint sequentially and record the 
      output of the model after feeding in the last row as the predicted label.
      * For instance consider the datapoint below with sequence length of 3:
      (tensor([[0.4485, 0.0000],
               [0.8026, 1.0000],
               [0.3918, 1.0000]]), tensor([[1.1945]]))
        + We first set the iitial hidden states of our RNN, h0, to zero. 
        + Then feed [0.4485, 0.0000] to our Elman RNN model. This should give us
        a prediction, which we can ignore, and updated hidden state h1.
        + Then we set the hidden state of our RNN to h1, and input the second
        element in the sequence, i.e. [0.8026, 1.0000]. This should give us a 
        prediction, which we can ignore, and updated hidden state h2.
        + Then we set the hidden state of our RNN to h2, and input the third
        element in the sequence, i.e. [0.3918, 1.0000]. This should give us a 
        prediction, which we store as the prediction of our model, and updated 
        hidden state h3.
  """

  x = np.zeros(shape=(sequence_length, 2))
  x[:, 0:1] = np.random.rand(sequence_length, 1)
  idx = np.random.choice(sequence_length, size=(2,), replace=False)
  x[idx[0], 1] = 1
  x[idx[1], 1] = 1

  y = x[idx[0], 0] + x[idx[1], 0]

  torch_x = torch.tensor(x, dtype=torch.float32)
  torch_y = torch.tensor(y, dtype=torch.float32)[...,None,None]
  return torch_x, torch_y


def generate_adding_dataset(count, sequence_length):
  """
  Returns a dataset with `count` datapoint. Each datapoint is a 2-tuple 
  (input, label)
  """

  inputs, targets = [], []
  for tt in range(count):
    x, y = generate_adding_sequence(sequence_length)
    inputs.append(x)
    targets.append(y)

  inputs = torch.stack(inputs).to(DEVICE)
  targets = torch.stack(targets).to(DEVICE)

  return CustomAddingDataset(inputs, targets)


def make_adding_train_val_dataset(train_count=10000,
                                  val_count=1000,
                                   sequence_length=10):
  """
  Generates a training dataset with `train_count` datapoints
  and a test dataset with `val_count` datapoints.
  
  Each datapoint is a 2-tuple (input, label)
  """

  TrainDataset = generate_adding_dataset(train_count, sequence_length)
  ValDataset = generate_adding_dataset(val_count, sequence_length)
  return TrainDataset, ValDataset
