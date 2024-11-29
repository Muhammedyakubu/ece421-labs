import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader,
               model,
               loss_fn,
               optimizer,
               batch_size,
               verbose=0):

  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  train_loss = 0

  for batch, (X, y) in enumerate(dataloader):
    
    pred = model(X)
    loss = loss_fn(pred, y)

    #### YOUR CODE HERE ####
    # Compute prediction and loss. Backpropagate and update parameters.
    # Don't forget to set the gradients to zero with 
    # optimizer.zero_grad(), after each update.
    # Our implementation has 3 lines of code, but feel free to deviate from that
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ## YOUR CODE ENDS HERE ##


  # After going through all batches and updating models parameter based on each
  # batch, we find the loss on the train dataset.
  # Evaluating the model with torch.no_grad() ensures that no gradients are
  # computed during test mode, also serves to reduce unnecessary gradient
  # computations and memory usage for tensors with requires_grad=True
  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X)
      train_loss += loss_fn(pred, y).item()

  train_loss /= num_batches
  print(f"Train Avg loss: {train_loss:>8f}")
  return train_loss


def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = 0

  # Evaluating the model with torch.no_grad() ensures that no gradients are
  # computed during test mode, also serves to reduce unnecessary gradient
  # computations and memory usage for tensors with requires_grad=True
  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X)
      test_loss += loss_fn(pred, y).item()

  test_loss /= num_batches
  print(f"Test  Avg loss: {test_loss:>8f}")
  return test_loss


def train(TrainDataset,
          ValDataset,
          model,
          loss_function,
          optimizer,
          max_epoch=100,
          batch_size=64):

  train_loss = torch.zeros(max_epoch)
  test_loss = torch.zeros(max_epoch)

  train_dataloader = DataLoader(dataset=TrainDataset,
                                batch_size=batch_size,
                                shuffle=True)
  test_dataloader = DataLoader(dataset=ValDataset,
                               batch_size=batch_size,
                               shuffle=True)

  for t in range(max_epoch):
    print(f"Epoch {t+1} -------------------")

    train_loss[t] = train_loop(train_dataloader,
                               model,
                               loss_function,
                               optimizer,
                               batch_size)

    test_loss[t] = test_loop(test_dataloader,
                             model,
                             loss_function)
  print("Done!")

  return train_loss, test_loss
