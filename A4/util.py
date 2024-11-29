import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss, sequence_len=10, hidden_size=5, lr=0.01, model_type='lstm'):
  fig, ax = plt.subplots(1, 2, figsize=(16, 5))
  plt.suptitle(f'Testing and validation loss for {model_type}\n Sequence Length is {sequence_len}, hidden_size = {hidden_size}, and learning_rate = {lr}')
  ax[0].set_xlabel('Epochs')
  ax[0].grid()
  ax[0].plot(train_loss, label='Training loss', color='r')
  ax[0].plot(val_loss, label='Validation loss', color='b')
  ax[0].set_ylabel("Loss")
  ax[0].legend()

  ax[1].set_xlabel('Epochs')
  ax[1].grid()
  ax[1].plot(train_loss, label='Training loss', color='r')
  ax[1].plot(val_loss, label='Validation loss', color='b')
  ax[1].legend()
  ax[1].set_ylim([0, 0.05])
  ax[1].set_title("Zoomed-in plot")
  ax[1].set_ylabel("Loss")
  plt.savefig(f'./figures/{model_type}_seqlen{sequence_len}_hiddensize{hidden_size}_lr{lr}.eps')
  plt.show()