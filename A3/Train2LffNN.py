# An attribute dictionary is exactly like a dictionary, 
# except you can access the values as attributes rather than keys.
class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


# Define layer arguments
fc1 = AttrDict({"activation": "relu", "n_out": 5})
fc_out = AttrDict({"activation": "softmax", "n_out": None}) # n_out is not defined for last layer. This will be set by the dataset.
layer_args = [fc1, fc_out]

# Define model, data, and logger arguments
optimizer_args = AttrDict({"lr": 0.01, "momentum_hyperparameter": 0.9})
model_args = AttrDict({"loss": "cross_entropy", "layer_args": layer_args, 
                       "optimizer_args": optimizer_args, "seed": 0})
data_args = AttrDict({"name": "iris", "batch_size": 25})
log_args = AttrDict({"save": True, "plot": True, "save_dir": "experiments/"})

# Set random seed. Random seed must be set before importing other modules.
import numpy as np
np.random.seed(model_args.seed)

# Define model name for saving.
model_name = f'{len(layer_args)}layers' + f'_{fc1["n_out"]}' + f'-lr{optimizer_args.lr}' +\
             f'_mom{optimizer_args.momentum_hyperparameter}' + f'_seed{model_args.seed}'

# Initialize logger, model, and dataset.
from model import NeuralNetworkModel
from util import Dataset, Logger
logger = Logger(model_name=model_name, model_args=model_args, data_args=data_args, 
                save=log_args.save, plot=log_args.plot, save_dir=log_args.save_dir)
model = NeuralNetworkModel(loss=model_args.loss, layer_args=model_args.layer_args,
                           optimizer_args=model_args.optimizer_args, logger=logger)
dataset = Dataset(
    np.load('datasets/iris/iris_train_data.npy'),
    np.load('datasets/iris/iris_train_labels.npy'),
    data_args.batch_size,
    np.load('datasets/iris/iris_val_data.npy'),
    np.load('datasets/iris/iris_val_labels.npy'),
    np.load('datasets/iris/iris_test_data.npy'),
    np.load('datasets/iris/iris_test_labels.npy')
    )

# Train model!
epochs = 100
print(f"Training feed forward neural network on {data_args.name} with SGD for {epochs} epochs...")
model.train(dataset, epochs=epochs)
test_log = model.test(dataset)
