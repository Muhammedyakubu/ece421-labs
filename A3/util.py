import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import os

def integers_to_one_hot(integer_vector, max_val=None):
    integer_vector = np.squeeze(integer_vector)
    if max_val == None:
        max_val = np.max(integer_vector)
    one_hot = np.zeros((integer_vector.shape[0], max_val + 1))
    for i, integer in enumerate(integer_vector):
        one_hot[i, integer] = 1.0
    return one_hot


def center(X, axis=0):
    return X - np.mean(X, axis=axis)


def normalize(X, axis=0, max_val=None):
    X -= np.min(X, axis=axis)
    if max_val is None:
        X /= np.max(X, axis=axis)
    else:
        X /= max_val
    return X


def standardize(X, axis=0):
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    X -= mean
    X /= std + 1e-10
    return X


class Data:
    def __init__(
        self,
        data,
        batch_size=50,
        labels=None,
        out_dim=None,
    ):

        self.data_ = data
        self.labels = labels
        self.out_dim = out_dim
        self.iteration = 0
        self.batch_size = batch_size
        self.n_samples = data.shape[0]
        self.samples_per_epoch = math.ceil(self.n_samples / batch_size)

    def shuffle(self):
        idxs = np.arange(self.n_samples)
        np.random.shuffle(idxs)

        self.data_ = self.data_[idxs]
        if self.labels is not None:
            self.labels = self.labels[idxs]

    def sample(self):
        if self.iteration == 0:
            self.shuffle()

        low = self.iteration * self.batch_size
        high = self.iteration * self.batch_size + self.batch_size

        self.iteration += 1
        self.iteration = self.iteration % self.samples_per_epoch

        if self.labels is not None:
            return self.data_[low:high], self.labels[low:high]
        else:
            return self.data_[low:high]

    def reset(self):
        self.iteration == 0


class Dataset:
    def __init__(self, training_set, training_labels, batch_size,
                 validation_set=None, validation_labels=None, test_set=None,
                 test_labels=None,):

        self.batch_size = batch_size
        self.n_training = training_set.shape[0]
        self.n_validation = validation_set.shape[0]
        self.out_dim = training_labels.shape[1]

        self.train = Data(data=training_set, batch_size=batch_size, labels=training_labels, out_dim=self.out_dim,)

        if validation_set is not None:
            self.validate = Data(data=validation_set, batch_size=batch_size,
                                 labels=validation_labels, out_dim=self.out_dim)

        if test_set is not None:
            self.test = Data(data=test_set, batch_size=batch_size, labels=test_labels,
                             out_dim=self.out_dim)


class Logger:
    def __init__( self, model_name, model_args, data_args, save=False, plot=False, save_dir="experiments/",):

        self.model_name = model_name
        self.model_args = model_args
        self.data_args = data_args
        self.save = save
        self.save_dir = save_dir + model_name + "/"
        self.plot = plot
        self.counter = 0
        self.log = {}

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        with open(self.save_dir + "model_args", "wb") as f:
            pickle.dump(self.model_args, f)

        with open(self.save_dir + "data_args", "wb") as f:
            pickle.dump(self.data_args, f)

    def push(self, log):
        if self.counter == 0:
            self.log = {k: {} for k in log.keys()}

            # self.log = {k: [] if k != "params" else {} for k in log.keys()}
            if "params" in log.keys():

                self.log["params"] = {
                    k: {"max": [], "min": []} for k in log["params"].keys()
                }

            self.log["loss"] = {"train": [], "validate": []}
            self.log["error"] = {"train": [], "validate": []}

        self.counter += 1
        for k, v in log.items():
            if k == "params":
                for param, vals in v.items():
                    self.log["params"][param]["max"].append(vals["max"])
                    self.log["params"][param]["min"].append(vals["min"])

            else:
                self.log[k]["train"].append(v["train"])
                self.log[k]["validate"].append(v["validate"])

        if self.save:
            with open(self.save_dir + "log", "wb") as f:
                pickle.dump(self.log, f)
            if self.plot:
                self._plot()

    def reset(self):
        self.log = {}
        self.counter = 0

    def _plot(self):
        for k, v in self.log.items():
            if k == "params":
                for param, vals in v.items():
                    plt.figure(figsize=(15, 10))
                    plt.plot(vals["max"], label="{}_max".format(param))
                    plt.plot(vals["min"], label="{}_min".format(param))
                    plt.legend()
                    plt.xlabel("epochs")
                    plt.ylabel(param)
                    plt.title(self.model_name)
                    plt.savefig(self.save_dir + param)
                    plt.close()
            else:
                plt.figure(figsize=(15, 10))
                plt.plot(v["train"], label="training")
                plt.plot(v["validate"], label="validation")
                plt.legend()
                plt.xlabel("epochs")
                plt.ylabel(k)
                plt.title(self.model_name)
                plt.savefig(self.save_dir + k)
                plt.close()



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
