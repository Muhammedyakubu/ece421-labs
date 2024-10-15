import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import sys
from collections import Counter

import myTorch


def plot_decision_parameter(f, w_dict, method_name):
    plt.figure(figsize=(9, 9))
    ax = {}
    i = 1
    for k in w_dict:
        ax[k] = plt.subplot(2, len(w_dict) // 2, i)
        ax[k].plot(np.arange(-5, 5, 0.1), f(np.arange(-5, 5, 0.1)))
        ax[k].scatter(w_dict[k], [f(k) for k in w_dict[k]])
        ax[k].scatter(w_dict[k][-1], f(w_dict[k][-1]))
        ax[k].set_title(f"{method_name.upper()}: lr = {k:0.4f}")
        ax[k].set_xlabel("x")
        ax[k].set_ylabel("f(x)")
        i += 1


def plot_loss(f, loss, method_name):
    plt.figure(figsize=(9, 9))
    ax = {}
    i = 1
    for k in loss:
        ax[k] = plt.subplot(2, len(loss) // 2, i)
        ax[k].plot(np.arange(len(loss[k])), loss[k])
        ax[k].set_title(f"{method_name.upper()}, lr = {k:0.4f}")
        ax[k].set_xlabel("Iteration")
        ax[k].set_ylabel("f(x)")
        i += 1


def show_trace_2d(f, w_dict, method_name):
    plt.figure(figsize=(9, 9))
    ax = {}
    i = 1
    for k in w_dict:
        ax[k] = plt.subplot(2, len(w_dict) // 2, i)
        i += 1
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = 10 * (0.1 * X**2 + 2 * Y**2)

        CS = ax[k].contour(X, Y, Z)
        ax[k].clabel(CS, inline=True, fontsize=5)
        ax[k].set_title(f"{method_name.upper()}: lr = {k:0.4f}")
        ax[k].set_xlabel(r"$x_1$")
        ax[k].set_ylabel(r"$x_2$")

        ax[k].scatter([j[0] for j in w_dict[k]], [j[1] for j in w_dict[k]])


def train_model(
    X_train, X_test, y_train, y_test, optimizer="sgd", lr=0.0001, n_iter=1000
):
    model = myTorch.MultiClassLogisticRegression(n_iter=n_iter)
    model.fit(X_train, y_train, lr=lr, optimizer=optimizer, verbose=False)
    print(f"{optimizer} Evaluation Score: {model.score(X_test,y_test)}")
    return model


def generateClusteringExamples(numExamples, numWordsPerTopic, numFillerWords):
    """
    Generate artificial examples inspired by sentiment for clustering.
    Each review has a hidden sentiment (positive or negative) and a topic (plot, acting, or music).
    The actual review consists of 2 sentiment words, 4 topic words and 1 filler word, for example:

        good:1 great:1 plot1:2 plot7:1 plot9:1 plot11:1 filler0:1

    numExamples: Number of examples to generate
    numWordsPerTopic: Number of words per topic (e.g., plot0, plot1, ...)
    numFillerWords: Number of words per filler (e.g., filler0, filler1, ...)
    """
    sentiments = [
        ["bad", "awful", "worst", "terrible"],
        ["good", "great", "fantastic", "excellent"],
    ]
    topics = ["plot", "acting", "music"]

    def generateExample():
        x = Counter()
        # Choose 2 sentiment words according to some sentiment
        sentimentWords = random.choice(sentiments)
        x[random.choice(sentimentWords)] += 1
        x[random.choice(sentimentWords)] += 1
        # Choose 4 topic words from a fixed topic
        topic = random.choice(topics)
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic - 1))] += 1
        # Choose 1 filler word
        x["filler" + str(random.randint(0, numFillerWords - 1))] += 1
        return x

    random.seed(42)
    examples = [generateExample() for _ in range(numExamples)]
    return examples


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in list(d2.items()):
        d1[f] = d1.get(f, 0) + v * scale


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))
