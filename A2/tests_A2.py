import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

import myTorch
import util


f1 = lambda x: (2 * x**2 + 4 * x + 2) * 5
df1dw = lambda x: (4 * x + 4) * 5

f2 = lambda x: (2 * x**2 + 4 * x + 2) * 11
df2dw = lambda x: (4 * x + 4) * 11

f3 = lambda x: 10 * (x**2 + np.sin(x * np.pi))
df3dw = lambda x: 10 * (2 * x + np.pi * np.cos(x * np.pi))

f4 = lambda x: 20 * (0.1 * x[0] ** 2 + 2 * x[1] ** 2)
df4dw = lambda x: 20 * (np.array([0.2 * x[0], 4 * x[1]]))

f42 = lambda x: 2000 * (0.1 * x[0] ** 2 + 2 * x[1] ** 2)
df42dw = lambda x: 2000 * (np.array([0.2 * x[0], 4 * x[1]]))


def q1():
    w_dict, loss_dict = test_sgd(f1, df1dw, w_init=5.0, max_iter=20)
    util.plot_decision_parameter(f1, w_dict, "sgd")
    util.plot_loss(f1, loss_dict, "sgd")


def q2():
    w_dict, loss_dict = test_sgd(f2, df2dw, w_init=5.0, max_iter=35)
    util.plot_decision_parameter(f2, w_dict, "sgd")
    util.plot_loss(f2, loss_dict, "sgd")


def q3():
    w_dict, loss_dict = test_sgd(f3, df3dw, w_init=5.0, max_iter=2000)
    util.plot_decision_parameter(f3, w_dict, "sgd")
    util.plot_loss(f3, loss_dict, "sgd")


def q4():
    w_dict, loss_dict = test_sgd(
        f4,
        df4dw,
        w_init=np.array([3.0, 3.0]),
        max_iter=500,
        lr_list=[0.001, 0.01, 0.02, 0.03],
    )
    util.show_trace_2d(f4, w_dict, "sgd")
    util.plot_loss(f4, loss_dict, "sgd")


def q5():
    w_dict, loss_dict = test_momentum(
        f3, df3dw, w_init=5.0, max_iter=2000, lr_list=[0.0003, 0.0004, 0.0006, 0.001]
    )
    util.plot_decision_parameter(f3, w_dict, "heavyball_momentum")
    util.plot_loss(f3, loss_dict, "heavyball_momentum")


def q6():
    w_dict, loss_dict = test_momentum(
        f4,
        df4dw,
        w_init=np.array([3.0, 3.0]),
        max_iter=500,
        lr_list=[0.0005, 0.0008, 0.001, 0.003],
    )
    util.show_trace_2d(f4, w_dict, "heavyball_momentum")
    util.plot_loss(f4, loss_dict, "heavyball_momentum")


def q7():
    w_dict, loss_dict = test_momentum(
        f3,
        df3dw,
        w_init=5.0,
        max_iter=2000,
        lr_list=[0.0003, 0.0004, 0.0006, 0.001],
        optimizer_type="nestrov_momentum",
    )
    util.plot_decision_parameter(f3, w_dict, "nestrov_momentum")
    util.plot_loss(f3, loss_dict, "nestrov_momentum")


def q8():
    w_dict, loss_dict = test_momentum(
        f4,
        df4dw,
        w_init=np.array([3.0, 3.0]),
        max_iter=500,
        lr_list=[0.0005, 0.0008, 0.001, 0.003],
        optimizer_type="nestrov_momentum",
    )
    util.show_trace_2d(f4, w_dict, "nestrov_momentum")
    util.plot_loss(f4, loss_dict, "nestrov_momentum")


def q9():
    w_dict, loss_dict = test_sgd(
        f4,
        df4dw,
        w_init=np.array([3.0, 3.0]),
        max_iter=500,
        lr_list=[0.1, 0.2, 0.3, 0.5],
        optimizer_type="adam",
    )
    util.show_trace_2d(f4, w_dict, "adam")
    util.plot_loss(f4, loss_dict, "adam")


def q10():
    w_dict, loss_dict = test_sgd(
        f42,
        df42dw,
        w_init=np.array([3.0, 3.0]),
        max_iter=500,
        lr_list=[0.1, 0.2, 0.3, 0.5],
        optimizer_type="adam",
    )
    util.show_trace_2d(f42, w_dict, "adam")
    util.plot_loss(f42, loss_dict, "adam")


def q11():
    w_dict, loss_dict = test_momentum(
        f42,
        df42dw,
        w_init=np.array([3.0, 3.0]),
        max_iter=500,
        lr_list=[0.0005, 0.0008, 0.001, 0.003],
        optimizer_type="nestrov_momentum",
    )
    util.show_trace_2d(f42, w_dict, "nestrov_momentum")
    util.plot_loss(f42, loss_dict, "nestrov_momentum")


def q12():
    logr = myTorch.MultiClassLogisticRegression()
    m = np.array([[0, 0], [3, 2], [0, 1]])
    expected = np.array([[1, 0, 0], [1, 3, 2], [1, 0, 1]])
    print(
        f"Input:\n{m}\n your output:\n{logr.add_bias(m)}\n expected output:\n{expected}"
    )


def q13():
    logr = myTorch.MultiClassLogisticRegression()

    y1 = ["black", "red", "yellow", "yellow", "red"]
    y = y1
    logr.classes = logr.unique_classes_(y)
    logr.class_labels = logr.class_labels_(logr.classes)
    print(
        f"Input: {y}\n expected classes: ['black' 'red' 'yellow']\n returned classes: {logr.classes}\n expected class_labels: {{'black': 0, 'red': 1, 'yellow': 2}}\n returned class_labels: {logr.class_labels}\n Expected one-hot-encoded:\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]\n [0. 0. 1.]\n [0. 1. 0.]]\n returned One-hot-encoded:\n{logr.one_hot(y)}"
    )


def q14():
    logr = myTorch.MultiClassLogisticRegression()

    z = np.array([[-1, -1, -1], [-100, -100, 100], [1, 1 + np.log(2), 1 + np.log(2)]])
    print(f"Input z:\n[[-1, -1, -1],\n [-100, -100, 100],\n [1,   1+ln(2),  1+ln(2)]]")
    print(
        f"Expected output (the displayed expected output is rounded for better illustration):\n[[1/3 1/3 1/3]\n [0.00 0.00 1.00]\n [0.20 0.40 0.40]]"
    )
    print(f"Returned output:\n{logr.softmax(z)}")


def q15():
    logr = myTorch.MultiClassLogisticRegression()
    logr.weights = np.array([[0, 1, 0], [0, 1, -2], [0, 2, 0]])
    X_aug = np.array([[1, 0, 1], [1, 1, 0], [1, 2, 1], [1, -1, -1]])
    print(f"Model's weight:\n{logr.weights}")
    print(f"Input augmented X:\n{X_aug}")
    print(
        f"Expected output (the displayed expected output is rounded for better illustration):\n[[0.46..., 0.063..., 0.46...],\n [0.21.., 0.21..., 0.57...],\n [0.11..., 0.015..., 0.86...],\n [0.11... , 0.84..., 0.042...]]"
    )
    print(f"Returned output:\n{logr.predict_with_X_aug_(X_aug)}")


def q16():
    logr = myTorch.MultiClassLogisticRegression()
    logr.weights = np.array([[0, 1, 0], [0, 1, -2], [0, 2, 0]])
    X = np.array([[0, 1], [1, 0], [2, 1], [-1, -1]])
    print(f"Model's weight:\n{logr.weights}")
    print(f"Input X:\n{X}")
    print(
        f"Expected output (the displayed expected output is rounded for better illustration):\n[[0.46..., 0.063..., 0.46...],\n [0.21.., 0.21..., 0.57...],\n [0.11..., 0.015..., 0.86...],\n [0.11... , 0.84..., 0.042...]]"
    )
    print(f"Returned output:\n{logr.predict(X)}")


def q17():
    logr = myTorch.MultiClassLogisticRegression()
    y = [1, 5, 15, 1, 5, 5, 5, 15, 15, 1, 15, 1]
    logr.classes = logr.unique_classes_(y)
    logr.class_labels = logr.class_labels_(logr.classes)

    logr.weights = np.array([[0, 1, 0], [0, 1, -2], [0, 2, 0]])
    X = np.array([[0, 1], [1, 0], [2, 1], [-1, -1]])

    y_hat = logr.predict_classes(X)

    print(f"Model's unique classes: {logr.classes}")
    print(f"Model's class labels: {logr.class_labels}")
    print(f"Model's weight:\n{logr.weights}")
    print(f"Input X:\n{X}")
    print(f"Expected output (type: value):\n<class 'numpy.ndarray'>: [ 1 15 15  5]")
    print(f"Returned output (type: value):\n{type(y_hat)}: {y_hat}")


def q18():
    logr = myTorch.MultiClassLogisticRegression()
    logr.classes = logr.unique_classes_([1, 5, 15, 1, 5, 5, 5, 15, 15, 1, 15, 1])
    logr.class_labels = logr.class_labels_(logr.classes)

    logr.weights = np.array([[0, 1, 0], [0, 1, -2], [0, 2, 0]])

    X = np.array([[0, 1], [1, 0], [1, 0], [-1, -1], [1, 1]])

    y = [1, 5, 15, 1, 15]

    print(f"Model's unique classes: {logr.classes}")
    print(f"Model's class labels: {logr.class_labels}")
    print(f"Model's weight:\n{logr.weights}")
    print(f"Input X:\n{X}")
    print(f"True labels: {y}")
    print(f"Expected score: 0.6")
    print(f"Returned score: {logr.score(X, y)}")


def q19():
    logr = myTorch.MultiClassLogisticRegression()
    logr.classes = logr.unique_classes_([1, 5, 15, 1, 5, 5, 5, 15, 15, 1, 15, 1])
    logr.class_labels = logr.class_labels_(logr.classes)

    logr.weights = np.array([[0, 1, 0], [0, 1, -2], [0, 2, 0]])

    X_aug = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0], [1, -1, -1], [1, 1, 1]])

    y = [1, 5, 15, 1, 15]
    logr.y_one_hot_encoded = logr.one_hot(y)

    print(f"Model's unique classes: {logr.classes}")
    print(f"Model's class labels: {logr.class_labels}")
    print(f"Model's weight:\n{logr.weights}")
    print(f"Augmented Input:\n{X_aug}")
    print(f"True labels: {y}")
    print(f"One-hot-encoded True labels:\n{logr.y_one_hot_encoded}")
    print(f"Expected output: 0.6")
    print(f"Returned output: {logr.evaluate_(X_aug, logr.y_one_hot_encoded)}")


def q20():
    logr = myTorch.MultiClassLogisticRegression()
    logr.classes = logr.unique_classes_([1, 5, 15, 1, 5, 5, 5, 15, 15, 1, 15, 1])
    logr.class_labels = logr.class_labels_(logr.classes)

    logr.weights = np.array([[0, 1, 0], [0, 1, -2], [0, 2, 0]])

    X_aug = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0], [1, -1, -1], [1, 1, 1]])

    y = [1, 5, 15, 1, 15]
    logr.y_one_hot_encoded = logr.one_hot(y)

    print(f"Model's unique classes: {logr.classes}")
    print(f"Model's class labels: {logr.class_labels}")
    print(f"Model's weight:\n{logr.weights}")
    print(f"Augmented Input:\n{X_aug}")
    print(f"True labels: {y}")
    print(f"One-hot-encoded True labels:\n{logr.y_one_hot_encoded}")
    print(f"Expected gradient: 1.076...")
    print(
        f"Returned output: {logr.cross_entropy(logr.y_one_hot_encoded, logr.predict_with_X_aug_(X_aug))}"
    )


def q21():
    logr = myTorch.MultiClassLogisticRegression()
    logr.classes = logr.unique_classes_([1, 5, 15, 1, 5, 5, 5, 15, 15, 1, 15, 1])
    logr.class_labels = logr.class_labels_(logr.classes)

    logr.weights = np.array([[0, 1, 0], [0, 1, -2], [0, 2, 0]])

    X_aug = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 0], [1, -1, -1], [1, 1, 1]])

    y = [1, 5, 15, 1, 15]
    logr.y_one_hot_encoded = logr.one_hot(y)

    print(f"Model's unique classes: {logr.classes}")
    print(f"Model's class labels: {logr.class_labels}")
    print(f"Model's weight:\n{logr.weights}")
    print(f"Augmented Input:\n{X_aug}")
    print(f"True labels: {y}")
    print(f"One-hot-encoded True labels:\n{logr.y_one_hot_encoded}")
    print(f"Expected gradient:")
    print(
        f"[[-0.146...,  0.313...,  0.122...],\n [ 0.073..., -0.276..., -0.149...],\n [ 0.073..., -0.036...,  0.026... ]]"
    )
    print(
        f"Returned output:\n{logr.compute_grad(X_aug, logr.y_one_hot_encoded, logr.weights)}"
    )


def q22():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2414
    )

    n_iter = 4000
    learning_rate = 0.02
    lr = util.train_model(
        X_train, X_test, y_train, y_test, lr=learning_rate, n_iter=n_iter
    )
    lr_moment = util.train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        lr=learning_rate,
        optimizer="heavyball_momentum",
        n_iter=n_iter,
    )
    lr_nag = util.train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        lr=learning_rate,
        optimizer="nestrov_momentum",
        n_iter=n_iter,
    )
    lr_adam = util.train_model(
        X_train, X_test, y_train, y_test, optimizer="adam", lr=0.02, n_iter=n_iter
    )

    aux = plt.subplot(111)
    plt.title("Loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.plot(
        np.arange(len(lr.loss)),
        lr.loss,
        "r--",
        lr_moment.loss,
        "b--",
        lr_nag.loss,
        "g--",
        lr_adam.loss,
        "y--",
    )
    aux.legend(["SGD", "heavyball momentum", "nestrov momentum", "adam"])


def q23():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_iter = 4000
    learning_rate = 0.005
    lr = util.train_model(
        X_train, X_test, y_train, y_test, lr=learning_rate, n_iter=n_iter
    )
    lr_moment = util.train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        lr=learning_rate,
        optimizer="heavyball_momentum",
        n_iter=n_iter,
    )
    lr_nag = util.train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        lr=learning_rate,
        optimizer="nestrov_momentum",
        n_iter=n_iter,
    )
    lr_adam = util.train_model(
        X_train, X_test, y_train, y_test, optimizer="adam", lr=0.01, n_iter=n_iter
    )

    aux = plt.subplot(111)
    plt.title("Loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.plot(
        np.arange(len(lr.loss)),
        lr.loss,
        "r--",
        lr_moment.loss,
        "b--",
        lr_nag.loss,
        "g--",
        lr_adam.loss,
        "y--",
    )
    aux.legend(["SGD", "heavyball momentum", "nestrov momentum", "adam"])


def test_sgd(
    f,
    dfdw,
    w_init=5.0,
    max_iter=10000,
    lr_list=[0.001, 0.005, 0.01, 0.05],
    optimizer_type="sgd",
):
    update_thres = 1e-6
    w_dict = {}
    loss_dict = {}
    for k, lr in enumerate(lr_list):
        optimizer = myTorch.Optimizer(optimizer_type, lr=lr)
        i = 0
        update = 0
        loss_dict[lr] = []
        w = np.copy(w_init)
        w_dict[lr] = []
        w_dict[lr].append(np.copy(w))
        while i < max_iter:
            loss_dict[lr].append(f(w))
            gradient = dfdw(w)
            update = optimizer.optimize(gradient)
            w += update
            w_dict[lr].append(np.copy(w))
            if np.abs(update).max() < update_thres:
                break
            i += 1
        try:
            display(
                Markdown(
                    f"* With learning rate {lr:0.4f}: $w^\star=$ {w:0.5f}, $f(w^\star)=$ {f(w):0.5f} and $\\nabla f(w^\star)=$ {dfdw(w):0.5f}, and converged after {i} iterations"
                )
            )
        except:
            display(
                Markdown(
                    f"* With learning rate {lr:0.4f}: $w^\star=$ {w}, $f(w^\star)=$ {f(w):0.5f} and $\\nabla f(w^\star)=$ {dfdw(w)}, and converged after {i} iterations"
                )
            )
    return w_dict, loss_dict


def test_momentum(
    f,
    dfdw,
    w_init=5.0,
    max_iter=10000,
    lr_list=[0.001, 0.005, 0.01, 0.05],
    gama=0.9,
    optimizer_type="heavyball_momentum",
):
    update_thres = 1e-6
    n_no_change_thresh = 4
    w_dict = {}
    loss_dict = {}
    for k, lr in enumerate(lr_list):
        optimizer = myTorch.Optimizer(optimizer_type, lr=lr, gama=gama)
        i = 0
        update = 0
        loss_dict[lr] = []
        w = np.copy(w_init)
        w_dict[lr] = []
        w_dict[lr].append(np.copy(w))
        n_no_change = 0
        while i < max_iter:
            loss_dict[lr].append(f(w))
            if optimizer_type == "nestrov_momentum":
                gradient = dfdw(w + gama * update)
            else:
                gradient = dfdw(w)
            update = optimizer.optimize(gradient)
            w += update
            w_dict[lr].append(np.copy(w))
            if np.abs(update).max() < update_thres:
                n_no_change += 1
            if n_no_change >= n_no_change_thresh:
                break
            i += 1
        try:
            display(
                Markdown(
                    f"* With learning rate {lr:0.4f}: $w^\star=$ {w:0.5f}, $f(w^\star)=$ {f(w):0.5f} and $\\nabla f(w^\star)=$ {dfdw(w):0.5f}, and converged after {i} iterations"
                )
            )
        except:
            display(
                Markdown(
                    f"* With learning rate {lr:0.4f}: $w^\star=$ {w}, $f(w^\star)=$ {f(w):0.5f} and $\\nabla f(w^\star)=$ {dfdw(w)}, and converged after {i} iterations"
                )
            )
    return w_dict, loss_dict
