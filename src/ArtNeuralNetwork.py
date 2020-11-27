# -*- coding: utf-8 -*-
# %%
from collections import Counter
from random import random, seed

import numpy as np
from tqdm import tqdm

import ANNLayer as annl

DEBUG = False


class ArtNeuralNetwork(object):
    """[summary]
        Artificial Neural Network
    """

    def __init__(self,
                 topology=[],
                 batch_size=32,
                 max_epochs=300,
                 learning_rate=0.01,
                 beta1=0.9,
                 beta2=0.99,
                 activation='relu',
                 loss_t="hinge",
                 optimizer='adam',
                 X_val=None,
                 y_val=None,
                 best_per_epoch=True,
                 ):
        self.topology = topology
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.X_val = X_val
        self.y_val = y_val
        self.best_per_epoch = best_per_epoch

        self.alpha = 0.1
        self.margin = 0.1
        self.best_accr = -1

        if activation == 'relu':
            self.activ_f = leaky_ReLU_meta(self.alpha)
        else:
            self.activ_f = sigmoid_meta()

        if loss_t == "hinge":
            self.loss_f = hinge_loss_meta(self.margin)
        else:
            self.loss_f = nll_loss_meta()

        self.loss_t = loss_t
        self.optimizer = optimizer

    def init_layers(self):
        D = len(self.topology)
        self.D = D - 1
        self.layers = []
        for i in range(1, D):
            f_in = self.topology[i-1]
            f_out = self.topology[i]
            layer = annl.ANNLayer(f_in, f_out, self.activ_f,
                                  optimizer=self.optimizer,
                                  learning_rate=self.learning_rate,
                                  beta1=self.beta1,
                                  beta2=self.beta2)
            self.layers.append(layer)
        # Output layer
        self.layers[-1].set_params(is_output=True, loss_t=self.loss_t)

    def save_best_state(self):
        self.best_layer_states = [None]*len(self.layers)
        for i, layer in enumerate(self.layers):
            layer_copy = layer
            layer_copy.W = np.array(layer.W, copy=True)
            self.best_layer_states[i] = layer_copy

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        self.init_layers()
        tr_losses = []
        val_losses = []
        val_accrs = []
        for i in tqdm(range(self.max_epochs), desc="ANN Training"):
            bX, bY = self.batch_sample(X, y)
            self.apply_SGD_step(bX, bY)

            accr_tr = self.accuracy_raw(bX, bY)
            accr_val = self.accuracy_raw(self.X_val, self.y_val)

            loss_tr = self.loss_f(self.forward_pass(bX), bY)
            loss_tr = np.average(loss_tr)
            loss_val = self.loss_f(self.forward_pass(self.X_val), self.y_val)
            loss_val = np.average(loss_val)

            tr_losses.append(loss_tr)
            val_losses.append(loss_val)
            val_accrs.append(accr_val)

            if accr_val > self.best_accr:
                self.save_best_state()
            if DEBUG:
                print('Epoch %d: train_accr = %f, val_accr = %f, avg_loss_tr = %f, avg_loss_val = %f' %
                      (i, accr_tr, accr_val, loss_tr, loss_val))

        if self.best_per_epoch:
            self.layers = self.best_layer_states
        return tr_losses, val_losses, val_accrs

    def forward_pass(self, X):
        f_feed = X.T
        for i in range(self.D):
            f_feed = self.layers[i].forward(f_feed)

        return f_feed

    def backward_pass(self, d_loss, y):
        b_feed = d_loss
        grad = [None] * self.D
        for i in reversed(range(self.D)):
            w_grads, b_feed = self.layers[i].backward(b_feed, y)
            grad[i] = w_grads

        return grad

    def apply_SGD_step(self, bX, bY):
        output_activ = self.forward_pass(bX)
        d_loss = self.loss_f(output_activ, bY, df=True)
        self.backward_pass(d_loss, bY)

    def batch_sample(self, X, y):
        b_idx = np.random.choice(
            X.shape[0], size=self.batch_size, replace=False)
        return X[b_idx, :], y[b_idx]

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        Ypred = np.zeros(num_test)
        scores = self.forward_pass(X)
        Ypred = np.argmax(scores.T, axis=1)

        return Ypred

    def accuracy(self, predY, y):
        return sum(predY == y)/y.shape[0]

    def accuracy_raw(self, X, y):
        predY = self.predict(X)
        return sum(predY == y)/y.shape[0]

    def detailed_score(self, predY, y, num_of_class=5):
        CR = [0]*num_of_class
        TCR = [0]*num_of_class
        for pY, cY in zip(predY, y):
            TCR[cY] = TCR[cY] + 1
            CR[cY] = CR[cY] + (pY == cY)
        CCRn = [cr/tcr for (cr, tcr) in zip(CR, TCR)]
        self.accR = sum(CR)/sum(TCR)
        return CCRn, self.accR

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def leaky_ReLU(X, alpha, df=False):
    if df:
        dX = np.ones(shape=X.shape)
        dX[X < 0] *= alpha
        return dX
    return np.maximum(alpha*X, X)


def leaky_ReLU_meta(alpha):
    return lambda X, df=False: leaky_ReLU(X, alpha, df)


def sigmoid(X, df=False):
    sig = 1/(1 + np.exp(-X))
    if df:
        return sig*(1-sig)
    return sig


def sigmoid_meta():
    return lambda X, df=False: sigmoid(X, df)


def nll_loss(S, y, df=False):
    sample_inds = list(range(len(y)))
    if df:
        hot_enc = np.zeros(shape=S.shape)
        hot_enc[y, sample_inds] = 1
        return S - hot_enc

    p = np.exp(S[y, sample_inds])/np.sum(np.exp(S[:, sample_inds]), axis=0)
    loss = -np.log(p)
    return loss


def nll_loss_meta():
    return lambda S, y, df=False: nll_loss(S, y, df)


def hinge_loss(S, y, margin=0, df=False):
    if df:
        sample_inds = list(range(len(y)))
        losses = 1*(S - S[y, sample_inds] + margin > 0)
        losses[y, sample_inds] = -1 * \
            np.sum(losses[:, sample_inds], axis=0) + margin
        return losses

    sample_inds = list(range(len(y)))
    losses = np.maximum(0, S - S[y, sample_inds] + margin)
    losses[y, sample_inds] = 0
    return losses


def hinge_loss_meta(margin):
    return lambda S, y, df=False: hinge_loss(S, y, margin, df)
