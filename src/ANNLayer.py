import numpy as np


class ANNLayer():

    def __init__(self, f_in, f_out, activ_f, is_output=False,
                 loss_f='hinge',
                 optimizer='adam',
                 learning_rate=0.01,
                 beta1=0.9,
                 beta2=0.99):

        self.activ_f = activ_f
        self.is_output = is_output
        self.loss_t = loss_f
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.mom = 0
        self.acc = 0

        self.W = np.random.randn(f_out, f_in)/np.sqrt(f_in)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, A_prev)

        if self.loss_t == 'nll':
            self.A = soft_max(self.Z)
            return self.A

        return self.activ_f(self.Z)

    def backward(self, dLdA, y=None):
        if self.loss_t == 'nll':
            dLdZ = soft_max(self.A, dLdA, df=True)
        else:
            dAdZ = self.activ_f(self.Z, df=True)
            dLdZ = dLdA*dAdZ

        m = self.A_prev.shape[1]
        dW = np.dot(dLdZ, self.A_prev.T)/m
        dA_prev = np.dot(self.W.T, dLdZ)

        self.update_weights(dW)
        return dW, dA_prev

    def update_weights(self, grad):

        if self.optimizer == 'basic':
            self.W -= self.learning_rate * grad
            return

        # Adam Optimizer
        self.mom = self.beta1*self.mom + (1-self.beta1)*grad
        self.acc = self.beta2*self.acc + \
            (1-self.beta2)*(grad*grad).sum()

        self.W -= self.learning_rate * \
            self.mom / (np.sqrt(self.acc) + 1e-7)


def soft_max(A, dLdA=None, df=False):
    if df:
        dA = np.ndarray(shape=A.shape)
        for bidx in range(A.shape[1]):
            bA = A[:, bidx]
            bdLdA = dLdA[:, bidx]
            m = A.shape[0]
            bA = (bA*np.ones(shape=(m, m))).T
            bA = bA * (np.identity(m) - bA.T)
            dAi = np.dot(bA, bdLdA)
            dA[:, bidx] = dAi
        return dA

    A = A - np.max(A, axis=0)[:, np.newaxis].T
    A_exp = np.exp(A)
    exp_sum = np.sum(A_exp.T, axis=1)
    exp_sum = np.array([exp_sum] * A_exp.shape[0])
    return A_exp/exp_sum

