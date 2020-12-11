"""
    This is a basic neural networks implementation using only numpy external lib.
    The algorithm used is 'stochastic gradient descent' which uses gradient descent for the optimization.
    Gradient of cost function w.r.t weights and biases are calculated using backpropogation algorithm.

    This code is studied from book written by Michael Nielsen (http://neuralnetworksanddeeplearning.com/)
"""

import numpy as np
import random
from nnets.lib.utils import Utils



def main ():
    '''
    :return:
    '''
    sizes = [784, 30, 10]
    epochs = 25
    mini_batch_size = 10
    eta = 3.0
    data_path = "../data/mnist.pkl.gz"
    train_data, valid_data, test_data = Utils.load_data_wrapper(data_path)
    net = Network(sizes)
    net.SGD(train_data, epochs, mini_batch_size, eta, test_data=test_data)


class Network:

    def __init__(self, sizes):
        '''
        :param sizes: For ex=> [2,3,1] 3 layers of 2,3,1 activation neurons
        biases = [(3x1),(1x1] # list of vectors
        weights = [(3x2), (1x3)] # list of vectors
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        '''
        :param training_data: zip of {v(784 x 50000), v(10 x 50000)}
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param test_data: zip of {v(784 x 10000), v(10 x 10000)}
        :return:
        '''
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
        :param mini_batch:
        :param eta:
        :return:
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = Utils.backprop(self, x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # np.argmax () returns max element index
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        # print(type(test_results))
        # print(type(test_results[0]))
        # print(test_results)
        # num = 0
        # for x, y in test_results:
        #     if x == y:
        #         num = num+1
        # print(num)
        num = sum(int(x == y) for (x, y) in test_results)
        return num

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = Utils.sigmoid(np.dot(w, a)+b)
        return a







if __name__ == "__main__":
	main()




