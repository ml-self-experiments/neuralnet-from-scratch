"""
This is a helper library used by src modules.
"""
import numpy as np
import gzip
import pickle

class Utils:

    @classmethod
    def backprop(cls, net, x, y):
        '''
        :param net: 'Network' object
        :param x: (input_neurons x 1)
        :param y: (actual_output_neuronsx 1)
        :return: (updated_bias, updated_weights) In Network.biases , Network.weights format
        '''
        nabla_b = [np.zeros(b.shape) for b in net.biases]
        nabla_w = [np.zeros(w.shape) for w in net.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(net.biases, net.weights):
            z = np.dot(w, activation ) +b
            zs.append(z)
            activation = Utils.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = Utils.cost_derivative(activations[-1], y) * \
                Utils.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, net.num_layers):
            z = zs[-l]
            sp = Utils.sigmoid_prime(z)
            delta = np.dot(net.weights[- l +1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[- l -1].transpose())
        return (nabla_b, nabla_w)

    @classmethod
    def sigmoid(cls, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def sigmoid_prime(cls, z):
        """Derivative of the sigmoid function."""
        return Utils.sigmoid(z) * (1 - Utils.sigmoid(z))

    @classmethod
    def cost_derivative(cls, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    @classmethod
    def load_data_wrapper(cls, data_path):
        '''
        :param data_path: pickle gzip path
        :return:    train_data = [(784x1, 10x1),
                                  (784x1, 10x1),
                                  ... 50000 rows]
                    validation_data and test_data are of same structure.
                    test_data = [(784x1, 10x1),
                                  (784x1, 10x1),
                                  ... 10000 rows]
        '''
        tr_data, val, tst_data = Utils.load_data(data_path)
        # print (np.shape(tr_data[0][:10,:]))
        train_input = [np.reshape(x, (784, 1)) for x in tr_data[0]]
        train_label = [Utils.vectorized_result(y) for y in tr_data[1]]
        train_data = list(zip(train_input, train_label))
        # print(type(train_data))
        # print(len(train_data))
        # for x in train_data:
        #     print(type(x))
        #     print(len(x))
        #     print(type(x[0]))
        #     print(len(x[0]))
        #     print(type(x[1]))
        #     print(len(x[1]))
        #     break

        valid_input = [np.reshape(x, (784, 1)) for x in val[0]]
        valid_label = [Utils.vectorized_result(y) for y in val[1]]
        validation_data = list(zip(valid_input, valid_label))

        test_input = [np.reshape(x, (784, 1)) for x in tst_data[0]]
        test_label = [Utils.vectorized_result(y) for y in tst_data[1]]
        test_data = list(zip(test_input, test_label))
        return (train_data, validation_data, test_data)

    @classmethod
    def vectorized_result(cls, j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    @classmethod
    def load_data(cls, data_path):
        '''
        :param data_path:
        :return:    training_data = [v(50000 x 748), v(50000 x 1)]
                    validation_data = [v(10000 x 748), v(10000 x 1)]
                    test_data = [v(10000 x 748), v(10000 x 1)]
        '''
        f = gzip.open(data_path, 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        f.close()
        return (training_data, validation_data, test_data)