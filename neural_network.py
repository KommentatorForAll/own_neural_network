import numpy as np
import scipy.special


class NeuralNetwork:

    def __init__(self,
                 input_nodes,
                 hidden_nodes,
                 target_nodes,
                 learn_rate,
                 activation_function=scipy.special.expit
                 ):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = target_nodes

        # weights input hidden
        self.wih = np.random.normal(0.0, self.inodes**-0.5, (self.hnodes, self.inodes))
        # weights hidden output
        self.who = np.random.normal(0.0, self.hnodes**-0.5, (self.onodes, self.hnodes))

        self.lr = learn_rate
        self.activation_function = activation_function

    def train(self, train_input, target_output):
        inputs = np.array(train_input, ndmin=2).T
        targets = np.array(target_output, ndmin=2).T

        # Transforming the input with the input weights to the hidden layer input
        hidden_inputs = np.dot(self.wih, inputs)

        # applying the hidden nodes to the inputs
        hidden_output = self.activation_function(hidden_inputs)

        # Processing the final layer
        final_inputs = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_inputs)

        # Calculate the error
        output_error = targets - final_output
        hidden_errors = np.dot(self.who.T, output_error)

        self.who += self.lr * np.dot((output_error * final_output * (1.0 - final_output)), np.transpose(hidden_output))
        self.wih += self.lr * np.dot((hidden_errors * hidden_output * (1.0 - hidden_output)), np.transpose(inputs))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
