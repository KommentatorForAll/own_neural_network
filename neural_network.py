import gzip
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
import scipy.special


class Layer:
    """
    This class represents a single layer of a NeuralNetworkV2.
    """

    def __init__(self,
                 nodes: int,
                 activation_function=scipy.special.expit,
                 weights=None
                 ):
        """
        Constructor of the Layer class.
        :param nodes: The amount of nodes this layer has
        :param activation_function: The activation function, which gets applied after calculating the new values.
        Defaults to sigmoid.
        It would be preferable if only set to functions from scipy.special module,
        else saving/loading of the model might raise issues
        :param weights: Only needed when loading an existing model.
        """
        self.nodes: int = nodes
        self.activation_function: Callable = activation_function
        self.weights: Optional[np.ndarray] = weights

    def _calculate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Passes the given inputs through this layer. The activation function is applied automatically.
        :param inputs: The inputs for this layer
        :return: The output of this layer
        """
        inp = np.dot(self.weights, inputs)
        return self.activation_function(inp)

    def _adjust_weights(self,
                        prev_output: np.ndarray,
                        output: np.ndarray,
                        error: np.ndarray,
                        learn_rate: float
                        ) -> None:
        """
        Adjusts the weights of the input connections by the neurons error.
        :param prev_output: The output of the previous layer
        :param output: The output of this layer
        :param error: The error values for each neuron
        :param learn_rate: The learn rate of the layer
        :return: None
        """
        self.weights += learn_rate * np.dot((error * output * (1.0 - output)), np.transpose(prev_output))


class NeuralNetworkV2:
    """
    The second version of the neural network.
    This version is able to handle n layers, each with custom activation functions and able to be saved and loaded
    """

    def __init__(self,
                 layers: List[Layer] = None,
                 learn_rate: float = 0.2,
                 path_or_buf: Optional[str] = None,
                 compress: bool = True
                 ):
        """
        Creates a new Neural network.
        Either layers xor path_or_buf must be given, else an error is thrown.
        You have to either create a new model, or load an existing one. Doing both or nothing is not possible.
        :param layers: A list of layers the neural network has. Must be instances of the Layer class
        :param learn_rate: Learn rate of the model. should be in range [0;1] to work
        :param path_or_buf: The path or buffer of an existing model, which gets loaded instead.
        :param compress: If the existing model is stored in a compressed format (.gz)
        """
        # Ensure valid parameters
        assert (layers is not None) != (path_or_buf is not None), ValueError('Either layers or network_str must be set')
        assert 0 < learn_rate <= 1, ValueError('learn_rate must be in range 0,1 (0 exclusive)')

        self.learn_rate: float = 0.2
        self.layers: List[Layer] = [] if layers is None else layers
        if path_or_buf is not None:
            # Loads existing model
            self._load(path_or_buf, compress)
            return
        self.learn_rate = learn_rate
        for i, layer in enumerate(self.layers[1:]):
            if layer.weights is None:
                # Assigns normal distributed random weights for all layers as an initial starting point.
                layer.weights = np.random.normal(0.0, self.layers[i].nodes ** -0.5, (layer.nodes, self.layers[i].nodes))

    def _load(self, path_or_buf: str, compress: bool = True) -> None:
        """
        Loads an already existing model.
        :param path_or_buf: The filepath of the model.
        :param compress: If the model file is compressed. Defaults to True
        :return: None
        """
        # Open the file either compressed or plain
        if compress:
            file = gzip.open(path_or_buf, 'rb')
        else:
            file = open(path_or_buf, 'rb')
        # read the content of the file. \r has to be replaced bc windows decodes linefeeds to \r\n instead of just \n
        content = file.read().decode('utf-8')\
            .replace('\r', '')\
            .split('\neof\n\n\n')
        file.close()

        # Plain part of the file. All information which is not the weights
        plain = content[0]
        # The weights, saved as np.ndarray.savetxt()
        weight_strs = content[1:]

        text = plain.split('\n')
        self.learn_rate = float(text[0])

        layer_information = text[1:-1]
        layer_cnt = len(layer_information)

        # reads all layers
        for i in range(layer_cnt):
            layer = layer_information[i].split(',')
            # First layer has no weight, therefore it is ignored
            if i != 0:
                # Temporarily write layer into a tmp file, to be loaded from numpy, bc one cannot create from string
                with open('tmp.tmp', 'w') as f:
                    f.write(weight_strs[i-1])
                # Load weights as np.ndarray
                weights = np.loadtxt('tmp.tmp')
            else:
                weights = None

            # Get the number of nodes
            nodes = int(layer[0])
            # Parse the activation function. Might cause issues with non-scipy functions.
            act_func_name = layer[1]
            try:
                act_func = eval(f'scipy.special.{act_func_name}')
            except AttributeError:
                act_func = eval(f'{act_func_name}')

            # Add the layer to the neural network.
            self.layers.append(Layer(nodes, act_func, weights))

    def train(self, inputs: pd.DataFrame, targets: pd.DataFrame) -> None:
        """
        Train a model on the given dataset.
        :param inputs: The input information
        :param targets: The target values
        :return: None
        """
        for i in range(len(inputs)):
            self._train(inputs.loc[i], targets[i])
            # Debug statement to keep track of progress
            if i % 5000 == 0:
                print(f'finished {i} data points')
        print(f'Finished training')

    def _train(self, train_input, target_output) -> None:
        """
        Trains the model on a single datapoint.
        :param train_input: The given datapoint
        :param target_output: The target value for the datapoint
        :return: None
        """
        inputs: List[np.ndarray] = [np.array(train_input, ndmin=2)]
        targets = np.array(target_output, ndmin=2).T
        # Queries the current prediction of the datapoint
        inputs = self._query(inputs[-1])
        # Calculates the error of the prediction
        output_error = targets - inputs[-1]
        # Iterates all layers backwards, adjusting the weights on the given error
        for i, layer in list(enumerate(self.layers[1:]))[::-1]:
            # Temp var for layer weights because they are being adjusted and should be the old ones for error forwarding
            layer_weights_tmp = layer.weights
            layer._adjust_weights(inputs[i], inputs[i + 1], output_error, self.learn_rate)
            output_error = np.dot(layer_weights_tmp.T, output_error)

    def query(self, inputs: pd.DataFrame) -> List[np.ndarray]:
        """
        Predicts Values for the given dataset
        :param inputs: The dataset to predict
        :return: a list of predictions
        """
        results = []
        for i in range(len(inputs)):
            # The [-1] is needed because only the prediction is returned and not the whole layer output chain
            results.append(self._query(inputs.loc[i])[-1])
        return results

    def _query(self, input_list) -> List[np.ndarray]:
        """
        Predicts a single datapoint
        :param input_list: The datapoint
        :return: The layer outputs.
        This is the whole chain, if only the prediction is needed use [-1] on the return value
        """
        inputs = [np.array(input_list, ndmin=2).T]
        for layer in self.layers[1:]:
            # Calculates the layer output using the last element aka the output of the previous layer as its input
            inputs.append(layer._calculate(inputs[-1]))
        return inputs

    def score(self, train_input, target_output, result_mod_function=None):
        """
        Returns the accuracy of the model for the given data
        :param train_input: The test input data
        :param target_output: the expected result for the input data
        :param result_mod_function: A function to transform the data before matching it with the expected results.
        e.g. np.argmax
        :return: the score of teh model
        """
        results = self.query(train_input)
        if result_mod_function is not None:
            results = [result_mod_function(res) for res in results]
        print(f'type: {type(results)}; res: {results}')
        print(f'type: {type(target_output)}; res: {target_output}')
        matches = results == target_output
        print(f'type: {type(matches)}; res: {matches}')
        return matches.sum() / len(matches)

    def save(self, file_name, compress=True) -> None:
        """
        Saves the model to a file.
        :param file_name: The file to save to.
        :param compress: If the model should be compressed using gzip
        :return: None
        """
        # All information which is not layer specific, + the first layer, as it has no weights
        network_str = f'{self.learn_rate}\n' \
                      f'{self.layers[0].nodes},{self.layers[0].activation_function.__name__}'
        tmp_file = 'tmp.tmp'
        weights = ''
        # Iterates through all layers saving their weights and information
        for layer in self.layers[1:]:
            # Uses numpy s save function to save the weights.
            # This is needed bc numpy cannot convert strings to arrays and vice versa
            np.savetxt(tmp_file, layer.weights)
            # Reads out the saved array
            with open(tmp_file, 'r') as f:
                weights += f.read()
            # appends a suffix to the layer weights, that it can be split for the other layers
            weights += '\neof\n\n\n'
            network_str += f"{layer.nodes},{layer.activation_function.__name__}\n"
        # opens the given filename in either compressed or uncompressed
        if compress:
            out = gzip.open(file_name, 'wb')
        else:
            out = open(file_name, 'wb')
        # Writes information to the file
        out.write(bytes(network_str, 'utf-8'))
        out.write(bytes('\neof\n\n\n', 'utf-8'))
        out.write(bytes(weights, 'utf-8'))

        out.close()


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
        self.wih = np.random.normal(0.0, self.inodes ** -0.5, (self.hnodes, self.inodes))
        # weights hidden output
        self.who = np.random.normal(0.0, self.hnodes ** -0.5, (self.onodes, self.hnodes))

        self.lr = learn_rate
        self.activation_function = activation_function

    def train(self, inputs: pd.DataFrame, targets: pd.DataFrame):
        for i in range(len(inputs)):
            self._train(inputs.loc[i], targets[i])
            if i % 5000 == 0:
                print(f'finished {i} data points')

    def _train(self, train_input, target_output):
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

    def query(self, inputs: pd.DataFrame) -> List[np.ndarray]:
        results = []
        for i in range(len(inputs)):
            results.append(self._query(inputs.loc[i]))
        return results

    def _query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
