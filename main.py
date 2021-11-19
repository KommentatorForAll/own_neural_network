from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


def transform_data(data: pd.DataFrame,
                   output_nodes: Optional[int] = None,
                   lower_bound: float = 0.01,
                   upper_bound: float = 0.99
                   ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    targets: Optional[pd.DataFrame] = None
    if output_nodes is not None:
        # as the first column is the target digit we are looking for
        target_number = data.pop(0)
        targets = np.zeros((len(data), output_nodes)) + lower_bound
        for i in range(len(target_number)):
            targets[i][target_number.loc[i]] = upper_bound
        # targets[target_number.to_numpy()] = upper_bound
    # scaling the inputs to [0.01; 0.99]
    data = data / 255 * upper_bound + lower_bound
    return data, targets


def main():
    # names, because else it would take the first as the header row
    train_data = pd.read_csv('./mnist_train.csv', names=list(range(785)))
    test_data = pd.read_csv('./mnist_test.csv', names=list(range(785)))  # .loc[:100]

    # 784 inputs as we have a 28 x 28 image
    input_nodes = 784
    # 100 hidden nodes bc why not
    hidden_nodes = 200
    # 10 output nodes bc there are 10 digits
    output_nodes = 10
    # 0.1 as a learn rate
    learn_rate = 0.1
    # amount of epochs we are training
    epochs = 3

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)

    train_data, targets = transform_data(train_data, output_nodes)

    for epoch in range(epochs):
        nn.train(train_data, targets)

    test_targets = test_data.pop(0)
    test_data, _ = transform_data(test_data)

    results = nn.query(test_data)
    results = [np.argmax(res) for res in results]
    matches: pd.DataFrame = test_targets == results
    print(matches.value_counts())
    print(results)
    accuracy = matches.sum()/len(matches)
    print(f'accuracy: {accuracy}\nerror: {1-accuracy}')
    # for i in range(len(results)):
    #    img_array = test_data.loc[i].to_numpy().reshape((28, 28))
    #    plt.imshow(img_array, cmap='Greys', interpolation='None')
    #    plt.show()
    #    print(f'result: {results[i]}; actual: {test_targets[i]}')


if __name__ == '__main__':
    main()
