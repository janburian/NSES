import numpy as np
from pathlib import Path
import random
from matplotlib import pyplot as plt


def load_data(data_path: Path):
    with open(data_path, 'r') as file:
        lines = file.readlines()
        data = []

        for line in lines:
            # Split the line into a list of strings using space as a separator
            number_strings_list = line.split()

            # Convert the strings to floats and store in a list
            numbers = [float(num) for num in number_strings_list]

            # Append the list of numbers to the data
            data.append(numbers)

    file.close()

    return data


def split_data_train_test(data: list):
    # Splitting data (80% training, 20% testing)
    split_index = int(0.8 * len(data))

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    return train_data, test_data


def initialize_parameters():
    W = np.random.randn(5, 2)  # initializing random weights (dim output = 5, dim input vector = 2)
    b = np.random.randn(5, 1)  # bias

    E_max = 1  # maximal error
    lr = 0.01
    num_epochs = 10

    return W, b, E_max, lr, num_epochs


def get_output_vector(u: float):
    # Bipolar
    output_vec = np.array([-1, -1, -1, -1, -1])
    output_vec[int(u)-1] = 1

    return output_vec.reshape(-1, 1)


def train_one_layered_nn(train_data: list):
    W, b, E_max, lr, num_epochs = initialize_parameters()

    errors = []
    E = 0
    for i in range(num_epochs):
        for k in range(len(train_data)):
            input_output = train_data[k]
            x = input_output[0:2]   # input vector
            u = input_output[2]  # output (target - only index)
            u_vec = get_output_vector(u)

            # Counting output
            x = np.array(x).reshape(-1, 1)
            xi = np.dot(W, x) + b
            y = np.sign(xi)

            # Counting error
            E = E + np.dot(1/2, np.dot((u_vec - y).reshape(1, -1), (u_vec - y)))

            # Modification of the weights and biases
            W = W + lr * (u_vec - y) * x.reshape(1, -1)
            b = b + lr * (u_vec - y)

        errors.append(int(E))

        if E <= E_max:
            break
        else:
            E = 0
            random.shuffle(train_data)

    return W, b, errors


def do_inference(W: np.array, b: np.array, test_data: list):
    E = 0
    num_correct = 0

    for k in range(len(test_data)):
        input_output = test_data[k]
        x = input_output[0:2]  # input vector
        u = input_output[2]  # output (target - only index)
        u_vec = get_output_vector(u)

        # Counting output
        x = np.array(x).reshape(-1, 1)
        xi = np.dot(W, x) + b
        y = np.sign(xi)

        # Counting error
        E = E + np.dot(1 / 2, np.dot((u_vec - y).reshape(1, -1), (u_vec - y)))

        if (y == u_vec).all():
            num_correct += 1

        visualize_points_clusters(x, y)

    accuracy = num_correct / len(test_data)
    plt.grid()
    plt.show()

    return int(E), accuracy


def visualize_points_clusters(x: np.array, y: np.array):
    vector = x
    colors = ["red", "green", "blue", "purple", "orange"]
    color_idx = [i for i, x in enumerate(y) if x > 0][0]
    plt.scatter(vector[0], vector[1], c=colors[color_idx])
    plt.title('Clusters (test data)')
    plt.xlabel('x')
    plt.ylabel('y')


def visualize_errors(errors: list):
    plt.plot(range(len(errors)), errors)
    # plt.plot(range(len(errors)), errors, 'bo')
    plt.xticks(range(0, len(errors)))
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid()
    plt.show()


def visualize_data(data: list):
    for i in range(len(data)):
        vector = data[i][0:2]
        plt.scatter(vector[0], vector[1])

    plt.title('Data visualisation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


def visualize_data_histogram(data: list):
    class_dict = {"1.0": 0, "2.0": 0, "3.0": 0, "4.0": 0, "5.0": 0}
    for i in range(len(data)):
        target = str(data[i][2])
        class_dict[target] += 1

    # Extract keys and values
    keys = list(class_dict.keys())
    values = list(class_dict.values())

    # Plotting the histogram
    plt.bar(keys, values)

    # Adding labels and title
    plt.xlabel('Classes')
    plt.ylabel('n')
    plt.title('Data histogram')

    # Display the plot
    plt.show()


def get_xy_min_max(data: list):
    x_values = []
    y_values = []

    for vector in data:
        x_values.append(vector[0])
        y_values.append(vector[1])

    return min(x_values), max(x_values), min(y_values), max(y_values)


def visualize_result(data: list, W: np.array, b: np.array, data_type: str):
    x_min, x_max, y_min, y_max = get_xy_min_max(data)

    for i in range(len(data)):
        vector = data[i][0:2]
        plt.scatter(vector[0], vector[1])

    if data_type == "train":
        plt.title('Result (train data)')
    else:
        plt.title('Result (test data)')

    x = np.linspace(x_min - 1, x_max + 1)
    for i in range(len(W)):
        y = (W[i, 0] * x + b[i]) / (-W[i, 1])
        plt.plot(x, y)

    plt.xlim([x_min - 1, x_max + 1])
    plt.ylim([y_min - 1, y_max + 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    data_path = Path("./data/tren_data1___03.txt")

    data = load_data(data_path)
    visualize_data(data)
    visualize_data_histogram(data)
    random.shuffle(data)
    train_data, test_data = split_data_train_test(data)

    W, b, errors = train_one_layered_nn(train_data)
    visualize_errors(errors)

    test_error, test_accuracy = do_inference(W, b, test_data)
    print(f"Test accuracy: {test_accuracy * 100} %")

    visualize_result(train_data, W, b, "train")
    visualize_result(test_data, W, b, "test")
