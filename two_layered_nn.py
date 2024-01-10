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


def visualize_data(data):
    for i in range(len(data)):
        vector = data[i][0:2]
        plt.scatter(vector[0], vector[1])

    plt.title('Data visualisation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


def visualize_data_histogram(data):
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


def initialize_parameters():
    num_neurons = 4  # K parameter
    W = np.random.randn(num_neurons, 2)  # initializing random weights (dim output = 5, dim input vector = 2)
    V = np.random.randn(5, num_neurons)

    b = np.random.randn(num_neurons, 1)  # bias
    d = np.random.randn(5, 1)

    E_max = 1  # maximal error
    lr = 0.01
    num_epochs = 1200  # TC_max

    return W, V, b, d, E_max, lr, num_epochs

def get_output_vector(u: float):
    output_vec = np.array([0, 0, 0, 0, 0])
    output_vec[int(u)-1] = 1

    return output_vec.reshape(-1, 1)

def activation_function(x: np.array, type_act_func: str):
    if type_act_func == "sigmoid":
        return 1 / (1 + np.exp(-x))

    elif type_act_func == "ReLU":
        # return np.maximum(0, x)
        return np.where(np.asarray(x) > 0, x, 0)

def derivate_activation_function(f, activation_function: str):
    if activation_function == "sigmoid":
        return f * (1 - f)

    if activation_function == "ReLU":
        return np.where(f > 0, 1, 0)

def train_two_layered_nn(train_data: list, type_act_func: str):
    W, V, b, d, E_max, lr, num_epochs = initialize_parameters()

    errors = []
    E = 0
    for i in range(num_epochs):
        for k in range(len(train_data)):
            input_output = train_data[k]
            x = input_output[0:2]  # input vector
            u = input_output[2]  # output (target - only index)
            u_vec = get_output_vector(u)

            # Counting output
            x_k = np.array(x).reshape(-1, 1)
            z_k = activation_function((np.dot(W, x_k) + b), type_act_func)
            y_k = activation_function((np.dot(V, z_k) + d), type_act_func)
            y = y_k

            # Counting error
            E = E + np.dot(1 / 2, np.dot((u_vec - y).reshape(1, -1), (u_vec - y)))

            dz = derivate_activation_function(z_k, type_act_func)
            dy = derivate_activation_function(y_k, type_act_func)

            # Modification of the weights and biases
            W = W + np.dot(lr * ((u_vec - y) * dy * V).T, np.ones((5, 1))) * dz @ x_k.T
            b = b + np.dot(lr * ((u_vec - y) * dy * V).T, np.ones((5, 1))) * dz

            V = V + lr * (u_vec - y) * dy @ z_k.T
            d = d + lr * ((u_vec - y) * dy)

        errors.append(int(E))

        if E <= E_max:
            break
        else:
            E = 0
            random.shuffle(train_data)

    return W, V, b, d, errors

def visualize_errors(errors):
    plt.plot(range(len(errors)), errors)
    # plt.plot(range(len(errors)), errors, 'bo')
    # plt.plot(range(len(errors)), errors)
    # plt.xticks(range(1, len(errors)))
    plt.locator_params(axis='x', nbins=10)
    plt.title('Error analysis')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid()
    plt.show()

def visualize_points_clusters(x, y):
    vector = x
    colors = ["red", "green", "blue", "purple", "orange"]
    color_idx = [i for i, x in enumerate(y) if x > 0][0]
    plt.scatter(vector[0], vector[1], c=colors[color_idx])
    plt.title('Clusters (test data)')
    plt.xlabel('x')
    plt.ylabel('y')

def do_inference(W, V, b, d, test_data, type_act_func: str):
    E = 0
    num_correct = 0

    for k in range(len(test_data)):
        input_output = test_data[k]
        x = input_output[0:2]  # input vector
        u = input_output[2]  # output (target - only index)
        u_vec = get_output_vector(u)

        # Counting output
        x_k = np.array(x).reshape(-1, 1)
        z_k = activation_function((np.dot(W, x_k) + b), type_act_func)
        y_k = activation_function((np.dot(V, z_k) + d), type_act_func)
        y = y_k

        y_max_idx = np.argmax(y)
        y_vec = np.zeros((5, 1))
        y_vec[y_max_idx][0] = 1

        # Counting error
        E = E + np.dot(1 / 2, np.dot((u_vec - y).reshape(1, -1), (u_vec - y)))

        if (y_vec == u_vec).all():
            num_correct += 1

        visualize_points_clusters(x, y_vec)

    accuracy = num_correct / len(test_data)
    plt.grid()
    plt.show()

    return int(E), accuracy


def get_xy_min_max(data: list):
    x_values = []
    y_values = []

    for vector in data:
        x_values.append(vector[0])
        y_values.append(vector[1])

    return min(x_values), max(x_values), min(y_values), max(y_values)


def visualize_result(data, W, b, data_type: str):
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
    data_path = Path("./data/tren_data2___03.txt")

    data = load_data(data_path)
    # visualize_data(data)
    # visualize_data_histogram(data)
    random.shuffle(data)
    train_data, test_data = split_data_train_test(data)

    act_func = "sigmoid"
    W, V, b, d, errors = train_two_layered_nn(train_data, act_func)
    visualize_errors(errors)

    test_error, test_accuracy = do_inference(W, V, b, d, test_data, act_func)
    print(f"Test accuracy: {test_accuracy * 100} %")

    visualize_result(train_data, W, b, "train")
    visualize_result(test_data, W, b, "test")