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


if __name__ == "__main__":
    data_path = Path("./data/tren_data2___03.txt")

    data = load_data(data_path)
    visualize_data(data)
    visualize_data_histogram(data)
    random.shuffle(data)
    train_data, test_data = split_data_train_test(data)

    # W, b, errors = train_one_layered_nn(train_data)
    # visualize_errors(errors)
    #
    # test_error, test_accuracy = do_inference(W, b, test_data)
    # print(f"Test accuracy: {test_accuracy * 100} %")
    #
    # visualize_result(train_data, W, b, "train")
    # visualize_result(test_data, W, b, "test")