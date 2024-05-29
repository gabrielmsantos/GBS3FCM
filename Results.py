import os


def find_max_accuracy(filepath):
    max_accuracy = 0.0

    with open(filepath, 'r') as file:
        for line in file:
            # Split the line and get the last part which is the accuracy
            accuracy = float(line.strip().split()[-1])
            # Update max_accuracy if the current accuracy is greater
            if accuracy > max_accuracy:
                max_accuracy = accuracy

    return max_accuracy


def compute_average_max_accuracies(directory):
    max_accuracies = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Only process .txt files
            filepath = os.path.join(directory, filename)
            max_accuracy = find_max_accuracy(filepath)
            max_accuracies.append(max_accuracy)

    # Compute the average of the max accuracies
    if max_accuracies:
        average_max_accuracy = sum(max_accuracies) / len(max_accuracies)
    else:
        average_max_accuracy = 0.0

    return average_max_accuracy


# Example usage:
directory = './RES-DIA/R30/'  # Replace with your directory path
average_max_accuracy = compute_average_max_accuracies(directory)
print(f"Average accuracy over 20 runs: {average_max_accuracy}")
