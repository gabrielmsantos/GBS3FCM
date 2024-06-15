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


def compute_average_max_accuracies(directory, prefix):
    max_accuracies = []
    i = 0
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if prefix in filename and filename.endswith(".txt"):  # Only process .txt files
            i += 1
            filepath = os.path.join(directory, filename)
            max_accuracy = find_max_accuracy(filepath)
            max_accuracies.append(max_accuracy)

    # Compute the average of the max accuracies
    if max_accuracies:
        average_max_accuracy = sum(max_accuracies) / len(max_accuracies)
    else:
        average_max_accuracy = 0.0

    return average_max_accuracy, i


# Example usage:
db_name = 'wdbc'
directory = f'./results/VNorm/{db_name}/'  # Replace with your directory path
percentages = [0, 5, 10, 15, 20, 25, 30]
for percentage in percentages:
    prefix = f"_R{percentage}_"
    average_max_accuracy, num = compute_average_max_accuracies(directory, prefix)
    print(f"[{db_name}-GB-{percentage}] Average accuracy over {num} runs: {average_max_accuracy}")

#average_max_accuracy = compute_average_max_accuracies(directory)
#print(f"Average accuracy over 20 runs: {average_max_accuracy}")
