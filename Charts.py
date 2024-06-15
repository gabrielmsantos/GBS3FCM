import pandas as pd
import matplotlib.pyplot as plt


# # Creating the DataFrame manually based on the provided data
# data = {
#     "Method": ["K-Means", "FCM", "SMKFCM", "SMKFC-ER", "CSNMF", "LHCS3FCM", "CS3FCM", "AS3FCM", "KGBS3FCM"],
#     "0%": [54.4, 52, 57, 52.2, 53, 56.8, 55.9, 57.2, 69.3],
#     "5%": [54.3, 52, 56.2, 52.2, 54.3, 56.4, 56.2, 56.6, 68.0],
#     "10%": [54.3, 52, 52.3, 52.2, 54.2, 56.5, 55.8, 56.2, 65.5],
#     "15%": [54.3, 52, 53.3, 52.2, 53.1, 56, 55.8, 55.6, 64.5],
#     "20%": [54.3, 52, 53.5, 52.2, 53.8, 55.4, 56.7, 55.4, 63.4],
#     "25%": [54.3, 52, 53.3, 52.2, 53.8, 55, 55.8, 55.3, 61.4],
#     "30%": [54.2, 52, 52.5, 52.2, 53.6, 54.8, 52, 55.4, 60.1]
# }

#df = pd.DataFrame(data)

# Load the CSV file into a pandas DataFrame
file_path = './chart/wdbc.csv'
df = pd.read_csv(file_path, delimiter=';', decimal=',')

# Set the 'Method' column as the index
df.set_index('Method', inplace=True)

# Transpose the DataFrame for plotting
df = df.T

# Define markers and line styles
markers = ['o', '*', 's', 'v', '^', 'D', 'X', 'p', 'h']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

# Plot the data
plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns):
    if i == len(df.columns) - 1:  # Last method
        plt.plot(df.index, df[column], marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)],
                 label=column, linewidth=3, color='red')
    else:
        plt.plot(df.index, df[column], marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)],
                 label=column, linewidth=2)

    #plt.plot(df.index, df[column], marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=column, linewidth=2)

plt.title('')
plt.xlabel('Mislabeling Ratio (%)', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.legend(title="Method")
plt.grid(True)

# Customize the plot to match the style of the provided image
plt.xticks(df.index, rotation=45)
plt.yticks(range( df.min().min().astype(int) -2, df.max().max().astype(int) +2, 2))
plt.ylim(df.min().min().astype(int) -2, df.max().max().astype(int) + 2)
plt.xlim(df.index[0], df.index[-1])

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.show()
