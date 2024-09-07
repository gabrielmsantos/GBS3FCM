import os
import re

# Directory containing your files
directory = '.'

db = 'WAVEFORM'
num = 30

# Pattern to match the files
pattern = re.compile(r'GB_' + db + '_R' + str(num) + '_T(\d+)\.txt')

for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        new_number = number + 10
        new_filename = f"GB_{db}_R{num}_T{new_number}.txt"
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {old_filepath} -> {new_filepath}')