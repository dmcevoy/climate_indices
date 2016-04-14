import csv
import numpy as np
import sys

#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        data_values_count = sum(1 for _ in f)
    
    data_values = np.full((data_values_count,), np.NaN)
    with open(sys.argv[1], 'rb') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            data_values[i] = row[0]
            i += 1
            
    values_string = ''
    for i, value in enumerate(data_values):
        new_value_string = str(value)
        if (i + 1) % 12 == 0:
            new_value_string = new_value_string + ',\n'
        else:
            new_value_string = new_value_string + ', '
        values_string = values_string + new_value_string
    print(values_string)