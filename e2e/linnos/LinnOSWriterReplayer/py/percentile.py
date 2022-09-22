
import csv
import numpy as np
import sys

LABEL_READ = '1'
LABEL_WRITE = '0'


def get_percentile_data(decimals, io_type, input_path, output_path):

    latency_array = []

    nr_samples = 0
    with open(input_path, 'r') as input_file:

        metrics = csv.reader(input_file)

        for row in metrics:
            if io_type is None:
                latency_array.append(int(row[3]))
            else:
                if row[4] == io_type:
                    if len(row) == 9:
                        if int(row[8]) > 0:
                            latency_array.append(int(row[3]))
                            nr_samples += 1
                    else:
                        latency_array.append(int(row[3]))
                        nr_samples += 1

        print(nr_samples)

    latency_array.sort()

    y_axis_array = []
    for i in range(1, 100*10**decimals):

        percent = i/(10**decimals)
        y_axis_array.append(percent)
    y_axis_array.append(100.0)

    percentile_array = np.percentile(latency_array, y_axis_array)

    # print(percentile_array)
    # print(y_axis_array)

    percentile_array = np.array(percentile_array).astype(int)
    y_axis_array = np.array(y_axis_array).astype(float)

    input_file_path = input_path.split('/')[-1]
    input_file_path = input_file_path.split('.')
    if io_type is not None:
        if io_type == LABEL_READ:
            input_file_path[-1] = 'read_percentile.csv'
        elif io_type == LABEL_WRITE:
            input_file_path[-1] = 'write_percentile.csv'
    else:
        input_file_path[-1] = 'percentile.csv'
    output_path += '.'.join(input_file_path)
    print(output_path)

    with open(output_path, 'w') as output_file:

        for percent, latency in zip(y_axis_array, percentile_array):

            output_file.write(str(latency)+'\t'+str(percent/100.0)+'\n')


if len(sys.argv) != 5:
    print('Illegal cmd parameters')
    quit(1)

nr_decimals = int(sys.argv[1])
io_type = sys.argv[2]
input_path = sys.argv[3]
output_folder = sys.argv[4]

if io_type == 'read':
    io_type = LABEL_READ
elif io_type == 'write':
    io_type = LABEL_WRITE
else:
    io_type = None

get_percentile_data(decimals=nr_decimals,
                    io_type=io_type,
                    input_path=input_path,
                    output_path=output_folder)
