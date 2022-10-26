
import csv
import numpy as np
import sys
import glob
import os


def generate_1d_var_T(v_name, v_type, digits, thres, input_path):

    v_literal = v_type + ' ' + v_name

    row_count = 0
    col_count = 0
    with open(input_path, 'r') as input_file:
        input_data = csv.reader(input_file)
        raw_array = []

        for row in input_data:
            row_count += 1
            col_count = len(row)
            raw_array.append([item for item in row])

    # print(raw_array)
    # print(np.array(raw_array).shape)
    raw_array = np.array(raw_array).T.tolist()
    # print(np.array(raw_array).shape)

    v_literal += '[' + str(col_count) + '*' + str(row_count) + '] = {\n'
    str_list = []
    for row in raw_array:
        str_list.append(','.join(str(round(float(item)*(10**digits))) for item in row)+'\n')
    v_literal += ','.join(item for item in str_list)
    v_literal += '};\n'

    print(v_literal)


if len(sys.argv) != 5:
    print('Illegal cmd format')
    exit(1)

workload = sys.argv[1]
drive = sys.argv[2]
input_folder = sys.argv[3]
if input_folder[-1] != '/':
    input_folder += '/'
# print(input_folder)
output_folder = sys.argv[4]
if output_folder[-1] != '/':
    output_folder += '/'
# print(output_folder)

# workload = 'bingindex'
# drive = 'nvme3'
# iters = '40'

file_list_w0 = glob.glob(input_folder+'*.weight_0.csv')
file_list_w1 = glob.glob(input_folder+'*.weight_1.csv')
file_list_b0 = glob.glob(input_folder+'*.bias_0.csv')
file_list_b1 = glob.glob(input_folder+'*.bias_1.csv')
# print(file_list_b1)
if len(file_list_w0) != 1 or len(file_list_w1) != 1 or len(file_list_b0) != 1 or len(file_list_b1) != 1:
    print('Illegal input')
    exit(1)

w0_path = file_list_w0[0]
w1_path = file_list_w1[0]
b0_path = file_list_b0[0]
b1_path = file_list_b1[0]

if not os.path.exists(output_folder):
    print('Illegal output')
    exit(1)

sys.stdout = open(output_folder+'/w_'+ workload+"_"+drive+'.h', 'w')

print('\n/* '+workload+' */\n')
from datetime import datetime
print('/**\n *  updated on '+datetime.now().strftime('%m/%d/%Y %H:%M:%S')+'\n */\n')
print('#ifndef __W_'+drive.upper()+'_H\n#define __W_'+drive.upper()+'_H\n')
print('char name_'+drive+'[] = \"'+workload+'/'+drive+'\";\n')
digits = 3
generate_1d_var_T('weight_0_T_'+drive, 'long', digits, None, w0_path)
generate_1d_var_T('weight_1_T_'+drive, 'long', digits, None, w1_path)
generate_1d_var_T('bias_0_'+drive, 'long', digits, None, b0_path)
generate_1d_var_T('bias_1_'+drive, 'long', digits*2, None, b1_path)
print('#endif')
