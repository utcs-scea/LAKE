
import csv
import sys
from enum import IntEnum

LEN_HIS_QUEUE = 4
IO_READ = '1'
IO_WRITE = '0'
LABEL_ISSUE = 'i'
LABEL_COMPLETION = 'c'


# this is how the replayer now outputs things
#        sprintf(buf, "%.3ld,%d,%d,%ld,%lu,%.3ld,%u,%lu", ts, latency, !op, 
#                size, offset, submission, device, io_index);
#

class ReplayFields(IntEnum):
    TS = 0
    LATENCY = 1
    OP = 2
    SIZE = 3
    OFFSET = 4
    SUBMISSION = 5
    DEVICE = 6
    INDEX = 7


def generate_raw_vec(input_path, output_path, device_index):
    with open(input_path, 'r') as input_file:
        input_csv = csv.reader(input_file)

        trace_list = []
        transaction_list = []
        index = 0
        for row in input_csv:
            if row[ReplayFields.DEVICE] != device_index:
                continue

            latency = int(row[ReplayFields.DEVICE])
            type_op = row[ReplayFields.OP]
            #size_ori = int(row[ReplayFields.SIZE])
            size = int((int(row[ReplayFields.SIZE])/512 + 7)/8)
            issue_ts = int(float(row[ReplayFields.SUBMISSION])) #this is in us now, so no *1000
            complete_ts = issue_ts+latency

            # trace_list.append([latency, type_op, size, issue_ts, complete_ts, 0])
            trace_list.append([size, type_op, latency, 0, index])
            transaction_list.append([index, issue_ts, LABEL_ISSUE])
            transaction_list.append([index, complete_ts, LABEL_COMPLETION])
            # index is used by trans to find corresponding trace_list
            index += 1

    #relying on stable sort, oof
    transaction_list = sorted(transaction_list, key=lambda x: x[1])
    print('trace loading completed:', len(trace_list), 'samples')

    with open(output_path, 'w') as output_file:
        count = 0
        skip = 0
        pending_io = 0
        history_queue = [[0, 0]]*LEN_HIS_QUEUE
        raw_vec = [0]*(LEN_HIS_QUEUE*2+1+1)
        # print(history_queue)

        # this is what this list looks like, but for some reason sorted by ts
        #  transaction_list.append([index, issue_ts, LABEL_ISSUE])
        #  transaction_list.append([index, complete_ts, LABEL_COMPLETION])
        # and this is trace_list
        # trace_list.append([size, type_op, latency, 0, index])
        for trans in transaction_list:
            io = trace_list[trans[0]]
            if trans[2] == LABEL_ISSUE:
                pending_io += io[0]
                io[3] = pending_io

                if io[1] == IO_READ and skip >= LEN_HIS_QUEUE:
                    count += 1
                    raw_vec[LEN_HIS_QUEUE] = io[3]
                    raw_vec[-1] = io[2]
                    for i in range(LEN_HIS_QUEUE):
                        raw_vec[i] = history_queue[i][1]
                        raw_vec[i+LEN_HIS_QUEUE+1] = history_queue[i][0]
                    output_file.write(','.join(str(x) for x in raw_vec)+'\n')

            elif trans[2] == LABEL_COMPLETION:
                pending_io -= io[0]

                if io[1] == IO_READ:
                    history_queue.append([io[2], io[3]])
                    del history_queue[0]
                    skip += 1

        # print(history_queue)
        print(pending_io)
        print('Done:', str(count), 'vectors')


def generate_ml_vec(len_pending, len_latency, input_path, output_path):
    max_pending = (10**len_pending)-1
    max_latency = (10**len_latency)-1
    # print(max_pending, max_latency)
    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        input_csv = csv.reader(input_file)
        for rvec in input_csv:
            tmp_vec = []
            for i in range(LEN_HIS_QUEUE+1):
                pending_io = int(rvec[i])
                if pending_io > max_pending:
                    pending_io = max_pending
                tmp_vec.append(','.join(x for x in str(pending_io).rjust(len_pending, '0')))
            for i in range(LEN_HIS_QUEUE):
                latency = int(rvec[i+LEN_HIS_QUEUE+1])
                if latency > max_latency:
                    latency = max_latency
                tmp_vec.append(','.join(x for x in str(latency).rjust(len_latency, '0')))
            tmp_vec.append(rvec[-1])
            output_file.write(','.join(x for x in tmp_vec)+'\n')

if len(sys.argv) < 2:
    print('illegal cmd format')
    exit(1)

mode = sys.argv[1]
if mode == 'raw':
    if len(sys.argv) != 4:
        print('illegal cmd format')
        exit(1)
    trace_path = sys.argv[2]
    raw_path = sys.argv[3]
    generate_raw_vec(trace_path, raw_path)
elif mode == 'ml':
    if len(sys.argv) != 6:
        print('illegal cmd format')
        exit(1)
    len_pending = int(sys.argv[2])
    len_latency = int(sys.argv[3])
    raw_path = sys.argv[4]
    ml_path = sys.argv[5]
    generate_ml_vec(len_pending, len_latency, raw_path, ml_path)
elif mode == 'direct':
    #only modified
    if len(sys.argv) != 8:
        print('illegal cmd format')
        exit(1)
    len_pending = int(sys.argv[2])
    len_latency = int(sys.argv[3])
    trace_path = sys.argv[4]  #mlData/TrainTraceOutput
    temp_file_path = sys.argv[5]  #mlData/temp1
    output_path = sys.argv[6] #mlData/"mldrive${i}.csv"
    device_index = sys.argv[7] #"$i"

    generate_raw_vec(trace_path, temp_file_path, device_index)
    generate_ml_vec(len_pending, len_latency, temp_file_path, output_path)
else:
    print('illegal mode code')
    exit(1)

# trace = 'WD_NVMe_1_6T.bingselection.drive0.rr1.exp_0'
# trace_path = '/Users/linanqinqin/Documents/DESS/Macrobench/replayML/'+trace+'.csv'
# raw_path = '/Users/linanqinqin/Documents/DESS/Macrobench/replayML/'+trace+'.raw_vec.csv'
# ml_path = '/Users/linanqinqin/Documents/DESS/Macrobench/replayML/'+trace+'.ml_vec.31.csv'
# generate_raw_vec(trace_path, raw_path)
# generate_ml_vec(3, 4, raw_path, ml_path)
