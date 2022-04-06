#!/usr/bin/python3

import sys
import os
from natsort import natsorted
import numpy as np
from post_process_cpu_stat import post_process_cpu_stat

def get_txt_files(path):
  return sorted([ f for f in os.listdir(path) if '.txt' in f ])

# return relevant data files, excluding unprocessed cpu files
# TODO: what are the 'mean' files? they don't seem to have a ton of relevant information
def get_data_files(path):
  return sorted([ f for f in get_txt_files(path) if f.split('_')[0] != 'stats'
    and f.split('_')[0] != 'mean' ])

def get_batch_size(f):
  f = f.replace('.txt', '')
  return f.split('_')[-1]

def get_data_source(f):
  return f.split('_')[0]

def cpu_stat_post_processing(path):
  cpu_stat_files = [ f for f in get_txt_files(path) if f.split('_')[0] == 'stats' ]
  for f in cpu_stat_files:
    post_process_cpu_stat(os.path.join(path, f), os.path.join(path, 'cpu_' + f))

def get_file_contents(path, files):
  contents = {}
  for f in files:
    batch_size = get_batch_size(f)
    data_source = get_data_source(f)
    with open(os.path.join(path, f), 'r') as fd:
      text = fd.read()
      if batch_size not in contents:
        contents[batch_size] = {}
      contents[batch_size][data_source] = text
  return contents

def parse_data_from_source(source, data):
  # parse into lines
  data = [ l.strip() for l in data.split('\n') if len(l.strip()) > 0 ]

  # return if there is no output
  if len(data) == 0:
    return {}

  # remove header
  data = data[1:]

  # parse values
  data = [ [ float(datum) for datum in d.split(',') if len(datum) > 0 ]
    for d in data
  ]

  if source == 'cpu':
    time = [ d[0] for d in data ]
    utilization = [ d[1] for d in data ]
    return {
      'cpu_t' : time,
      'cpu_utilization' : utilization
    }

  if source == 'gpu':
    time = [ d[0] for d in data ]
    utilization = [ d[1] for d in data ]
    clock_mhz = [ d[2] for d in data ]
    temp = [ d[3] for d in data ]
    return { 
      'gpu_t' : time,
      'gpu_utilization' : utilization,
      'clock_mhz' : clock_mhz,
      'temp' : temp
    }

  if source == 'sysfs':
    time = [ d[0] for d in data ]
    pages_per_sec = [ d[1] for d in data ]
    total_time_100_scans = [ d[2] for d in data ]
    return { 
      'sysfs_t' : time,
      'pages_per_sec' : pages_per_sec,
      'total_time_100_scans' : total_time_100_scans
    }

def parse_data(contents):
  data = {}
  for batch_size in contents:
    data[batch_size] = {}
    for data_source in contents[batch_size]:
      data[batch_size][data_source] = parse_data_from_source(
        data_source,
        contents[batch_size][data_source]
      )
  return data

def clean_datum(datum, value):
  if datum == 'pages_per_sec':
    return [ v for v in value if v != 0 ]

  if datum == 'total_time_100_scans':
    return max(value)

  return value

def clean_data(data):
  cleaned_data = {}
  for batch_size in data:
    cleaned_data[batch_size] = {}
    for data_source in data[batch_size]:
      for datum in data[batch_size][data_source]:
        cleaned_data[batch_size][datum] = clean_datum(
          datum,
          data[batch_size][data_source][datum]
        )
  return cleaned_data

def print_data(data, o_file):
  if o_file == 'stdout':
    fp = sys.stdout
  else:
    fp = open(o_file, 'w')

  fp.write(
    'type,' +
    'batch_size,' +
    'pages_per_sec,' +
    'cpu_utilization,' +
    'gpu_utilization,' +
    'total_time_100_scans,'
  )
  fp.write('\n')

  for batch_size in natsorted(data.keys()):
    if batch_size == '0':
      fp.write('cpu,')
      fp.write('0,')
    else:
      fp.write('gpu,')
      fp.write(batch_size + ',')
    fp.write(str(np.mean(data[batch_size]['pages_per_sec'])) + ',')
    if 'cpu_utilization' in data[batch_size]:
      fp.write(str(np.mean(data[batch_size]['cpu_utilization'])) + ',')
    else:
      fp.write('0.0,')
    fp.write(str(np.mean(data[batch_size]['gpu_utilization'])) + ',')
    fp.write(str(np.mean(data[batch_size]['total_time_100_scans'])) + ',')
    fp.write('\n')

def post_process(data_dir, o_file):
  input_path = os.path.abspath(data_dir)
  cpu_stat_post_processing(input_path)
  files = get_data_files(input_path)
  contents = get_file_contents(input_path, files)
  data = parse_data(contents)
  cleaned_data = clean_data(data)
  print_data(cleaned_data, o_file)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser('Data cleaning and formatting')
  parser.add_argument('-d', type=str, default='./')
  parser.add_argument('-o', type=str, default='stdout')

  args = parser.parse_args()

  post_process(args.d, args.o)
