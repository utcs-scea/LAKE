import pandas as pd
import sys

data_input_path = sys.argv[1]
data = pd.read_csv(data_input_path, dtype='float32',sep=',',header=None)
print(data.info())
print(data.shape)
print(data.describe(percentiles = [0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.80,0.82,0.84,0.86,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99])[31])
