# the granularity distances are measured
BLOCK_SZ = 32
MAX_DIST = 1024
#MAX_DIST = 8
#the max dist measurable will be BLOCK_SZ*MAX_DIST pages
#32 and 1024 have  range of ~134 MB (half each way)

#how many distances are input to the model
SLICE_LEN = 8

EPOCHS = 10
LEARN_RATE = 0.01


LSTM_EPOCHS = 10


#SSD LSTM

SSD_N_CLASSES = 100
SSD_WINDOW_SZ = 8
SSD_EPOCHS = 10