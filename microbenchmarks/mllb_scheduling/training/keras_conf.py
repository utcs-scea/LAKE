TEST_SIZE = 50000
MODEL_TAG='default'
DO_LOAD         = 0
DO_TRAIN        = 1
DO_SAVE         = 1
DO_DUMP         = 0
DO_EVALUATE     = 1
LOAD_EVALUATE   = 0
EVALUATE_TAG = 'default'

# Cross validation
X_val           = 0

EPOCHS = 3
BATCH_SIZE = 64

""" Features
src_non_pref_nr,    delta_hot,      cpu_idle,       cpu_not_idle,   cpu_newly_idle,
same_node,          prefer_src,     prefer_dst,     src_len,        src_load,
dst_load,           dst_len,        delta_faults,   extra_fails,    buddy_hot
"""
