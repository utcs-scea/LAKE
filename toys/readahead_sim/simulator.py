#!/usr/bin/env python3
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from cache import LRUCache

# I'm so done with this absolute/relative python bs
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(this_dir, "ml"))
from ml.ssdlstm import LSTM_SSD

READAHEAD_SIZE = 32
LRU_CACHE_SIZE = 1024

class Input:
    def __init__(self):
        self.reads = []

    def parse(self, fname):
        with open(fname, "r") as f:
            for line in f:
                offset = line.split(",")[-1].rstrip()
                self.reads.append(int(offset))
        print(f"Parsed {len(self.reads)} inputs")

#
#  Fixed Readahead Policies
#
class NoPrefetch:
    def __init__(self, **kwargs):
        pass

    def readahead(self, idx, is_majfault):
        return []

# by default linux uses 128kb, which is 32 pages
class Linux:
    def __init__(self, **kwargs):
        if "reads" not in kwargs.keys():
            print("Error on Linux")
        self.reads = kwargs["reads"]

    def readahead(self, idx, is_majfault):
        if is_majfault:
            cur_page = self.reads[idx]
            return [x for x in range(cur_page+1,cur_page+READAHEAD_SIZE+1)]
        return []

class Oracle:
    def __init__(self, **kwargs):
        if "reads" not in kwargs.keys():
            print("Error on Linux")
        self.reads = kwargs["reads"]

    def readahead(self, idx, is_majfault):
        if is_majfault:
            ras = []
            for j in range(1,READAHEAD_SIZE+1):
                if idx+j == len(self.reads): break
                ras.append(self.reads[idx+j])
            return ras
        return []

#
#  Main
#
algs = {
        "NoPrefetch": NoPrefetch,
        "Oracle": Oracle,
        "Linux": Linux,
        "SSD_LSTM": LSTM_SSD,
    }

#
#  Main
#
def main(args):
    fname = args.trace
    trace = fname.split("/")[-1]
    trace = trace.split(".")[-2]

    print(f"Simulating trace {trace}")

    # create inputs from trace file
    inputs = Input()
    inputs.parse(sys.argv[1])
    # create LRU page cache
    lru = LRUCache(LRU_CACHE_SIZE)

    ten_pct = len(inputs.reads)/10
    
    print("\n\n")
    print("******************************")
    for name, alg_class in algs.items():
        # create alg object
        alg = alg_class(reads=inputs.reads, readahead_size=READAHEAD_SIZE, trace=trace, simulator=True, 
            models_path=os.path.join(this_dir, "ml", "models"))
        majfault = 0
        cur_pct = 0
        for i,r in enumerate(inputs.reads):
            is_majfault = False
            if lru.get(r) == -1:
                is_majfault = True
                majfault += 1
            
            ras = alg.readahead(i, is_majfault)
            assert len(ras) <= READAHEAD_SIZE, f"Dont you dare cheat on me {len(ras)} != {READAHEAD_SIZE}"

            for ra in ras:
                lru.put(ra, 0)

            if i % ten_pct == 0:
                print(f"  at {cur_pct}%")
                cur_pct += 10

        print(f"{name}, {majfault}")
    print("******************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="path to trace file", type=str)
    args = parser.parse_args()
    main(args)
