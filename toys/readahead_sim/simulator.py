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
LRU_CACHE_SIZE = 1024*2

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
        if "reads" not in kwargs.keys():
            print("Error on Linux")
        self.reads = kwargs["reads"]

    def readahead(self, idx, is_majfault):
        return [self.reads[idx]] if is_majfault else []

# by default linux uses 128kb, which is 32 pages
class Linux:
    def __init__(self, **kwargs):
        if "reads" not in kwargs.keys():
            print("Error on Linux")
        self.reads = kwargs["reads"]
        self.readahead_size = kwargs["readahead_size"]

    def readahead(self, idx, is_majfault):
        if is_majfault:
            cur_page = self.reads[idx]
            return [x for x in range(cur_page,cur_page+self.readahead_size)]
        return []

class Oracle:
    def __init__(self, **kwargs):
        if "reads" not in kwargs.keys():
            print("Error on Linux")
        self.reads = kwargs["reads"]
        self.readahead_size = kwargs["readahead_size"]

    def readahead(self, idx, is_majfault):
        if is_majfault:
            ras = []
            for j in range(0,self.readahead_size):
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
    
    ten_pct = len(inputs.reads)/10
    
    ra_sizes = []
    for i in range(1, 16):
        ra_sizes.append(READAHEAD_SIZE*i)

    print("\n\n")
    print("******************************")
    print(f"policy, major_faults, cache_hits, io_reads, unaccessed_cache, useful_io")
    for name, alg_class in algs.items():
        # create alg object
        for rasize in ra_sizes:
            # create LRU page cache
            lru = LRUCache(LRU_CACHE_SIZE)

            alg = alg_class(reads=inputs.reads, readahead_size=rasize, trace=trace, simulator=True, 
                models_path=os.path.join(this_dir, "ml", "models"))
            majfault = 0
            io_reads = 0
            cur_pct = 0
            cache_hits = 0
            # Loop for every read in the trace
            for i in range(len(inputs.reads)):
                cur_read = inputs.reads[i]
                is_majfault = False
                if lru.get(cur_read) is None:
                    is_majfault = True
                    majfault += 1
                else:
                    cache_hits += 1

                ras = alg.readahead(i, is_majfault)
                assert len(ras) <= rasize, f"Dont you dare cheat on me {len(ras)} != {rasize}"
                assert (len(ras) == 0 and not is_majfault) or is_majfault, "Cant readahead on non-major fault"
                if is_majfault:
                    assert cur_read in ras, f"faulted on page {cur_read}, but it's not on readahead list"

                # do readahead. ras will be empty if this is not a major fault
                for ra in ras:
                    io_reads += lru.put(ra)
                # do a get so we increase the access count and put it on the front
                lru.get(cur_read)

                #if i % ten_pct == 0:
                #    print(f"  at {cur_pct}%")
                #    cur_pct += 10

            bads = lru.get_never_access_count()

            #m1 = (cache_hits - bads - io_reads) / len(inputs.reads)
            m1 = (io_reads-cache_hits) / len(inputs.reads)
            m2 = cache_hits/io_reads

            print(f"{name}_{rasize}, {majfault}, {cache_hits}, {io_reads}, {bads}, {m1:.4f}, {m2:.4f}")
    print("******************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="path to trace file", type=str)
    args = parser.parse_args()
    main(args)
