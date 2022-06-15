import os, sys

from cache import LRUCache

class Input:
    def __init__(self):
        self.reads = []

    def parse(self, fname):
        with open(fname, "r") as f:
            for line in f:
                offset = line.split(",")[-1].rstrip()
                self.reads.append(int(offset))
        print(f"Parsed {len(self.reads)} inputs")

def nothing(reads, i, is_majfault, lru):
    return []

readahead_n = 32

# by default linux uses 128kb, which is 32 pages
def linux_readahead(reads, i, is_majfault, lru):
    if is_majfault:
        page = reads[i]
        return [x for x in range(page+1,page+readahead_n+1)]
    return []

def oracle_readahead(reads, i, is_majfault, lru):
    if is_majfault:
        ras = []
        for j in range(1,readahead_n+1):
            if i+j == len(reads): break
            ras.append(reads[i+j])
        return ras
    return []

def main():
    if len(sys.argv) != 2:
        print("Need path to input trace")
        sys.exit(1)

    inputs = Input()
    inputs.parse(sys.argv[1])

    lru = LRUCache(1024)

    algs = {
        "nothing": nothing,
        "oracle": oracle_readahead,
        "linux": linux_readahead,
    }

    print("\n\n")
    print("******************************")
    for name, alg in algs.items():
        majfault = 0
        for i,r in enumerate(inputs.reads):
            is_majfault = False
            if lru.get(r) == -1:
                is_majfault = True
                majfault += 1
            
            ras = alg(inputs.reads, i, is_majfault, lru)

            for ra in ras:
                lru.put(ra, 0)

        print(f"{name}, {majfault}")
    print("******************************")

if __name__ == "__main__":
    main()
