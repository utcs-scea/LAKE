from abc import ABC, abstractmethod
import pandas as pd
import math
import logging

from numa_map import cpu_nodemap, NR_NODES
from dump_config import COLUMNS

sysctl_migrate_cost = 500000  # Defined in kernel

logger = logging.getLogger('datasource')
fhdlr = logging.FileHandler('write.log', mode='w')
fhdlr.terminator=''
logger.addHandler(fhdlr)


class DataSource(ABC):

    @abstractmethod
    def update(self, event):
        pass

    @abstractmethod
    def dump(self):
        pass


class CanMigrateData(DataSource):
    def __init__(self, write_file, append=False, write_size=None):
        self.columns = COLUMNS
        self.entries = []
        self.write_size = write_size or 1000
        self.write_cd = self.write_size
        self.write_file = write_file
        if not append:
            with open(self.write_file, 'w') as f:
                f.write(','.join(self.columns) + '\n')

        print('DataSource initialized. Writing to {}'.format(write_file))
        print('write-size: {}, appending: {}'.format(self.write_size, append))
        logger.warn('{}, {}\n'.format(write_file, self.write_size))

    def update(self, event):
        row = {}
        row['ts'] = event.ts
        src_cpu = event.src_cpu
        dst_cpu = event.dst_cpu
        src_nid = cpu_nodemap[src_cpu]
        dst_nid = cpu_nodemap[dst_cpu]
        row['curr_pid'] = event.curr_pid
        row['pid'] = event.pid
        row['src_cpu'] = src_cpu
        row['dst_cpu'] = dst_cpu
        preferred_nid = event.numa_preferred_nid
        row['same_node'] = 1 if src_nid == dst_nid else 0
        row['prefer_src'] = 1 if preferred_nid == src_nid else 0
        row['prefer_dst'] = 1 if preferred_nid == dst_nid else 0
        row['total_faults'] = event.total_numa_faults
        src_faults = event.p_numa_faults[src_nid]
        dst_faults = event.p_numa_faults[dst_nid]
        row['delta_faults'] = dst_faults - src_faults
        row['imbalance'] = event.imbalance
        #  row['delta'] = math.log(event.delta / sysctl_migrate_cost)
        row['delta'] = event.delta
        row['cpu_idle'] = event.cpu_idle
        row['cpu_not_idle'] = event.cpu_not_idle
        row['cpu_newly_idle'] = event.cpu_newly_idle
        row['src_len'] = event.src_nr_running
        row['src_numa_len'] = event.src_nr_numa_running
        row['src_preferred_len'] = event.src_nr_preferred_running
        row['dst_len'] = event.dst_nr_running
        row['src_load'] = event.src_load
        row['dst_load'] = event.dst_load
        row['nr_fails'] = event.nr_balance_failed
        row['cache_nice_tries'] = event.cache_nice_tries
        row['buddy_hot'] = event.buddy_hot
        row['p_running'] = event.p_running
        #  row['fair_class'] = event.fair_class
        row['throttled'] = event.throttled
        row['can_migrate'] = event.can_migrate

        row['test_aggressive'] = event.test_aggressive
        row['pc_0'] = event.perf_count_0
        row['pc_1'] = event.perf_count_1
        self.entries.append([str(row[col]) for col in self.columns])
        #  self.df.loc[ts] = row
        self.write_cd -= 1
        if self.write_cd == 0:
            print('.', end=' ', flush=True)
            logger.warn('.')
            with open(self.write_file, mode='a') as f:
                for row in self.entries:
                    f.write(','.join(row) + '\n')

            self.write_cd = self.write_size
            self.entries = []

    def dump(self, filename=None):
        filename = filename or self.write_file or 'output.csv'
        with open(self.write_file, mode='a') as f:
            for row in self.entries:
                f.write(','.join(row) + '\n')
        self.entries = []
        return 'Entries Dumped'
