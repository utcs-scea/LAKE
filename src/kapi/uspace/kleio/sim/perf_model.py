import csv
from profile import *

class Platform:
  def __init__(self, name, local_lat, rem_lat, local_bw, rem_bw, period_cost, migr_cost):
    self.name = name
    self.local_lat = local_lat
    self.rem_lat = rem_lat
    self.local_bw = local_bw
    self.rem_bw = rem_bw
    self.period_cost = period_cost
    self.migr_cost = migr_cost

class PerfModel:
  def __init__(self, prof, platform_name, policy, cap_ratio, num_reqs):
    self.profile = prof
    self.platform_name = platform_name
    self.platform = None
    self.policy = policy
    self.cap_ratio = cap_ratio
    self.num_reqs_per_period = num_reqs
    self.stats = {}

  def init(self):
    self.set_platform()
    self.profile.scheduler.init(self.profile.traffic, self.profile.hmem, self.policy, self.num_reqs_per_period, self.cap_ratio)
  
  def init_hybrid(self, oracle_page_ids):
    self.profile.hmem.init_hybrid(oracle_page_ids)

  def set_platform(self):
    # Default simulation parameters
    fast_lat = 50  # 50 ns
    fast_bw = 10  # 10 GB / sec
    period_cost = 3000  # 3 us
    migr_cost = 1000  # 1 us
  
    if self.platform_name == 'Fast:NearFast':
      self.platform = Platform('Fast:NearFast', fast_lat, 2.2 * fast_lat, fast_bw, fast_bw, period_cost, migr_cost)
    elif self.platform_name == 'Fast:NearSlow':
      self.platform = Platform('Fast:NearSlow', fast_lat, 3 * fast_lat, fast_bw, 0.37 * fast_bw, period_cost, migr_cost)  # 2.7x slower BW
    elif self.platform_name == 'Fast:FarFast':
      self.platform = Platform('Fast:FarFast', fast_lat, 1000 + 2.2 * fast_lat, fast_bw, 0.1 * fast_bw, period_cost, migr_cost)  # 10x slower BW
    elif self.platform_name == 'Fast:FarSlow':
      self.platform = Platform('Fast:FarSlow', fast_lat, 1000 + 3 * fast_lat, fast_bw, 0.1 * fast_bw, period_cost, migr_cost)  # 10x slower BW

  def run(self):
    self.profile.scheduler.run()
    self.compute_baselines()
    self.compute_perf()
    self.compute_other_metrics()
  
  def dump_stats(self, resfile):
    w = csv.writer(open(resfile, "w"))
    for key, val in self.stats.items():
      w.writerow([key, val])

  def compute_perf(self):
    self.stats['Fast_Hitrate'] = round((self.profile.scheduler.l1_hits / float(self.profile.traffic.num_reqs)) * 100.0, 2)
    self.stats['Fast_Part_of_Runtime'] = self.profile.scheduler.l1_hits * self.platform.local_lat
    self.stats['Slow_Part_of_Runtime'] = (self.profile.traffic.num_reqs - self.profile.scheduler.l1_hits) * self.platform.rem_lat
    self.stats['Period_Overhead'] = self.profile.scheduler.num_periods * self.platform.period_cost
    self.stats['Migration_Overhead'] = self.profile.scheduler.num_migr * self.platform.migr_cost
    
    bytes_transferred = (self.profile.traffic.num_reqs * 64) + (self.profile.scheduler.num_migr * 4096)
    time_allowed_bw = bytes_transferred / float(self.platform.rem_bw)
    time_to_transfer = self.stats['Fast_Part_of_Runtime'] + self.stats['Slow_Part_of_Runtime'] + self.stats['Migration_Overhead']
    # i cannot transfer data faster than what the bottleneck bandwidth allows!
    self.stats['Queue_Overhead'] = 0
    if time_to_transfer < time_allowed_bw:
      self.stats['Queue_Overhead'] = time_allowed_bw - time_to_transfer
    
    self.stats['Runtime'] = self.stats['Fast_Part_of_Runtime'] + self.stats['Slow_Part_of_Runtime'] + self.stats['Period_Overhead'] + self.stats['Migration_Overhead'] + self.stats['Queue_Overhead']
    self.stats['Slowdown_from_all_fast'] = round(((self.stats['Runtime'] / self.stats['All_Fast_Runtime']) - 1) * 100.0, 2)
    
  def compute_other_metrics(self):
    self.stats['Period_Duration'] = self.profile.scheduler.num_reqs_per_ep
    self.stats['Number_of_Periods'] = self.profile.scheduler.num_periods
    self.stats['Num_Pages'] = self.profile.hmem.num_pages
    self.stats['Num_Pages_Misplaced'] = sum([page.misplacements > 0 for page in self.profile.hmem.page_list])
    self.stats['Policy'] = self.profile.scheduler.policy
    self.stats['Num_Pages_Under_Oracle'] = len(self.profile.hmem.oracle_page_ids)
    self.stats['Num_Patterns'] = self.profile.hmem.num_patterns

  def compute_single_tiering(self, lat, bw):
    bytes_transferred = self.profile.traffic.num_reqs * 64
    time_to_transfer = self.profile.traffic.num_reqs * lat
    time_allowed_bw = bytes_transferred / float(bw)
    stalls = 0
    if time_to_transfer < time_allowed_bw:
      stalls = time_allowed_bw - time_to_transfer
    return time_to_transfer + stalls

  def compute_baselines(self):
    self.stats['All_Fast_Runtime'] = self.compute_single_tiering(self.platform.local_lat, self.platform.local_bw)
    self.stats['All_Slow_Runtime'] = self.compute_single_tiering(self.platform.rem_lat, self.platform.rem_bw)
