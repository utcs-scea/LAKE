from .traffic import *
from .memory import *
from .scheduler import *

class PProfile:
  def __init__(self, trace_file):
    self.trace_file = trace_file
    self.traffic = TrafficGen(self.trace_file)
    self.hmem = AddressSpace()
    self.scheduler = Scheduler()
  
  def init(self):
    self.traffic.parse_pin_trace()
    self.traffic.print_traffic_sum()
    self.hmem.populate(self.traffic)