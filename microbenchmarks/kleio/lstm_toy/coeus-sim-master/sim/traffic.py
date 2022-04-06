class Request:
  def __init__(self, id, page_id):
    self.id = id
    self.ep = 0
    self.page_id = page_id
    self.loc = -1
    
class TrafficGen():
  def __init__(self, trace_file):
    self.req_seq = []
    self.num_pages = 0
    self.num_reqs = 0
    self.trace_file = trace_file

  def parse_pin_trace(self):
    with open(self.trace_file, 'r') as infile:
      seq = []
      page_map = {}
    
      for line in infile.readlines():
        row = line.split(' ')
        self.num_reqs += 1
        paddr = int(row[0], 0)
        base_addr = paddr - (paddr % 4096)  # 4 KB pages
        key = str(base_addr)
        page_id = self.num_pages
        if key not in page_map:
          page_map[key] = self.num_pages
          self.num_pages = self.num_pages + 1
        else:
          page_id = page_map[key]
        # PC
        req = Request(self.num_reqs - 1, page_id)
        self.req_seq.append(req)
        
          
  def print_traffic_sum(self):
    print ("[Traffic] Num Pages =", self.num_pages, "Num Reqs =", self.num_reqs)