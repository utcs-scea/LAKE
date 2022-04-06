import numpy as np
from sim.perf_model import *

class PageSelector:
  def __init__(self, prof, platform_name, cap_ratio, num_reqs_per_ep, resdir_prefix):
    self.prof = prof
    self.platform_name = platform_name
    self.cap_ratio = cap_ratio
    self.num_reqs_per_ep = num_reqs_per_ep
    self.solution = ''
    self.resdir_prefix = resdir_prefix
    
  def get_misplaced_pages_sim(self):
    page_ids = []
    for page in self.prof.hmem.page_list:
      if page.misplacements > 0:
        page_ids.append(page.id)
    return page_ids
    
  def get_misplaced_pages(self):
    self.run_scheduler('history', [], 0)
    page_ids = []
    for page in self.prof.hmem.page_list:
      if page.misplacements > 0:
        page_ids.append(page.id)
    return page_ids
  
  def get_distinct_access_patterns(self, page_ids):
    distinct_page_cnts_seq = set()
    for page_id in page_ids:
      page = self.prof.hmem.page_list[page_id]
      if str(page.oracle_counts_binned_ep) not in distinct_page_cnts_seq:
        distinct_page_cnts_seq.add(str(page.oracle_counts_binned_ep))
    return distinct_page_cnts_seq
  
  def select_k_page_groups(self, ordered_page_ids, k):
    selected_page_ids = []
    selected_patterns = set()
    for page_id in ordered_page_ids:
      page = self.prof.hmem.page_list[page_id]
      if str(page.oracle_counts_binned_ep) not in selected_patterns and len(selected_patterns) < k:
        selected_patterns.add(str(page.oracle_counts_binned_ep))
      if str(page.oracle_counts_binned_ep) in selected_patterns:
        selected_page_ids.append(page_id)
    return selected_page_ids
  
  def get_ordered_pages(self, page_ids):
    benefit_per_page = []
    for id in page_ids:
      page = self.prof.hmem.page_list[id]
      benefit = sum(page.oracle_counts_binned_ep) * page.misplacements
      print(f"page {id} has {sum(page.oracle_counts_binned_ep)} sum of counts and {page.misplacements} misplacements")
      benefit_per_page.append(benefit)
    
    sorted_idxs = np.argsort(benefit_per_page)[::-1] # descending order
    ordered_page_ids = [page_ids[i] for i in sorted_idxs]
    return ordered_page_ids
    
  def run_scheduler(self, policy, selected_pages_for_oracle, num_rnns):
    sim = PerfModel(self.prof, self.platform_name, policy, self.cap_ratio, self.num_reqs_per_ep)
    sim.init()
    if policy == 'hybrid' or policy == 'hybrid-group':
      sim.init_hybrid(selected_pages_for_oracle)
      #policy += str(len(selected_pages_for_oracle))
    sim.run()
    sim.stats['Num_RNN'] = num_rnns
    sim.dump_stats(self.resdir_prefix + self.solution + '_' + policy + '.csv')
    
  def run_kleio(self):
  
    # Step 1: get pages misplaced by a history-based page scheduler
    pages_misplaced = self.get_misplaced_pages()

    # Step 2: Order pages by benefit = # accesses x # misplacements
    pages_ordered = self.get_ordered_pages(pages_misplaced)
    
    # Step 3: Get performance curve when oracle manages part of the selected pages and history the rest
    perc_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for perc in perc_list:
      npages = int(perc * len(pages_ordered))
      self.run_scheduler('hybrid', pages_ordered[:npages])
 
    # Hybrid for first k pages
    self.run_scheduler('hybrid', pages_ordered[:100])

    # Step 4: Get oracle baseline
    self.run_scheduler('oracle', [])

  def run_compare(self):
  
    for solution in ['cori', 'kleio']:
      self.solution = solution
      if solution == 'kleio':
        self.num_reqs_per_ep = 100
        
      pages_misplaced = self.get_misplaced_pages()
      pages_ordered = self.get_ordered_pages(pages_misplaced)
      if solution == 'cori':
        cori_num_patterns = len(self.get_distinct_access_patterns())
  
      # Hybrid for first k pages
      self.run_scheduler('hybrid', pages_ordered[:cori_num_patterns])
  
      # Hybrid for first k group of pages
      selected_page_ids = self.select_k_page_groups(pages_ordered, cori_num_patterns)
      self.run_scheduler('hybrid-group', selected_page_ids)
    
      self.run_scheduler('oracle', [])
  
  def run_same_rnn_number(self):

    # run coeus, get the num patterns of coeus and then use these for kleio
    self.solution = 'coeus'
    pages_misplaced = self.get_misplaced_pages()
    pages_ordered = self.get_ordered_pages(pages_misplaced)
    # number of RNNs is unique patterns of pages misplaced != total number of patterns
    max_num_rnns = len(self.get_distinct_access_patterns(pages_ordered))
    selected_page_ids = self.select_k_page_groups(pages_ordered, max_num_rnns)
    self.run_scheduler('hybrid', selected_page_ids, max_num_rnns)

    self.solution = 'kleio'
    self.num_reqs_per_ep = 100
    pages_misplaced = self.get_misplaced_pages()
    pages_ordered = self.get_ordered_pages(pages_misplaced)
    selected_page_ids = pages_ordered[:max_num_rnns]  # same rnns are coeus
    self.run_scheduler('hybrid', selected_page_ids, max_num_rnns)


  def run_max_rnn_number(self):
    # run coeus, get the num patterns of coeus and then use these for kleio
    self.solution = 'coeus'
    pages_misplaced = self.get_misplaced_pages()
    pages_ordered = self.get_ordered_pages(pages_misplaced)
    # number of RNNs is unique patterns of pages misplaced != total number of patterns
    max_num_rnns = len(self.get_distinct_access_patterns(pages_ordered))
    selected_page_ids = self.select_k_page_groups(pages_ordered, max_num_rnns)
    self.run_scheduler('hybrid', selected_page_ids, max_num_rnns) # oracle

    self.solution = 'kleio'
    self.num_reqs_per_ep = 100
    pages_misplaced = self.get_misplaced_pages()
    pages_ordered = self.get_ordered_pages(pages_misplaced)
    self.run_scheduler('hybrid', pages_ordered, len(pages_ordered)) # oracle

      

      
      
      
      
      
      