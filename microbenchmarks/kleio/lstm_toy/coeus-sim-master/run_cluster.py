import sys, csv
from multiprocessing import Process
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sim.perf_model import *
from scipy.cluster.hierarchy import dendrogram, linkage
from kleio.page_selector import *

class Cluster:
  def __init__(self, trace_file):
    self.trace_file = trace_file
    self.prof = PProfile(trace_file)
    self.features_initial = []
    self.features_standard = []
    self.inertia_curve = []
    self.max_clusters = 0
    
  def cluster_all_pages(self):
    self.prof.init()
    sim = PerfModel(self.prof, 'Fast:NearSlow', 'oracle', 0.2, 100) # kleio period duration
    sim.init()
    # features: per page access counts across periods
    for page in self.prof.hmem.page_list:
      self.features_initial.append(page.oracle_counts_binned_ep)
    self.get_patterns(self.prof, range(self.prof.hmem.num_pages))
    
  def cluster_misplaced_pages(self):
    self.prof.init()
    sim = PerfModel(self.prof, 'Fast:NearSlow', 'history', 0.2, 100) # kleio period duration
    sim.init()
    sim.run()
    # Get misplaced pages eligible for RNNs.
    page_selector = PageSelector(self.prof, 'Fast:NearSlow', '0.2', 100, '') # with Kleio's frequency
    pages_misplaced = page_selector.get_misplaced_pages_sim()
    pages_ordered = page_selector.get_ordered_pages(pages_misplaced)
    # features: per page access counts across periods
    for page_id in pages_ordered:
      page = self.prof.hmem.page_list[page_id]
      self.features_initial.append(page.oracle_counts_binned_ep)
    self.get_patterns(self.prof, pages_ordered)
      
  def prepare_features(self):
    # normalize
    scaler = preprocessing.Normalizer()
    self.features_standard = scaler.fit_transform(self.features_initial)
    
  def kmeans_curve(self, cluster_sizes):
    for i in cluster_sizes:
      kmeans = KMeans(n_clusters=i)
      kmeans.fit(self.features_standard)
      self.inertia_curve.append(kmeans.inertia_)
  
  def kmeans(self, num_clusters):
    km = KMeans(n_clusters=num_clusters)
    km.fit(self.features_standard)
    print km.inertia_

  def get_patterns(self, prof, page_ids):
    distinct_page_cnts_seq = set()
    for id in page_ids:
      page = prof.hmem.page_list[id]
      if str(page.oracle_counts_binned_ep) not in distinct_page_cnts_seq:
        distinct_page_cnts_seq.add(str(page.oracle_counts_binned_ep))
    self.max_clusters = len(distinct_page_cnts_seq)
    
def dump_stats(resfile, line):
  w = csv.writer(open(resfile, "a"))
  w.writerow(line)
  
def run_cluster_app(trace_file, resfile, app_label):
  cluster = Cluster(trace_file)
  cluster.cluster_misplaced_pages()
  cluster.prepare_features()
  cluster_sizes = range(2, cluster.max_clusters, 50) + [cluster.max_clusters]
  cluster.kmeans_curve(cluster_sizes)
  dump_stats(resfile, [app_label, cluster_sizes, cluster.inertia_curve])

if __name__ == "__main__":
  trace_dir = sys.argv[1]
  
  apps = ['backprop_10000', 'kmeans_5000', 'hotspot_256', 'quicksilver_500', 'cpd_10000', 'lud_512', 'bfs_128k', 'bptree_100k', 'pennant_leblanc']
  app_labels = ['backprop', 'kmeans', 'hotspot', 'quicksilver', 'cpd', 'lud', 'bfs', 'bptree', 'pennant']
  
  resfile = 'cluster_inertia_apps.csv'
  
  threads = []
  for app_idx in range(1):
    app = apps[app_idx]
    app_label = app_labels[app_idx]
    trace_file = trace_dir + 'trace_' + app + '.txt'
    p = Process(target=run_cluster_app, args=(trace_file, resfile, app_label))
    p.start()
    threads.append(p)
  for thr in threads:
    thr.join()
    
