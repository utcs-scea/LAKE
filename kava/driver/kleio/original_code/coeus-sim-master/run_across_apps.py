import sys
from multiprocessing import Process
from sim.perf_model import *
from kleio.page_selector import *

def run_thread(prof, platform_name, ratio, reqs_per_ep, resdir_prefix):
  page_selector = PageSelector(prof, platform_name, ratio, reqs_per_ep, resdir_prefix)
  #page_selector.run_compare()
  page_selector.run_same_rnn_number()

if __name__ == "__main__":
  # Input
  trace_dir = sys.argv[1]
  resdir = sys.argv[2]
  
  apps = ['backprop_10000', 'kmeans_5000', 'hotspot_256', 'quicksilver_500', 'cpd_10000', 'lud_512', 'bfs_128k', 'bptree_100k', 'pennant_leblanc']
  app_labels = ['backprop', 'kmeans', 'hotspot', 'quicksilver', 'cpd', 'lud', 'bfs', 'bptree', 'pennant']
  
  cori_best_perf_reqs_apps = [9000, 26000, 24900, 31100, 35500, 23200, 12400, 27600, 11700]
  cori_dom_reuse_reqs_apps = [9000, 2961, 8384, 31262, 35650, 1938, 2495, 3519, 11738]

  profiles = {}
  for app in apps:
    prof = PProfile(trace_dir + 'trace_' + app + '.txt')
    prof.init()
    profiles[app] = prof

  threads = []
  for app_idx in range(len(apps)):
    app = apps[app_idx]
    app_label = app_labels[app_idx]
    cori_reqs_per_ep = cori_dom_reuse_reqs_apps[app_idx]
    p = Process(target=run_thread, args=(profiles[app], 'Fast:NearSlow', 0.2, cori_reqs_per_ep, resdir + app_label + '_'))
    p.start()
    threads.append(p)
  for thr in threads:
    thr.join()
