# Coeus: Page clustering mechanism for machine learning-based hybrid memory management

## Run Coeus
```
python run_across_apps.py
```

## Run k-means
```
python run_cluster.py
```

## Run Coeus + Kleio
```
python run_cluster_lstm.py
```

This repo contains the following code:
- `sim/` includes the hybrid memory simulation created in Cori and validated against an Intel Optane PMEM platform.
- `traces/` includes memory access traces used in the experimental evaluation of Cori and Coeus.
- `kleio/` includes code that replicates the system design of Kleio as reported in the paper. It is not the original code developped while at AMD. 


## Paper references
- <b>Coeus: Clustering (A)like Patterns for PracticalMachine Intelligent Hybrid Memory Management.</b> <br/>
Thaleia Dimitra Doudali, Ada Gavrilovska.<br/>
In Proceedings of the 22nd IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGrid 2022).

- <b>Cori: Dancing to the Right Beat of Periodic Data Movements over Hybrid Memory Systems.</b><br/>
Thaleia Dimitra Doudali, Daniel Zahka, Ada Gavrilovska. <br/>
In Proceedings of the 35th IEEE International Parallel and Distributed Processing Symposium (IPDPS 2021).

- <b>Kleio: a Hybrid Memory Page Scheduler with Machine Intelligence.</b><br/>
Thaleia Dimitra Doudali, Sergey Blagodurov, Abhinav Vishnu, Sudhanva Gurumurthi, Ada Gavrilovska. <br/>
In Proceedings of the 28th International Symposium on High-Performance Parallel and Distributed Computing (HPDC 2019).
