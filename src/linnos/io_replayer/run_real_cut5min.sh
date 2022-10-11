#!/bin/bash

#sudo ./replayer baseline /dev/nvme0n1p1-/dev/nvme1n1p1-/dev/nvme2n1p1 ./testTraces/testdrive0.trace-cut.trace ./testTraces/testdrive1.trace-cut.trace ./testTraces/testdrive2.trace-cut.trace outputs/real_5min
sudo ./replayer failover /dev/nvme0n1p1-/dev/nvme1n1p1-/dev/nvme2n1p1 ./testTraces/testdrive0.trace-cut.trace ./testTraces/testdrive1.trace-cut.trace ./testTraces/testdrive2.trace-cut.trace outputs/real_5min