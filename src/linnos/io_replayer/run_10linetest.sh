#!/bin/bash

#sudo ./replayer baseline /dev/nvme0n1p1-/dev/nvme1n1p1-/dev/nvme2n1p1 ./testTraces/100lines.trace ./testTraces/100lines.trace ./testTraces/100lines.trace outputs/test10output
sudo ./replayer failover /dev/vdb-/dev/vdb-/dev/vdb ./testTraces/100lines.trace ./testTraces/100lines.trace ./testTraces/100lines.trace outputs/test10output
