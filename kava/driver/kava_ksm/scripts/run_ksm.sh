#!/bin/bash

batch_size=$1

echo 'Setting sleep time'
echo 0 | sudo tee /sys/kernel/mm/ksm/sleep_millisecs

echo 'Setting pages_to_scan'
echo 4096 | sudo tee /sys/kernel/mm/ksm/pages_to_scan

# Tune the max_page_sharing parameter to set how many pages can share a single
# node in the stable tree. Nodes in the stable tree are "KSM pages" according to
# the Linux documentation. This is the metric which the ksm sysfs interface
# calls pages_shared.
#
# i.e. for 10000 identical pages with max_page_sharing = 2, we will have
# pages_shared = 5000, aka 5000 nodes in the stable tree.
#
# For more info, see the documentation at:
# https://github.com/torvalds/linux/blob/master/Documentation/admin-guide/mm/ksm.rst

echo 'Setting page sharing limit'
echo 256 | sudo tee /sys/kernel/mm/ksm/max_page_sharing

echo 'Setting run'
echo 1 | sudo tee /sys/kernel/mm/ksm/run
