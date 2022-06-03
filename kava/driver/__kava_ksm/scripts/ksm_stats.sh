#!/bin/bash

echo 'KSM full scans:'
cat /sys/kernel/mm/ksm/full_scans

echo 'Number of pages to scan before ksmd sleeps:'
cat /sys/kernel/mm/ksm/pages_to_scan

echo 'Number of shared KSM pages being used:'
cat /sys/kernel/mm/ksm/pages_shared

echo 'How many more sites are sharing the pages (total savings):'
cat /sys/kernel/mm/ksm/pages_sharing

echo 'Number of pages unique but repeatedly checked for merging:'
cat /sys/kernel/mm/ksm/pages_unshared

echo 'Number of pages changing too fast to be placed in a tree:'
cat /sys/kernel/mm/ksm/pages_volatile

echo 'Number of KSM pages that hit the max_page_sharing limit:'
cat /sys/kernel/mm/ksm/stable_node_chains

echo 'Number of duplicated KSM pages:'
cat /sys/kernel/mm/ksm/stable_node_dups

echo '========== KAVA PERFORMANCE =========='

echo 'Time to scan all pages 100 times'
cat /sys/kernel/mm/ksm/scan_time

echo 'Throughput'
cat /sys/kernel/mm/ksm/throughput
