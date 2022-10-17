#!/bin/bash
set -o pipefail

kernel_version=$(uname -r)
echo $kernel_version
if [ "$kernel_version" != "6.0.0-lake" ]; then
  echo "Error : Required Kernel Version not found"
fi

nvidia_check=$(nvidia-smi | grep Driver)
if [ "$nvidia_check" == "" ]; then
  echo "Error : Driver not found"
fi
