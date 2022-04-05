#!/bin/bash

sudo insmod mvnc_inception.ko input_graph="/home/edwardhu/kava/driver/kava-benchmarks/ml/mvnc/kernel/data/inception_v3_movidius.graph" input_labels="/home/edwardhu/kava/driver/kava-benchmarks/ml/mvnc/kernel/data/imagenet_slim_labels.txt" input_image="/home/edwardhu/kava/driver/kava-benchmarks/ml/mvnc/kernel/data/grace_hopper.jpg" total_images=1 batch_mode=0
