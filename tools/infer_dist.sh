#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=1
workdir=.
OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/inference.py \
	 $workdir/configs/svg/svg_pointT.yaml  $workdir/configs/svg/best.pth --out ./results/floorplan_wo_layers --datadir dataset/json/floorplan_wo_layer
