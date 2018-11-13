#!/bin/bash

. setup.sh

# Get the IP address of our head node
headIP=$(ip addr show ipogif0 | grep '10\.' | awk '{print $2}' | awk -F'/' '{print $1}')

# Use a unique cluster ID for this job
clusterID="cori_${SLURM_JOB_ID}"
echo "Launching IPyParallel Controller..."
./start_kale_task.py --mhost="$1" --mport="$2" --ipcontroller --task_args="--ip=\"$headIP\" --cluster-id=\"$clusterID\"" &
 
sleep 20

echo "Launching $3 IPyParallel Engines..."
srun -N "$3" ./start_kale_task.py --mhost="$1" --mport="$2" --ipengine --task_args="--cluster-id=\"$clusterID\""
