#!/bin/sh
#
#
. /etc/profile
module load julia/1.6.1
module load gurobi/gurobi-811
echo "My SLURM_ARRAY_TASK_ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE
julia solve_dlr_project.jl $LLSUB_RANK $LLSUB_SIZE