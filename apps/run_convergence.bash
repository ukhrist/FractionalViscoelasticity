#!/usr/bin/env bash

read -p "Alpha: " alpha
read -p "Maxindex (1000*2^i): " maxindex
export PYTHONPATH=.

python apps/script_Convergence_IC.py $alpha

for ((i = 0 ; i <= $maxindex; i++)); do
    echo "Running n =" $i
    mpirun -np 1 python apps/script_Convergence.py $alpha $i $maxindex&
done

wait
echo "All done!"