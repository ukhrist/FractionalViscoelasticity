#!/usr/bin/env bash

read -p "Alpha: " alpha
read -p "Maxindex (1000*2^i): " maxindex
read -p "Correct IC: " correct
export PYTHONPATH=.

python apps/script_Convergence_IC.py $alpha $correct

for ((i = 0 ; i <= $maxindex; i++)); do
    echo "Running n =" $i
    mpirun -np 1 python apps/script_Convergence.py $alpha $i $maxindex $correct&
done

wait
echo "All done!"