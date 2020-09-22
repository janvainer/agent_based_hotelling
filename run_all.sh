#!/bin/bash
source venv/bin/activate

for dr in "0 0" "0 0.4" "0 0.8" "0.4 0.4" "0.4 0.8" "0.8 0.8"; 
do
    echo  Running simulations for discount rates $dr
    echo ============================================
    python3 simulation.py --discount_rates "$dr" --n_jobs 20;
done
