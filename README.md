# agent_based_hotelling

## Installation instructions

Copy the following instructions to clone the repo and install dependencies in virtual environment
```
git clone git@github.com:LordOfLuck/agent_based_hotelling.git
cd agent_based_hotelling
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run simulations
Once the virtual environment is active (`source venv/bin/activate`), run the following:
```
python simulation.py -h  # see help
python simulation.py --discount_rates "0 0.8" --n_jobs 4  # run simulation for discount rate combination [0, 0.8] with 4 processes
```
To run simulations for all discount rate combinations at once, run the following shell script

```
chmod +x run_all.sh
./run_all.sh
```
