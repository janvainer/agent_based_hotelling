import nashpy as ns
import time, csv, pickle
from tabulate import tabulate

from nashq_agent import NashQ
from next_state import next_state
from profit_function import profit

# allows to save an potentially again retrieve the agents with their learned valules
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file with given name
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def simulate(run, iterations = 10000, init_state = (3,3), param = [0.5,0.5]):
    print(f"Starting run {run} for parameters {param}")

    # run is the order of this simulation
    # Parameters of agents
    p = param
    # price agents = contextual Nash-bandits, location agents = Nash-Q agents
    # 5 possible prices, 3 possible movements (-1,0,1)
    # PA1 and LA1 together represent the first agent etc.
    PA1 = NashQ(5, l = p[0])
    PA2 = NashQ(5, l = p[1])
    LA1 = NashQ(3, l = p[0])
    LA2 = NashQ(3, l = p[1])

    # data from simulation
    data = []

    # initial state
    state = init_state
    iter = iterations

    # otherwise PAs never see the first state due to our loop
    PA1._insert_new_state(state)
    PA2._insert_new_state(state)

    # Run the simulation cycle
    for i in range(iter):

        # move = next move decided by an agent, eula = expected utility of the agent
        a1, eula1 = LA1.next_action(state)
        a2, eula2 = LA2.next_action(state)

        # calculate new state from location actions
        next_st = next_state(state,a1,a2)

        # choose prices in the new state, and also return expected utilities
        p1, eupa1 = PA1.next_action(next_st)
        p2, eupa2 = PA2.next_action(next_st)

        # calculate profits
        r1,r2 = profit(next_st,p1,p2)
        # Give feedback for their actions
        PA1.take_response(r1,r2,next_st,p1,p2,next_st)
        PA2.take_response(r2,r1,next_st,p2,p1,next_st)
        LA1.take_response(r1,r2,state,a1,a2,next_st)
        LA2.take_response(r2,r1,state,a2,a1,next_st)

        # set next state according to agents' actions
        state = next_st
        # Append a data row from particular simulation round  to the data
        data.append([state[0],state[1],p1,p2, r1,r2, eula1, eula2, eupa1,eupa2])
        # print to watch progress real-time
        # print([state[0],state[1],p1,p2, r1,r2, eula1, eula2, eupa1,eupa2])

    # export to data and agent folders
    _export(data,run,p,PA1, PA2, LA1, LA2)

# create folders data and agents in the code folder to be able to retrieve the simulation data
def _export(data,run,p, PA1, PA2, LA1, LA2):
    # insert headers
    data.insert(0,["Location 1","Location 2","Price 1", "Price 2","Profit 1", "Profit 2", "eula1", "eula2", "eupa1", "eupa2"])

    # create unique data name
    data_name = "run_"+str(run)
    for i in range(2):
        data_name += "_"+str(p[i])

    csvfile = "./data/"+data_name+".csv"

    # save data as csv
    with open(csvfile,"w") as output:
        writer = csv.writer(output, lineterminator ='\n')
        writer.writerows(data)

    # agent names
    name11 = "./agents/PA1_"+data_name
    name12 = "./agents/LA1_"+data_name
    name21 = "./agents/PA2_"+data_name
    name22 = "./agents/LA2_"+data_name

    # save objects with unique name
    save_object(PA1,name11)
    save_object(PA2,name21)
    save_object(LA1,name12)
    save_object(LA2,name22)


if __name__ == "__main__":
    import time
    import argparse
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser()
    parser.add_argument("--discount_rates", choices={"0 0", "0 0.4", "0 0.8", "0.4 0.4", "0.4 0.8", "0.8 0.8"})
    parser.add_argument("--simulation_repeats", default=100, type=int)
    parser.add_argument("--n_iters", default=30000, type=int)
    parser.add_argument("--n_jobs", default=1, type=int)
    args = parser.parse_args()
    param = [float(dr) for dr in args.discount_rates.split()]


    if args.n_jobs == 1:
        for run in range(args.simulation_repeats):
            simulate(run, iterations=args.n_iters, param=param)
    elif args.n_jobs > 1:
        Parallel(n_jobs=args.n_jobs)(delayed(simulate)(run, args.n_iters, param) for run in range(args.simulation_repeats))
    else:
        raise ValueError("n_jobs must be a positive integer")

