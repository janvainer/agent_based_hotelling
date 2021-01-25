import os
import time
import argparse
from joblib import Parallel, delayed

import nashpy as ns
import time, csv, pickle
from tabulate import tabulate

from nashq_agent import NashQ
from next_state import next_state
from profit_function import profit
from agents import RandomAgent, QAgent

AGENTS = {
    "random": RandomAgent,
    "q-learn": QAgent,
    "nashq-learn": NashQ,
}

# allows to save an potentially again retrieve the agents with their learned valules
def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file with given name
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class Simulation:
    def __init__(self, PA1, PA2, LA1, LA2, save_folder, init_state=(3, 3)):
        # price agents = contextual Nash-bandits, location agents = Nash-Q agents
        # 5 possible prices, 3 possible movements (-1,0,1)
        # PA1 and LA1 together represent the first agent etc.
        self.PA1 = PA1
        self.PA2 = PA2
        self.LA1 = LA1
        self.LA2 = LA2
        self.init_state = init_state
        self.param = (PA1.l, PA2.l)
        self.save_folder = save_folder

    def reset(self):
        self.PA1.reset()
        self.PA2.reset()
        self.LA1.reset()
        self.LA2.reset()

    def run(self, run, iterations=10000, init_state=(3, 3)):
        # data from simulation
        data = []

        # initial state
        state = self.init_state

        # otherwise PAs never see the first state due to our loop
        self.PA1._insert_new_state(state)
        self.PA2._insert_new_state(state)

        # Run the simulation cycle
        for i in range(iterations):

            # move = next move decided by an agent, eula = expected utility of the agent
            a1, eula1 = self.LA1.next_action(state)
            a2, eula2 = self.LA2.next_action(state)

            # calculate new state from location actions
            next_st = next_state(state, a1, a2)

            # choose prices in the new state, and also return expected utilities
            p1, eupa1 = self.PA1.next_action(next_st)
            p2, eupa2 = self.PA2.next_action(next_st)

            # calculate profits
            r1, r2 = profit(next_st, p1, p2)
            # Give feedback for their actions
            self.PA1.take_response(r1, r2, next_st, p1, p2, next_st)
            self.PA2.take_response(r2, r1, next_st, p2, p1, next_st)
            self.LA1.take_response(r1, r2, state, a1, a2, next_st)
            self.LA2.take_response(r2, r1, state, a2, a1, next_st)

            # set next state according to agents' actions
            state = next_st
            # Append a data row from particular simulation round  to the data
            data.append(
                [state[0], state[1], p1, p2, r1, r2, eula1, eula2, eupa1, eupa2]
            )

        # export to data and agent folders
        _export(data, run, self.param, self.PA1, self.PA2, self.LA1, self.LA2, self.save_folder)


# create folders data and agents in the code folder to be able to retrieve the simulation data
def _export(data, run, p, PA1, PA2, LA1, LA2, save_folder):
    # insert headers
    data.insert(
        0,
        [
            "Location 1",
            "Location 2",
            "Price 1",
            "Price 2",
            "Profit 1",
            "Profit 2",
            "eula1",
            "eula2",
            "eupa1",
            "eupa2",
        ],
    )

    # create unique data name
    data_name = "run_" + str(run)
    for i in range(2):
        data_name += "_" + str(p[i])

    os.makedirs(save_folder, exist_ok=True)
    csvfile = os.path.join(save_folder, data_name + ".csv")

    # save data as csv
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(data)

    # agent names
    os.makedirs(os.path.join(save_folder, "agents"), exist_ok=True)
    name11 = os.path.join(save_folder, "agents/PA1_" + data_name)
    name12 = os.path.join(save_folder, "agents/LA1_" + data_name)
    name21 = os.path.join(save_folder, "agents/PA2_" + data_name)
    name22 = os.path.join(save_folder, "agents/LA2_" + data_name)

    # save objects with unique name
    save_object(PA1, name11)
    save_object(PA2, name21)
    save_object(LA1, name12)
    save_object(LA2, name22)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discount_rates",
        choices={"0 0", "0 0.4", "0 0.8", "0.4 0.4", "0.4 0.8", "0.8 0.8"},
    )
    parser.add_argument("--simulation_repeats", default=100, type=int)
    parser.add_argument("--n_iters", default=30000, type=int)
    parser.add_argument("--n_jobs", default=1, type=int)
    parser.add_argument("--agent", default="nashq-learn", choices=["random", "q-learn", "nashq-learn"])
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    param = [float(dr) for dr in args.discount_rates.split()]

    agent = AGENTS[args.agent]
    save_folder = f"data/{args.agent}"
    simulation = Simulation(
        PA1=agent(5, l=param[0]),
        PA2=agent(5, l=param[1]),
        LA1=agent(3, l=param[0]),
        LA2=agent(3, l=param[1]),
        save_folder=save_folder,
    )

    if args.n_jobs == 1:
        for run in range(args.simulation_repeats):
            print(f"run {run}, param {param}")
            simulation.reset()
            simulation.run(run, iterations=args.n_iters)
    elif args.n_jobs > 1:
        Parallel(n_jobs=args.n_jobs)(
            delayed(simulate)(run, args.n_iters, param)
            for run in range(args.simulation_repeats)
        )
    else:
        raise ValueError("n_jobs must be a positive integer")
