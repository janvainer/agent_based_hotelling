from Simulation import simulate
import time

# Run this file to start the simulation. Set the number of iterations in order
# to define, how many simulations of the same parameter are required

start = time.time()
for i in range(1):
	simulate(i, iterations=10000, param = [0.4,0.4])
end = time.time()
print(end-start)
