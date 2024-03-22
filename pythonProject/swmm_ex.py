from concurrent.futures import ThreadPoolExecutor
from pyswmm import Simulation
import os
import time

def swmm_ex_batch(folder_path):
    i=1
    inp_files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.inp')]
    for sim_path in inp_files:
        with Simulation(sim_path) as sim:
            print("\nSimulation info:")
            print(f"Title: {sim_path}")
            print("Start Time: {}".format(sim.start_time))
            start_time = time.time()
            for step in sim:
                pass
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed Time: ", elapsed_time)
            print("Simulation done:", i, " of ", len(inp_files))
            i += 1

if __name__ == '__main__':
    folder_path = '.\\pythonProject\\sim_test'
    swmm_ex_batch(folder_path)
