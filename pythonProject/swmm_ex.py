from concurrent.futures import ThreadPoolExecutor
from pyswmm import Simulation
import os
import time

def run_simulation(sim_path):
    
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

def swmm_ex_batch(folder_path):
    with ThreadPoolExecutor(max_workers=16) as executor:
        inp_files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.inp')]
        executor.map(run_simulation, inp_files)


if __name__ == '__main__':
    folder_path = '.\\pythonProject\\sim_test'
    swmm_ex_batch(folder_path)


## Test this code from https://github.com/pyswmm/pyswmm/issues/222
# import os
# import pyswmm
# import multiprocessing as mp

# def worker(swmm_filename):
#     print("Starting:", swmm_filename)
#     sim = pyswmm.Simulation(swmm_filename)
#     sim.execute()

# def main():
#     swmm_basenames = [ "Example1.inp", "Example2.inp", "Example3.inp", "Example4.inp" ]
#     swmm_filenames = [ os.path.join(os.getcwd(), basename)  for basename in swmm_basenames ]

#     processes = []
#     for swmm_filename in swmm_filenames:
#         p = mp.Process(target=worker, args=(swmm_filename, ))
#         processes.append(p)
#         p.start()

#     [ p.join() for p in processes ]

#     print("All done")

# if __name__ == '__main__':
#     main()