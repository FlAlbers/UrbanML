## Test this code from https://github.com/pyswmm/pyswmm/issues/222
import os
import pyswmm
import multiprocessing as mp
from multiprocessing import Lock
import time

def worker(swmm_filename):
    lock = Lock()
    lock.acquire()
    try:
        start_time = time.time()
        sim = pyswmm.Simulation(swmm_filename)
        
        sim.execute()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\n\nStarting: ", os.path.basename(swmm_filename))
        # print("Start Time: {}".format(sim.start_time))
        print("Elapsed simulation time: ", elapsed_time)
    finally:
        lock.release()

def main():
    inp_files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.inp')]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(worker, inp_files)

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print("\n\nTotal simulation time: ", total_elapsed_time)
    print("All done")

if __name__ == '__main__':
    total_start_time = time.time()
    folder_path = 'pythonProject\\inp'
    os.getcwd()
    main()