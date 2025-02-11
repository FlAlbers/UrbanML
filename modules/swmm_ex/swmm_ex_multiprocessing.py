"""
Author: Flemming Albers
Script to run SWMM simulations in parallel using multiprocessing.
To use this script only the 'folder_path' is needed that points to the directory containing the SWMM input files.
"""

import os
import pyswmm
import multiprocessing as mp
from multiprocessing import Lock
import time

def worker(swmm_filename):
    """
    Executes a SWMM simulation for a given SWMM input file.
    Output and report file will be generated in the same directory as the input file.

    Args:
        swmm_filename (str): The path to the SWMM input file.
    Returns:
        None
    """
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

def swmm_mp(folder_path):
    """
    Main function that executes the SWMM simulations in parallel using multiprocessing.
    """
    total_start_time = time.time()
    inp_files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.inp')]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(worker, inp_files)

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print("\n\nTotal simulation time: ", total_elapsed_time)
    print("All done")

if __name__ == '__main__':
    folder_path = '03_sim_data\\sim_test'
    swmm_mp(folder_path)