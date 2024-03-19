from pyswmm import Simulation
import os

def swmm_ex_batch(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.inp'):
            sim_path = os.path.join(folder_path, file_name)
            
            with Simulation(sim_path) as sim:
                print("Simulation info:")
                print(f"Title: {file_name}")
                print("Start Time: {}".format(sim.start_time))
                print("End Time: {}".format(sim.end_time))
                
                for step in sim:
                    pass

if __name__ == '__main__':
    folder_path = '.\\pythonProject\\sim_test'
    swmm_ex_batch(folder_path)
