from pyswmm import Simulation
import os




folder_path = '.\\pythonProject\\inp'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.inp'):
        sim_path = os.path.join(folder_path, file_name)
        
        with Simulation(sim_path) as sim:
            print("Simulation info:")
            print("Start Time: {}".format(sim.start_time))
            print("End Time: {}".format(sim.end_time))
            
            for step in sim:
                pass

