from modules.swmm_ex import swmm_ex_multiprocessing as ex
import os

# Specify the path to the folder containing the inp files for the simulations
sim_path = os.path.join('03_sim_data', 'inp_1d_max')

ex.swmm_mp(sim_path)