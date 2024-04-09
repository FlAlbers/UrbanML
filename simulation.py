from modules.swmm_ex import swmm_ex_multiprocessing as ex
import os

sim_path = os.path.join('03_sim_data', 'inp')

ex.swmm_mp('03_sim_data\\inp')