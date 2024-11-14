from refractiveindex import RefractiveIndexMaterial
import frenel
import numpy as np
import green_func
import dipoles
import matplotlib.pyplot as plt
from tqdm import tqdm

wavelength_nm = 600  # [nm]
Si = RefractiveIndexMaterial(shelf='main', book='Si', page='Green-2008')
Au = RefractiveIndexMaterial(shelf='main', book='Au', page='Johnson')

STOP = np.inf
R = 146*1e-9
point = [0,0,20*1e-9+R]
angle = 25*np.pi/180
phase = 0
a = 0


print(green_func.green_ref_00(wavelength_nm, point[2], Au))

#wl = np.linspace(600, 1200, 10)


# for wl_i in wl:
#     print(green_func.green_ref_00(wl_i, point[2], Au))
