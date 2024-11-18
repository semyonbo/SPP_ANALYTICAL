import numpy as np
import dipoles
import matplotlib.pyplot as plt
from tqdm import tqdm
import frenel

eps_Au = frenel.get_interpolate('Au')
eps_Si = frenel.get_interpolate('Si')

wavelength_nm = 800  # [nm]
STOP = 20
R = 14
dist = 20
point = [0,0,dist+R]
angle = 25*np.pi/180
phase = 0
a = 0.5


p, m = dipoles.calc_dipoles_v2(wavelength_nm, eps_Au, point, R, eps_Si, angle, amplitude=1, phase=phase, a=a, stop=STOP )

print(np.round(p,37))

print(np.round(m,32))


p, m = dipoles.calc_dipoles(wavelength_nm, eps_Au, STOP, R , eps_Si, point , angle, amplitude=1, phase=phase, a=a )
print(np.round(p,37))

print(np.round(m,32))