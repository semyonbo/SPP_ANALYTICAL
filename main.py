import numpy as np
import force
import matplotlib.pyplot as plt
from tqdm import tqdm
import frenel
import dipoles
import green_func


eps_Au = frenel.get_interpolate('Au')
eps_Si = frenel.get_interpolate('Si')

wl = 800  # [nm]
STOP = 45
R = 146
dist = 20
point = [0,0,dist+R]
angle = 25*np.pi/180
phase = 0
a = 0.5


F =  force.F(wl, eps_Au, point, R, eps_Si, angle,amplitude=1,phase=phase,a=a, stop=STOP, full_output=True)

# dz_rot_G_E, dz_rot_G_H = green_func.dz_rot_green_E_H(wl, point[2], eps_Au, STOP)

# print(dz_rot_G_E)

print('F_x', F[0,0])
print('Fx_e0', F[0,1])
print('Fx_e1', F[0,2])
print('Fx_e2', F[0,3])
print('Fx_m0', F[0,4])
print('Fx_m1', F[0,5])
print('Fx_m2', F[0,6])
print('Fx_crest', F[0,7])
print('------------')
print('F_y', F[1,0])
print('Fy_e0', F[1,1])
print('Fy_e1', F[1,2])
print('Fy_e2', F[1,3])
print('Fy_m0', F[1,4])
print('Fy_m1', F[1,5])
print('Fy_m2', F[1,6])
print('Fy_crest', F[1,7])


# print(green_func.dy_green_E_H_yz(wl,point[2], eps_Au, STOP))