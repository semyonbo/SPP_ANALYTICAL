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

F =  force.F(wl, eps_Au, point, R, eps_Si, angle,amplitude=1,phase=phase,a=a, stop=STOP)


print(F)



# print(green_func.dy_green_E_H_yz(wl,point[2], eps_Au, STOP))