import frenel
import numpy as np
from green_func import green_ref_00, green_ref_00_integrand, rot_green_ref_00
import matplotlib.pyplot as plt
from cmath import sqrt
from scipy.integrate import quad
import dipoles

wls = np.linspace(500, 1000, 20)
# wli=800
eps_Si = frenel.get_interpolate('Si')

z0 = (14+20)

dipoles.get_alpha(14, eps_Si, 800)
# eps_Au = frenel.get_interpolate('Au')
# res0 = []
# res1 = []
# res2 = []
# rot1 = []
# rot2 = []
# rot3 = []


# for wli in wls:
# #     # print(quad( lambda kr: green_ref_00_integrand(kr, wli, z0, eps_Au)[0][0, 0], 0, 1, complex_func=True))
    
#     res = green_ref_00(wli, z0, eps_Au)[0]

#     rot = rot_green_ref_00(wli, z0, eps_Au)[0]

#     res0.append(res[0, 0])
#     res1.append(res[1, 1])
#     res2.append(res[2, 2])
    
#     rot1.append(rot[0, 1])
#     rot2.append(rot[1, 0])

#     res0.append(green_ref_00(wli, z0, eps_Au)[0][2,2])


# plt.plot(wls, np.real(res1)/np.max(np.abs(res0)), label='Gref10')
# plt.plot(wls, np.real(res1)/np.max(np.abs(res1)), label='Gref11')
# plt.plot(wls, np.real(res2)/np.max(np.abs(res2)), label='Gref22')
# plt.legend()
# plt.show()


# plt.plot(wls, np.real(rot1)/np.max(np.abs(rot1)), label='Grot10')
# plt.plot(wls, np.real(rot2)/np.max(np.abs(rot1)), label='Grot11')

# plt.legend()
# plt.show()
