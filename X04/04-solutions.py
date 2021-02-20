# =========================================================================== #

# data source (webpage):
# https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
# data source (data file):
# https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.xls

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sciConsts
from scipy.optimize import curve_fit

# read the data into memory
with open("astmg173.csv", "r") as handle :
    reader = csv.reader(handle)
    
    title = next(reader)
    heads = next(reader)
    
    wavelengths = []
    intensities = []
    
    for line in reader :
        wavelengths.append(float(line[0]))
        intensities.append(float(line[1]))

wavelengths = np.array(wavelengths)
intensities = np.array(intensities)

# shorthands for nature constants
h  = sciConsts.h
c  = sciConsts.c
kB = sciConsts.Boltzmann

# define the fit form
def planck(wavelength, T) :
    # see https://en.wikipedia.org/wiki/Planck%27s_law
    wl = 1e-9 * wavelength      # input is in nanometers; we need SI units
    
    return  (2 * h * c**2) / (wl**5) * \
            1E-13 / (np.exp( (h * c) / (wl * kB * T) ) - 1)

T_correct = 5778

xct = planck(wavelengths, T_correct)

popt, pcov = curve_fit(planck, wavelengths, intensities, p0 = (5000,))
fit = planck(wavelengths, *popt)

print(popt)

plt.plot(wavelengths, intensities, label="ASTMG173 data")
plt.plot(wavelengths, fit, label=f"fit with {popt[0]:4.0f} K")
plt.plot(wavelengths, xct, label=f"curve for literature value {T_correct} K")
plt.xlabel(heads[0])
plt.ylabel(heads[1])
plt.legend()
plt.show()

## =========================================================================== #
#
#import numpy as np
#from scipy.integrate import odeint
#import matplotlib.pyplot as plt
#
#def gField_acceleration (state, t) :
#    rx, ry, vx, vy, = state
#    
##    print("v =", (vx, vy))
##    print("r =", (rx, ry))
##    print()
#    
#    r = np.sqrt(rx**2 + ry**2)
#    rx /= r
#    ry /= r
#    
#    g = 1/(r**2)
#    
#    ax = -g * rx
#    ay = -g * ry
#    
#    return [vx, vy, ax, ay]
#
#state_0 = np.array([0, 1, .4, 0])
#
#N = 1000
#t = np.linspace(0, 10, N)
#
#sol = odeint(gField_acceleration, state_0, t)
#
#
#plt.plot(0, 0, 'ro')
#plt.plot(0, 1, 'bo')
#plt.plot(sol[:, 0], sol[:, 1])
#plt.show()
#
## =========================================================================== #
#
#import numpy as np
#from scipy.signal    import convolve
#from scipy.integrate import odeint
#import matplotlib.pyplot as plt
#
#laplacian_matrix = np.array([[ 0,  1,  0],
#                             [ 1, -4,  1],
#                             [ 0,  1,  0]])
#
#def heatODE(T, t, alpha, Nx, Ny) :
#    # dT/dt = alpha laplacian T
#    
#    T = T.reshape((Nx, Ny))
#    dT = alpha * convolve(T, laplacian_matrix, mode='same')
#    T = T.reshape((Nx * Ny,))
#    
#    dT[0, :] = 0
#    dT = dT.reshape((Nx * Ny,))
#    
#    return dT
#
#Nx = 50
#Ny = 50
#T0 = np.zeros( (Nx, Ny) )
#T0[0, :] = 10
#T0 = T0.reshape(Nx * Ny)
#
#
#Nt = 100
#t = np.linspace(0, 100, Nt)
#
#alpha = 5
#
#sol = odeint(heatODE, T0, t, args=(alpha, Nx, Ny))
#
#sol = sol.reshape((Nt, Nx, Ny))
#
#plt.figure()
#plt.pcolor(sol[0])
#plt.show()
#
#plt.pcolor(sol[1])
#plt.show()
#
#plt.pcolor(sol[-1])
#plt.show()
