# =========================================================================== #
# problem 1: Curve Fitting

# data source (webpage):
# https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
# data source (data file):
# https://www.nrel.gov/grid/solar-resource/assets/data/astmg173.xls

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sciConsts
from scipy.optimize import curve_fit

print("CURVE FITTING")

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

exact = planck(wavelengths, T_correct)

popt, pcov = curve_fit(planck, wavelengths, intensities, p0 = (5000,))
fit = planck(wavelengths, *popt)

print(popt)

plt.plot(wavelengths, intensities, label="ASTMG173 data")
plt.plot(wavelengths, fit        , label=f"fit with {popt[0]:4.0f} K")
plt.plot(wavelengths, exact      , label=f"curve for literature value {T_correct} K")
plt.xlabel(heads[0])
plt.ylabel(heads[1])
plt.legend()
plt.show()

# =========================================================================== #
# Problem 2: JPEG principle

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.ascent.html

import scipy.misc
import scipy.fft
import matplotlib.pyplot as plt

print("FOURIER TRANSFORM")

ascent = scipy.misc.ascent()
FT = scipy.fft.fft2(ascent, norm='ortho')

fig = plt.figure( figsize=(10, 5) )
plt.gray()

drw = fig.add_subplot(121)
drw.set_title("Original Image")
drw.imshow(ascent)

drw = fig.add_subplot(122)
drw.set_title("Fourier Domain")
drw.imshow(np.log(np.abs(FT)))
fig.show()

# understand each pixel in the Fourier Domain as the amplitude of a wave with
# wave vector corresponding to the pixel coordinate.
# That is, Fourier coordinate (i,j) is a wave repeating every (i,j) steps.
# Note also that a wave more than half the size of the image will "roll over 
# from behind", hence the symmetrical structure of the Fourier Plot.


filterWidthHP = 32

highpass = FT.copy()
highpass[:filterWidthHP , :filterWidthHP ] = 0
highpass[-filterWidthHP:, -filterWidthHP:] = 0

recHigh = scipy.fft.ifft2(highpass, norm='ortho').real

fig = plt.figure( figsize=(10, 5) )

drw = fig.add_subplot(121)
drw.set_title("Reconstruction with High Pass Filter")
drw.imshow(recHigh)

drw = fig.add_subplot(122)
drw.set_title("Fourier Domain with High Pass Filter")
drw.imshow(np.log(np.abs(highpass)))
plt.show()

# Due to the symmetry involved, we need to block out both ends of the sepctrum.
# log(0) is undefined, and so will leave a white box in the Fourier Domain
# plots.
# In the reconstructed/filtered image, we see all structures are still 
# recognizeable, but all large-scale structures are essentially in the same
# shade, as these slow variations of colour were filtered out. The stairway as
# well as the plattform on top of the picture are the same shade, for example.


filterWidthLP = 96

lowpass = FT.copy()
lowpass[:filterWidthLP, -filterWidthLP:] = 0
lowpass[-filterWidthLP:, :filterWidthLP] = 0

recLow = scipy.fft.ifft2(lowpass, norm='ortho').real

fig = plt.figure( figsize=(10, 5) )

drw = fig.add_subplot(121)
drw.set_title("Reconstruction with Low Pass Filter")
drw.imshow(recLow)

drw = fig.add_subplot(122)
drw.set_title("Fourier Domain with Low Pass Filter")
drw.imshow(np.log(np.abs(lowpass)))
plt.show()

# In the low pass filter, on the other hand, we essentially only see the 
# plattform as it is substantially darker than the rest of the picture. All
# other details are the result of sharp contrast, which is due to quick
# oscillations which we filtered out.


mask = np.ones( shape=FT.shape, dtype=bool )
mask[:filterWidthHP , :filterWidthHP ] = False
mask[-filterWidthHP:, -filterWidthHP:] = False
mask[:filterWidthLP , -filterWidthLP:] = False
mask[-filterWidthLP:, :filterWidthLP ] = False
reduced = FT.copy()
reduced[mask] = 0
recReduced = scipy.fft.ifft2(reduced, norm='ortho').real

fig = plt.figure( figsize=(10, 5) )

drw = fig.add_subplot(121)
drw.set_title("Reconstruction from Reduced Data")
drw.imshow(recReduced)

drw = fig.add_subplot(122)
drw.set_title("Fourier Domain with Reduced Data")
drw.imshow(np.log(np.abs(reduced)))
plt.show()

# using only these bits we previously filtered out already allows to 
# reconstruct most of the image! It is a bit blurry, but look at the Fourier
# Domain plot: we've eliminated *by far most* of the information that used to
# be in the image! These four small squares are all what's needed to recreate
# an image of this quality. Making them bigger improves the image quality.
# You only need to find a good compromise between compression rate and 
# remaining image quality -- and bam, that's JPEG for you.

# =========================================================================== #
# Problem 3: orbit

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

print("ORBIT")

def gField_acceleration (state, t) :
    rx, ry, vx, vy, = state
    
    r = np.sqrt(rx**2 + ry**2)   # these two steps take long to evaluate
    g = 1/(r**3)                 # to speed things up, pre-compute them and read them from an array!
    
    ax = -g * rx
    ay = -g * ry
    
    return [vx, vy, ax, ay]

state_0 = np.array([0, 1, .4, 0])

N = 10000                         # more steps make accuracy better, but also require more computation time
t = np.linspace(0, N / 10, N)     # more time steps will eventually lead to aberant behaviour.

sol = odeint(gField_acceleration, state_0, t)

plt.plot(sol[:, 0], sol[:, 1])
plt.plot(0, 0, 'ro')
plt.plot(0, 1, 'bo')

# if we leave N = 1000 (corresponding to t=0..100), we see one nice orbit,
# nothing wrong there. However, with every subsequent step we accumulate more
# and more errors. They are the biggest near the sun (there, r is small, so 1/r
# is big, and the error we make gets multiplied by big factors), and after some
# 10 orbits we've noticeably deviated from the true solution.

plt.show()

# =========================================================================== #
# Problem 4: Heat Equation

import numpy as np
from scipy.signal    import convolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt

print("HEAT EQUATION")

deltaX = 1

laplacian_matrix = deltaX**2 * np.array(
        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]])

quadraticInX = np.array([[x**2 for x in range(5)] for row in range(5)])

print(quadraticInX)
nablaSquaredQIX = convolve(quadraticInX, laplacian_matrix, mode='same')
print(nablaSquaredQIX)

# the second derivative of x² is two, and that's what we see in the result
# matrix nablaSquaredQIX. However, the derivative is always only defined on
# open sets, never on the boundary of a region. When evaluating there, scipy
# uses a filler value, which essentially makes the corresponding values 
# useless. You could interpret this as reading matrix entries with negative
# coordinates.
# If you have SciPy Version 1.6.1 or higher, you can make it so that convolve
# uses cyclyical boundary conditions. This will allow you (for this one 
# example) to extend the range of valid cells to the top and bottom row, BUT
# NOT the left/right column.
# For us that does not matter, since we later set the derivative there to zero,
# anyway.

def heatODE(T, t, alpha, Nx, Ny) :
    T = T.reshape((Nx, Ny))
    dT = alpha * convolve(T, laplacian_matrix, mode='same')                    # dT/dt = alpha laplacian T
    
    dT[ 0,  :] = 0                                                             # dT in the boundaries is 0
    dT[-1,  :] = 0
    dT[ :,  0] = 0
    dT[ :, -1] = 0
    dT = dT.reshape((Nx * Ny,))
    
    return dT

plt.set_cmap('inferno')
Nx = 40
Ny = 40
T0 = np.zeros( (Nx, Ny) )
T0[0, :] = 10
T0 = T0.reshape(Nx * Ny)


Nt = 100
t = np.linspace(0, 100, Nt)

alpha = 5

print("solving the PDE... this may take several seconds...")
sol = odeint(heatODE, T0, t, args=(alpha, Nx, Ny))
print("done")

sol = sol.reshape((Nt, Nx, Ny))

plt.figure()
plt.pcolor(sol[0])
plt.title("Temperature Distribution at t=0")
plt.colorbar()
plt.show()

plt.pcolor(sol[1])
plt.title("Temperature Distribution at t=1")
plt.colorbar()
plt.show()

plt.pcolor(sol[-1])
plt.title(f"Temperature Distribution at t={t[-1]}")
plt.colorbar()
plt.show()
