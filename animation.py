import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing, os, sys, cProfile, pstats, io, traceback, time, random
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from matplotlib.ticker import LogLocator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from functools import lru_cache
from joblib import Parallel, delayed
import warnings
from numba import jit, cuda 
from collections import defaultdict

filepath = '/Users/AbSir/Desktop/cj22/CJ-code/fitpack/nucl/phi.wbarel_AV18'
gamma_list = []
y_list = []
f2_list = []
f0_list = []
with open(filepath, 'r') as file:
    for line in file:
        try:
            columns = [float(col) for col in line.split()]
            if len(columns) == 8:
                gamma_list.append(columns[0])
                y_list.append(columns[1])
                f2_list.append(columns[6])
                f0_list.append(columns[2])
        except:
            continue
gamma_array = np.array(gamma_list)
y_array = np.array(y_list)
f2_array = np.array(f2_list)
f0_array = np.array(f0_list)

def phi_int2d_fixed_gamma(y_D, gamma_fixed):
    # Find the indices of the gamma values that are equal to the fixed gamma value
    mask = gamma_array == gamma_fixed
    y_filtered = np.array(y_array[mask])
    f2_filtered = np.array(f2_array[mask])
    f0_filtered = np.array(f0_array[mask])
    # y_filtered = 0.5 * y_filtered

    # Perform 1D interpolation along the y axis for the fixed gamma value
    interpolator = interp1d(y_filtered, f0_filtered, kind = 'linear', fill_value = 'extrapolate')

    # Interpolate at the given y value
    result = interpolator(y_D)

    return result

y_D = np.linspace(0, 1, 1000)

y_d_values = []
# all_y_d = []
hisaab = defaultdict(list)
x_noter = []

def f2_function(x, y_D):
    k = x / y_D
    return 4 * k**0.7 * (1 - k)**3.5 * (1 + 4 * k) + 0.15 * k**(-0.4) * (1 - k)**9 * (1 + 16 * k)

def f2nxn_function(x):
    x_N = 2 * x # Convert from x_D to x_N
    return 4 * x_N**0.7 * (1 - x_N)**3.5 * (1 + 4 * x_N) + 0.15 * x_N**(-0.4) * (1 - x_N)**9 * (1 + 16 * x_N)

def phi_int2d_fixed_gamma(y_D, gamma_fixed):
    # Find the indices of the gamma values that are equal to the fixed gamma value
    mask = gamma_array == gamma_fixed
    y_filtered = np.array(y_array[mask])
    f0_filtered = np.array(f0_array[mask])

    # Perform 1D interpolation along the y axis for the fixed gamma value
    interpolator = interp1d(y_filtered, f0_filtered, kind = 'linear', fill_value = 'extrapolate')
    # Interpolate at the given y value
    result = interpolator(y_D)

    return result

# Define the product of f2 and f0 for the integrand
def integrand(y_D, x):
    global y_d_values
    y_d_values.append(y_D)
    f0 = phi_int2d_fixed_gamma(y_D, gamma_fixed)
    f2 = f2_function(x, y_D)
    return f2 * f0

# Perform the integration
def integrate_f2_f0(x):
    global y_d_values
    # all_y_d.append(len(y_d_values))
    hisaab[x].append(len(y_d_values))
    hisaab[x].append(y_d_values)
    y_d_values = [] # Clear the list before each integration to avoid mixing values from different calls
    f2dxd, error = quad(integrand, x, 1, args = x, epsabs = 1e-6, epsrel = 1e-6, limit = maxsub)
    # f2dxn = f2dxd / 2 # Getting structure function in terms of nucleon momentum fraction
    f2dxn = f2dxd
    return f2dxn

gamma_fixed = 1.0
xno = 500
maxsub = 21
x = np.linspace(0, 1, xno) # This is x_D
simple_f2_list = [f2nxn_function(x_val) for x_val in x]
integral_result_list = [integrate_f2_f0(x_val) for x_val in x]
ratio = np.array(integral_result_list) / np.array(simple_f2_list)

gamma_fixed = 1.0
y_plot = np.linspace(0, 1, 500)
x_values = np.linspace(0.01, 1.0, 50)

fig, ax = plt.subplots()
line, = ax.plot([], [], color='black')
title = ax.set_title('')
ax.set_xlabel(r'$y_D$')
ax.set_ylabel(r'$S \cdot F_{2N}$')
ax.grid(alpha=0.2)
ax.set_xlim(0, 1)
# ax.set_ylim(0, 25)

def product(y_D, x):
    f0 = phi_int2d_fixed_gamma(y_D, gamma_fixed)
    f2 = f2_function(x, y_D)
    return f2 * f0

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x_fixed = x_values[frame]
    prodl = [product(y, x_fixed) for y in y_plot]
    prodl = np.array(prodl)
    gunakar = prodl[~np.isnan(prodl)]
    y_filtered = y_plot[~np.isnan(prodl)]
    line.set_data(y_filtered, gunakar)
    peak_value = np.max(gunakar)
    ax.set_ylim(0, peak_value)  # Adjust y-axis limits
    ax.set_title(f'x = {x_fixed:.2f}, peak at y = {y_filtered[np.argmax(gunakar)]:.3f}')
    return line,

# ani = FuncAnimation(fig, update, frames = len(x_values), init_func = init, blit = True)
ani = FuncAnimation(fig, update, frames = len(x_values), blit = False, interval = 700)

# Save the animation
ani.save(f'animation_{time.time():.0f}.mp4', writer = 'ffmpeg', fps = 2)  # fps controls the frame rate

plt.show()
