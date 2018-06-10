"""
This file takes the NIST data for He I and lists the
resonance lines and their atomic properties. In cases
when the oscillator strengths and/or damping constant
are unknown, the values are fitting with a low order
function.

The atomic data is then saved into a LaTeX table.
"""

import numpy as np

from matplotlib import pyplot as plt
import copy
import mpfit


def func_function(x, p):
    y = p[0] + p[1]*np.exp(p[2] * x**p[3])
    return y


def myfunct(p, fjac=None, x=None, y=None, err=None):
    model = func_function(x, p)
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    return [status, (y-model)/y]

wavarr, fvlarr, gamarr = np.loadtxt("HI_lines.dat", unpack=True)

xplt = np.arange(fvlarr.size)+1

ww = np.where(xplt <= 9)
coeff = np.polyfit(np.log10(xplt[ww]), np.log10(fvlarr[ww]), 3)
fmodel = np.polyval(coeff, np.log10(xplt))

coeff = np.polyfit(np.log10(xplt[ww]), np.log10(gamarr[ww]), 2)
gmodel = np.polyval(coeff, np.log10(xplt))

if False:
    # Fit the atomic data to extrapolate
    # Set some reasonable starting conditions
    p0 = [0.0, 1.0, 1.0, 1.0]

    # Set some constraints you would like to impose
    param_base = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.]}

    # Make a copy of this 'base' for all of our parameters
    param_info = []
    for i in range(len(p0)):
        param_info.append(copy.deepcopy(param_base))
        param_info[i]['value'] = p0[i]

    # Now tell the fitting program what we called our variables
    ww = np.where(fvlarr != -1.0)
    fa = {'x': xplt[ww], 'y': np.log10(fvlarr[ww]), 'err': np.ones(fvlarr.size)[ww]}
    m = mpfit.mpfit(myfunct, p0, parinfo=param_info, functkw=fa, quiet=False)
    if m.status <= 0:
        print('error message = ', m.errmsg)
    fmodel = func_function(xplt, m.params)

    # Now tell the fitting program what we called our variables
    ww = np.where(gamarr != -1.0)
    fa = {'x': xplt[ww], 'y': np.log10(gamarr[ww]), 'err': np.ones(gamarr.size)[ww]}
    m = mpfit.mpfit(myfunct, p0, parinfo=param_info, functkw=fa, quiet=False)
    if m.status <= 0:
        print('error message = ', m.errmsg)
    gmodel = func_function(xplt, m.params)

plt.subplot(221)
plt.plot(np.log10(xplt), np.log10(fvlarr), 'bx')
plt.plot(np.log10(xplt), fmodel, 'r-')
plt.subplot(222)
plt.plot(xplt, np.log10(gamarr), 'bx')
plt.plot(xplt, gmodel, 'r-')
# Residuls
plt.subplot(223)
plt.plot(xplt, 100.0*(fvlarr - 10.0**fmodel)/fvlarr, 'bx')
plt.plot(xplt, np.zeros(xplt.size), 'r-')
plt.subplot(224)
plt.plot(xplt, 100.0*(gamarr - 10.0**gmodel)/gamarr, 'bx')
plt.plot(xplt, np.zeros(xplt.size), 'r-')
plt.show()
