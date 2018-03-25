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
    y = p[0] + p[1]*(x**p[2])
    return y


def myfunct(p, fjac=None, x=None, y=None, err=None):
    model = func_function(x, p)
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    return [status, (y-model)/err]

lines = open("HeI_lines.dat", 'r').readlines()

term = None
wavarr = np.array([])
fvlarr = np.array([])
gamarr = np.array([])
for i in range(len(lines)):
    # Check if it's a resonant line
    linspl = lines[i].split("|")
    try:
        tst = float(linspl[5].split("-")[0].strip())
        wave = float(linspl[4].strip())
        lower = linspl[2].split("-")[0].strip()
        upper = linspl[2].split("-")[1].strip()
    except:
        continue
    try:
        term = linspl[1].split("-")[1].strip()
    except:
        pass
    # Check it's resonant
    if tst != 0.0 or upper != "1P*":
        continue
    # Get the oscillator strength if it exists
    try:
        fval = float(linspl[8].strip())
    except:
        fval = -1.0
    # Get the Einstein coefficient if it exists
    try:
        gamma = float(linspl[7].strip())
    except:
        gamma = -1.0
    # Calculate the transition strength
    if gamma != -1.0:
        # Find all levels below this one, and sum their Aki
        for j in range(len(lines)):
            if i == j:
                continue
            linspl = lines[j].split("|")
            # See if the level is a match
            try:
                glterm = linspl[1].split("-")[0].strip()
                gterm = linspl[1].split("-")[1].strip()
                glower = linspl[2].split("-")[0].strip()
                gupper = linspl[2].split("-")[1].strip()
            except:
                continue
            if glterm == "1s2":
                continue
            if glower == lower and gupper == upper and gterm == term:
                pass
            else:
                continue
            # Get the Einstein coefficient if it exists
            try:
                Aki = float(linspl[7].strip())
            except:
                Aki = -1.0
            if Aki != -1.0:
                gamma += Aki
    wavarr = np.append(wavarr, wave)
    fvlarr = np.append(fvlarr, fval)
    gamarr = np.append(gamarr, gamma)
    # print "He I   {0:.4f} {1:f}  {2:E}".format(wave, fval, gamma)

xplt = np.arange(wavarr.size)+1

ww = np.where(fvlarr != -1.0)
coeff = np.polyfit(np.log10(xplt[ww]), np.log10(fvlarr[ww]), 3)
fmodel = np.polyval(coeff, np.log10(xplt))

ww = np.where(gamarr != -1.0)
coeff = np.polyfit(np.log10(xplt[ww]), np.log10(gamarr[ww]), 2)
gmodel = np.polyval(coeff, np.log10(xplt))

if False:
    # Fit the atomic data to extrapolate
    # Set some reasonable starting conditions
    p0 = [0.0, 1.0, 1.0]

    # Set some constraints you would like to impose
    param_base = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.]}

    # Make a copy of this 'base' for all of our parameters
    param_info = []
    for i in range(len(p0)):
        param_info.append(copy.deepcopy(param_base))
        param_info[i]['value'] = p0[i]

    # Now tell the fitting program what we called our variables
    ww = np.where(fvlarr != -1.0)
    fa = {'x': xplt[ww], 'y': fvlarr[ww], 'err': np.ones(wavarr.size)[ww]}

    #######################################
    #  PERFORM THE FIT AND PRINT RESULTS
    #######################################

    m = mpfit.mpfit(myfunct, p0, parinfo=param_info, functkw=fa, quiet=False)
    if m.status <= 0:
        print('error message = ', m.errmsg)
    print("param: ", m.params)
    print("error: ", m.perror)

    # Generate the best-fitting model
    fmodel = func_function(xplt, m.params)

plt.subplot(231)
plt.plot(xplt, np.log10(wavarr))
plt.subplot(232)
plt.plot(np.log10(xplt), np.log10(fvlarr), 'bx')
plt.plot(np.log10(xplt), fmodel, 'r-')
plt.subplot(233)
plt.plot(xplt, np.log10(gamarr), 'bx')
plt.plot(xplt, gmodel, 'r-')
# Residuls
plt.subplot(235)
plt.plot(xplt, np.log10(fvlarr) - fmodel, 'bx')
plt.plot(xplt, np.zeros(xplt.size), 'r-')
plt.subplot(236)
plt.plot(xplt, np.log10(gamarr) - gmodel, 'bx')
plt.plot(xplt, np.zeros(xplt.size), 'r-')
plt.show()

textab = open("table.tex", 'w')
tablines = []
print((fvlarr-10.0**fmodel)/10.0**fmodel)
print((gamarr-10.0**gmodel)/10.0**gmodel)
print("-------------------")
for i in range(wavarr.size):
    if fvlarr[i] == -1.0:
        fv = 10.0**fmodel[i]
        fstr = "^{a}"
    else:
        fv = fvlarr[i]
        fstr = ""
    if gamarr[i] == -1.0:
        gv = 10.0**gmodel[i]
        gstr = "^{a}"
    else:
        gv = gamarr[i]
        gstr = ""
    if np.log10(fv) > -1.0: fvs = "{0:.4f}".format(fv)
    elif np.log10(fv) > -2.0: fvs = "{0:.5f}".format(fv)
    elif np.log10(fv) > -3.0: fvs = "{0:.6f}".format(fv)
    elif np.log10(fv) > -4.0: fvs = "{0:.7f}".format(fv)
    tablines.append("        He\\,\\textsc{i}" + "& ${0:.4f}$ & ${1:s}{3:s}$ & {2:4.3E}${4:s}$\\\\\n".format(wavarr[i], fvs, gv, fstr, gstr))
    print("He I   {0:.4f} {1:f}{3:s}  {2:E}{4:s}".format(wavarr[i], fv, gv, fstr, gstr))
print("* = extrapolated value")

textab.write("\\begin{table}\n")
textab.write("    \\centering\n")
textab.write("    \\caption{This is an example table. Captions appear above each table.}\n")
textab.write("    \\label{tab:atomic}\n")
textab.write("    \\begin{tabular}{lccc}\n")
textab.write("        \\hline\n")
textab.write("        Ion & wavelength & $f$ & $\\Gamma$\\\\\n")
textab.write("            & (\\AA)     &     & $({\\rm s}^{-1})$\\\\\n")
textab.write("        \\hline\n")
for i in tablines:
    textab.write(i)
textab.write("        \\hline\n")
textab.write("    \\end{tabular}\n\n")
textab.write("$^{\\rm a}${No atomic data exist for these transitions. Instead, these values were extrapolated based on a polynomial fit to the known measures from lower order lines (see text).}\\\\\n")
textab.write("\\end{table}\n")
textab.close()
