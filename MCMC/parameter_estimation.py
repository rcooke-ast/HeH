#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import emcee
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import sys

cd_nam = ['H', 'H+',
          'He', 'He+', 'He+2',
          'C', 'C+', 'C+2', 'C+3',
          'O', 'O+', 'O+2', 'O+3', 'O+4', 'O+5', 'O+6',
          'Mg', 'Mg+', 'Mg+2',
          'Si', 'Si+', 'Si+2', 'Si+3', 'Si+4']

# Set the column density data (the zeroth element must be N(H I))
# Fakedata
# ('UVBslope = ', 0.0)
# ('[C/H] = ', -1.8)
# ('Hescl = ', 0.93000000000000005)
# ('nH = ', -3.2000000000000002)
# ('log NHI = ', 17.399999999999999)
# ('H = ', 17.400038652321157)
# ('He = ', 16.451907652754723)
# ('C+ = ', 13.860730790524665)
# ('C+3 = ', 15.244282069998507)
# ('Si+ = ', 12.565506621894549)
# ('Si+3 = ', 13.806593035830895)
yn =          ['H',  'He', 'C+', 'C+3', 'Si+', 'Si+3']
y  = np.array([17.400038652321157, 16.451907652754723, 13.860730790524665,  15.244282069998507,  12.565506621894549,  13.806593035830895])
ye = 1.0e-4 * np.array([0.1,  0.1,  0.1,   0.1,   0.1,   0.1])

# Set the real data
# yn =          ['H',  'He', 'C+', 'C+3', 'Si+', 'Si+3']
# y  = np.array([17.0, 16.4, 13.8,  15.2,  12.5,  13.8])
# ye = np.array([ 0.1,  0.1,  0.1,   0.1,   0.1,   0.1])

yi = []
Ncol = y.size
x = np.zeros(Ncol)  # A fudge - this array is not used

# Load the model
filename = "data/radiation_z1p724_uvbslope_data.npy"
allmodel = np.load(filename)
# Extract the columns that are of interest to us
Ndims = 5  # Number of parameters to estimate
Hcol = allmodel[:, Ndims] + allmodel[:, Ndims+1]
Hecol = allmodel[:, Ndims+2] + allmodel[:, Ndims+3] + allmodel[:, Ndims+4]
model_slp = allmodel[:, 0]
model_met = allmodel[:, 1]
model_yp = allmodel[:, 2] * np.median((Hecol/Hcol)/allmodel[:, 2])
model_hden = allmodel[:, 3]
model_NHI = allmodel[:, 4]

mn_sic, mx_sic = -1.0, 1.0   # [Si/C] ratio
mn_slp, mx_slp = np.min(model_slp), np.max(model_slp)
mn_met, mx_met = np.min(model_met), np.max(model_met)
mn_yp, mx_yp = np.min(model_yp), np.max(model_yp)
mn_hden, mx_hden = np.min(model_hden), np.max(model_hden)
mn_NHI, mx_NHI = np.min(model_NHI), np.max(model_NHI)

unq_slp = np.unique(model_slp)
unq_met = np.unique(model_met)
unq_yp = np.unique(model_yp)
unq_hden = np.unique(model_hden)
unq_NHI = np.unique(model_NHI)

# diff = unq_yp[1:] - unq_yp[:-1]
# print(np.where(diff < np.median(diff)))

value_cden = []
for i in range(len(yn)):
    for j in range(len(cd_nam)):
        if yn[i] == cd_nam[j]:
            yi.append(Ndims+j)
            value_cden.append(np.log10(allmodel[:, Ndims+j]))
            break

if len(yi) != len(yn):
    print("Error with column density names")
    sys.exit()

if True:
    # This routine is faster
    print("Preparing model grids")
    XS, XM, XY, XH, XN = np.meshgrid(unq_slp, unq_met, unq_yp, unq_hden, unq_NHI, indexing='ij')
    print("Interpolating model grids")
    pts = [unq_slp, unq_met, unq_yp, unq_hden, unq_NHI]
    model_cden = []
    for i in range(Ncol):
        print("{0:d}/{1:d}".format(i+1, Ncol))
        vals = value_cden[i].reshape((unq_slp.size, unq_met.size, unq_yp.size, unq_hden.size, unq_NHI.size))
        model_cden.append(RegularGridInterpolator(pts, vals, method='linear', bounds_error=False, fill_value=np.inf))
        # model_cden.append(LinearNDInterpolator(pts, value_cden[i], fill_value=np.inf))
    print("Complete")
else:
    # This routine is slower
    print("Interpolating model grids")
    model_cden = []
    pts = np.array((model_slp, model_met, model_yp, model_hden, model_NHI )).T
    for i in range(Ncol):
        # s, m, y, n, h = theta
        # ['H', 'He', 'C+', 'C+3', 'Si+', 'Si+3']
        model_cden.append(LinearNDInterpolator(pts, value_cden[i], fill_value=np.inf))
    print("Complete")


def get_model(theta):
    model = np.zeros(Ncol)
    for ii in range(Ncol):
        model[ii] = model_cden[ii]([[theta[1:]]])
        if 'Si' in yn[ii]:
            model[ii] += theta[0]
    return model


# Define the probability function as likelihood * prior.
def lnprior(theta):
    r, s, m, y, n, h = theta
    if mn_met <= m <= mx_met and \
       mn_yp <= y <= mx_yp and \
       mn_hden <= n <= mx_hden and \
       mn_NHI <= h <= mx_NHI and \
       mn_sic <= r <= mx_sic and \
       mn_slp <= s <= mx_slp:
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr):
    model = get_model(theta)
    inv_sigma2 = 1.0/yerr**2
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# Find the maximum likelihood value by brute force.
chi = np.zeros((model_yp.size, Ncol))
for i in range(Ncol):
    chi[:, i] = (y[i]-value_cden[i])/ye[i]
chi2 = chi**2
chisq = np.sum(chi2, axis=1)
bst = np.argsort(chisq)
printbst = 1
for i in range(printbst):
    print("""------------------------\n
        Maximum likelihood result {5:d}/{6:d} {7:.4f}:\n
        [M/H]  = {0}\n
        yp     = {1}\n
        n(H)   = {2}\n
        N(H I) = {3}\n
        slope  = {4}\n""".format(model_met[bst[i]], model_yp[bst[i]], model_hden[bst[i]], model_NHI[bst[i]],
                                 model_slp[bst[i]], i+1, printbst, chisq[bst[i]]))
modvals = [0.0, model_slp[bst[i]], model_met[bst[i]], model_yp[bst[i]], model_hden[bst[i]], model_NHI[bst[i]]]

# Set up the sampler.
ndim, nwalkers = 6, 100
# maxlike = np.array([model_ms[bst], model_ex[bst], model_mx[bst]])
# minv_ms, maxv_ms = 19.0, 22.0
# minv_ex, maxv_ex = 1.0, 5.0
# minv_mx, maxv_mx = 0.0, 0.0001
# minv_ms, maxv_ms = np.min(model_ms[bst[:printbst]]), np.max(model_ms[bst[:printbst]])
# minv_ex, maxv_ex = np.min(model_ex[bst[:printbst]]), np.max(model_ex[bst[:printbst]])
# minv_mx, maxv_mx = np.min(model_mx[bst[:printbst]]), np.max(model_mx[bst[:printbst]])
pos = [np.array([np.random.uniform(mn_sic, mx_sic),
                 np.random.uniform(mn_slp, mx_slp),
                 np.random.uniform(mn_met, mx_met),
                 np.random.uniform(mn_yp, mx_yp),
                 np.random.uniform(mn_hden, mx_hden),
                 np.random.normal(y[0], ye[0])]) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, ye), threads=ndim)

# Clear and run the production chain.
print("Running MCMC...")
nmbr = 100
a=time.time()
for i, result in enumerate(sampler.run_mcmc(pos, nmbr, rstate0=np.random.get_state())):
    if True:#(i+1) % 100 == 0:
        print("{0:5.1%}".format(float(i) / nmbr))
print("Done.")
print((time.time()-a)/60.0, 'mins')

print("Saving samples")
np.save("samples{0:d}.npy".format(nmbr), sampler.chain)
