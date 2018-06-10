import numpy as np
import corner
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import plotting_routines as pr

burnin = 1000
ndim = 6
onevar = -1
#chains = np.load('UVBfgal_samples105000.npy')
#chains = np.load('UVBredshift_samples2000.npy')
#chains = np.load('UVBredshift_samples105000.npy')
#chains = np.load('UVBredshift_samples15000_inflate10.npy')
#chains = np.load('UVBredshift_samples15000_inflate1_zabsPrior.npy')
chains = np.load('FIXED1Ryd_samples3000.npy')
samples = chains[:, burnin:, :].reshape((-1, ndim))

#prenams = ['[C/Si]', 'UVBslope', '[C/H]', 'yp', 'nH', 'log NHI']
prenams = ['[C/Si]', 'redshift', '[C/H]', 'yp', 'nH', 'log NHI']

# Plot the timeline
if True:
	pl.clf()
	fig, axes = pl.subplots(ndim, 1, sharex=True, figsize=(8, 9))
	for i in range(ndim):
		axes[i].plot(chains[:, :, i].T, color="k", alpha=0.4)
		axes[i].yaxis.set_major_locator(MaxNLocator(5))
		#axes[i].axhline(modvals[i], color="#888888", lw=2)
		axes[i].axvline(burnin, color="r", lw=2)
		axes[i].set_ylabel(prenams[i])
	fig.tight_layout(h_pad=0.0)
	fig.savefig("time_evolution.png")

# Make the triangle plot.
levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)
contour_kwargs, contourf_kwargs = dict({}), dict({})
contour_kwargs["linewidths"] = [1.0, 1.0]
contourf_kwargs["colors"] = ((1, 1, 1), (0.502, 0.651, 0.808), (0.055, 0.302, 0.5727))
hist_kwargs = dict({})
hist_kwargs["color"] = contourf_kwargs["colors"][-1]
fig = corner.corner(samples, bins=[50, 50, 50, 50, 50, 50], levels=levels, plot_datapoints=False, fill_contours=True, smooth=1,
                    plot_density=False, contour_kwargs=contour_kwargs, contourf_kwargs=contourf_kwargs, hist_kwargs=hist_kwargs,
                    labels=[r"[C/Si]", r"$z_{\rm eff}$", r"[C/H]", r"$y_{\rm P}$", r"$n_{\rm H}~({\rm cm}^{-3})$", r"log~$N$(H\,\textsc{i})/${\rm cm}^{-2}$"])
axes = np.array(fig.axes).reshape((6, 6))

# Set the rotation of the labels, and turn off y ticks for diagonals
for aa in range(ndim):
    [l.set_rotation(0) for l in axes[aa, 0].get_yticklabels()]
    axes[aa, 0].set_ylabel(axes[aa, 0].get_ylabel(), labelpad=100)
for aa in range(ndim):
    [l.set_rotation(0) for l in axes[ndim-1, aa].get_xticklabels()]
[axes[i,i].yaxis.set_ticks_position('none') for i in range(ndim)]

# Set the limits and tick locations (the fast and last element represent limits, the inner values represent tick locations)
if False:
	lims_CSi = [0.1, 0.2, 0.3, 0.4, 0.45]
	#lims_zeff = [0.0, 0.6, 1.2, 1.8, 2.4, 2.7]
	lims_zeff = [-4.0, -2.0, -1.5, -1.0, -0.5, 0.0]
	lims_CH = [-1.85, -1.8, -1.7, -1.6, -1.53]
	lims_yp = [0.054, 0.06, 0.08, 0.10, 0.12, 0.125]
	lims_nH = [-2.9, -2.7, -2.4, -2.1, -2.0]
	lims_NHI = [16.77, 16.8, 16.9, 17.0, 17.03]
	lims = [lims_CSi, lims_zeff, lims_CH, lims_yp, lims_nH, lims_NHI]
	#for xx in range(ndim):
	#	[axes[i,xx].set_xlim(lims[xx][0], lims[xx][-1]) for i in range(xx)]
	#	[axes[i,xx].set_xticks(lims[xx][1:-1]) for i in range(xx)]
	for xx in range(ndim):
		[axes[xx,i].set_xlim(lims[i][0], lims[i][-1]) for i in range(xx+1)]
		[axes[xx,i].set_xticks(lims[i][1:-1]) for i in range(xx+1)]
	for xx in range(1,ndim):
		[axes[xx,i].set_ylim(lims[xx][0], lims[xx][-1]) for i in range(xx)]
		[axes[xx,i].set_yticks(lims[xx][1:-1]) for i in range(xx)]
		[axes[xx,i].tick_params(bottom=True, top=True, left=True, right=True) for i in range(xx)]

# Draw the canvas
fig.canvas.draw()

[axes[i, 0].yaxis.set_label_coords(-0.3, 0.5) for i in range(1, ndim)]
[axes[-1, i].xaxis.set_label_coords(0.5, -0.2) for i in range(ndim)]
pr.plot_pm(axes[-1, 0], xy="x", zero=False)
for aa in range(1, ndim):
	for bb in range(aa):
		pr.replot_ticks(axes[aa, bb])

fig.savefig("parameter_estimation_FIXED1Ryd.pdf")

if onevar >= 0:
	# Just plot a single parameter as a function of all the other parameters
	fig2=plt.figure(figsize=(3.504, 6.0))
	
	gs = gridspec.GridSpec(3, 2)
	#gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
	gs.update(left=0.16, right=0.98, top=0.98, bottom = 0.07, wspace=0.025, hspace=0.05)

	# Add each subplot
	ax1 = fig2.add_subplot(gs[0])

	for i in range(1, ndim):
		axes[i, onevar].remove()
		axes[i, onevar].figure = fig2
		fig2.axes.append(axes[i, onevar])
		fig2.add_axes(axes[i, onevar])

		dummy = fig2.add_subplot(gs[i])
		axes[i, onevar].set_position(dummy.get_position())
		dummy.remove()
	plt.close(fig)
	fig2.savefig("parameter_estimation_onevar.pdf")


#[([tk.set_visible(True) for tk in ax.get_yticklabels()], [tk.set_visible(True) for tk in ax.get_yticklabels()]) for ax in axes.flatten()]

# Compute the quantiles.
mcmc_CSi, mcmc_UVB, mcmc_CH, mcmc_yp, mcmc_nH, mcmc_NHI = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                              zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print("""MCMC result:
    [C/Si] = {0[0]} +{0[1]} -{0[2]})
    Slope  = {1[0]} +{1[1]} -{1[2]})
    [C/H]  = {2[0]} +{2[1]} -{2[2]})
    y_P    = {3[0]} +{3[1]} -{3[2]})
    n_H    = {4[0]} +{4[1]} -{4[2]})
    N(H I) = {5[0]} +{5[1]} -{5[2]})
""".format(mcmc_CSi, mcmc_UVB, mcmc_CH, mcmc_yp, mcmc_nH, mcmc_NHI))
