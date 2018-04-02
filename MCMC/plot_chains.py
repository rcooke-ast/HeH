import numpy as np
import corner
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import plotting_routines as pr

burnin = 0
ndim = 6
chains = np.load('samples100.npy')
samples = chains[:, burnin:, :].reshape((-1, ndim))
prenams = ['[C/Si]', 'UVBslope', '[C/H]', 'yp', 'nH', 'log NHI']

# Plot the timeline
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
contourf_kwargs["colors"] = ((1, 1, 1), (0.6, 0.6, 0.6), (0.3, 0.3, 0.3))
fig = corner.corner(samples, bins=[200, 50, 50, 50, 50, 50], levels=levels, plot_datapoints=False, fill_contours=True,
                    plot_density=False, contour_kwargs=contour_kwargs, contourf_kwargs=contourf_kwargs, smooth=1,
                    labels=[r"[C/Si]", r"$\alpha_{\rm UV}$", r"[C/H]", r"$y_{\rm P}$", r"$n_{\rm H}~({\rm cm}^{-3})$", r"log~$N$(H\,\textsc{i})~$({\rm cm}^{-2})$"])
axes = np.array(fig.axes).reshape((6, 6))
# [axes[i,0].set_xlim(18.0,23.0) for i in range(3)]
# [axes[i,1].set_xlim(0.3,10.0) for i in range(3)]
# [axes[i,2].set_xlim(0.0,0.25) for i in range(3)]
# axes[1,0].set_ylim(0.3,10.0)
# [axes[2,i].set_ylim(0.0,0.25) for i in range(2)]
for aa in range(ndim):
    [l.set_rotation(0) for l in axes[aa, 0].get_yticklabels()]
for aa in range(ndim):
    [l.set_rotation(0) for l in axes[ndim-1, aa].get_xticklabels()]
[axes[i,i].yaxis.set_ticks_position('none') for i in range(ndim)]
[axes[ndim-1, i].xaxis.set_label_coords(0.5, -0.18) for i in range(ndim)]
[axes[i, 0].yaxis.set_label_coords(-0.25, 0.5) for i in range(1, ndim)]
# axes[2,0].set_xticks([19.0, 20.0, 21.0, 22.0])
# axes[2,2].set_xticks([0.0, 0.1, 0.2])
# axes[2,0].set_yticks([0.0, 0.1, 0.2])
# axes[2,1].set_yticks([0.0, 0.1, 0.2])
# pr.replot_ticks(axes[1, 0])
# pr.replot_ticks(axes[2, 0])
# pr.replot_ticks(axes[2, 1])
fig.savefig("parameter_estimation.pdf")

[([tk.set_visible(True) for tk in ax.get_yticklabels()], [tk.set_visible(True) for tk in ax.get_yticklabels()]) for ax in axes.flatten()]

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
