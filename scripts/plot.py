import os
from helper import *
import matplotlib.pyplot as plt
from atlasify import atlasify
import numpy as np

def plotHists(dset, variables, labels, pxranges, processes,
              colors, xlabels, scale, nbins, ylabel, weights, 
              outputplotsdir = "plots", 
              alpha=0.6, normalize=True, ATLASlabel = []):

    plt.rc('axes', labelsize=12)
    plt.rc('legend', fontsize=12)
    
    # ==>> Loop over variables
    for ivar, var in enumerate(variables):
        
        # ==>> Loop over processes
        for iproc, proc in enumerate(processes):

            plt.hist(dset[proc][var]*scale[ivar], nbins[ivar],
                     facecolor=colors[iproc], alpha=alpha,
                     range=pxranges[ivar], density=normalize,
                     weights=weights[iproc],
                     label=labels[iproc])

        plt.xlabel(xlabels[ivar])
        if normalize: plt.ylabel("Normalized "+ylabel)
        else: plt.ylabel(ylabel)

        if len(ATLASlabel) == 2:
            atlasify(ATLASlabel[0], ATLASlabel[1])

        outputdir = getOutputDir()
        outfilename = var+".png"
        if not normalize: outfiliename = var+"_noNorm.png"
        mkdir_p(os.path.join(outputdir, outputplotsdir))
        outfilepath = os.path.join(outputdir, outputplotsdir, outfilename)
        print("Saving figure: {}".format(outfilepath))
        plt.savefig(outfilepath, dpi=360)

        plt.clf()


def poisson_limits(N, kind, confidence=0.6827):
    alpha = 1 - confidence
    upper = np.zeros(len(N))
    lower = np.zeros(len(N))
    if kind == 'sqrt':
        err = np.sqrt(N)
        lower = N - err
        upper = N + err
    else:
        raise ValueError('Unknown errorbar kind: {}'.format(kind))
    # clip lower bars
    lower[N==0] = 0
    return N - lower, upper - N

def histpoints(x, bins, xerr=None, yerr='sqrt', normed=False, **kwargs):
    """
    Plot a histogram as a series of data points.
    Compute and draw the histogram of *x* using individual (x,y) points
    for the bin contents.
    By default, vertical poisson error bars are calculated using the
    gamma distribution.
    Horizontal error bars are omitted by default.
    These can be enabled using the *xerr* argument.
    Use ``xerr='binwidth'`` to draw horizontal error bars that indicate
    the width of each histogram bin.
    Parameters
    ---------
    x : (n,) array or sequence of (n,) arrays
        Input values. This takes either a single array or a sequence of
        arrays, which are not required to be of the same length.
    """
    import matplotlib.pyplot as plt

    h, bins = np.histogram(x, bins=bins)
    width = bins[1] - bins[0]
    bincenter = (bins[:-1] + bins[1:]) / 2
    area = sum(h * width)

    if isinstance(yerr, str):
        yerr = poisson_limits(h, yerr)

    if xerr == 'binwidth':
        xerr = width / 2

    if normed:
        h = h / area
        yerr = yerr / area
        area = 1.

    return bincenter, h, (yerr[0], yerr[1]), area
