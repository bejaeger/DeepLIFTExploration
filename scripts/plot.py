import os
from helper import *
import matplotlib.pyplot as plt
from atlasify import atlasify

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

        try:
            outputdir = os.environ["OUTPUTDIR"]
        except KeyError:
            print("ERROR: Environment variable 'OUTPUTDIR' " \
                  "not found! Maybe you forgot to execute " \
                  "'source setup.sh'?")
            os._exit(0)
    
        outfilename = var+".png"
        if not normalize: outfiliename = var+"_noNorm.png"
        mkdir_p(os.path.join(outputdir, outputplotsdir))
        outfilepath = os.path.join(outputdir, outputplotsdir, outfilename)
        print("Saving figure: {}".format(outfilepath))
        plt.savefig(outfilepath, dpi=360)

        plt.clf()

