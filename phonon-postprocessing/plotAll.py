import numpy as np
import seaborn as sns
import pandas as pd
import yaml as yaml
import time
import sys

import matplotlib.pylab as plt
from matplotlib import rcParams
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib.lines import Line2D

# PLOT FORMAT 
rcParams.update({'figure.autolayout': True})
sns.set_style("whitegrid", rc={"axes.edgecolor": "k", "axes.linewidth":2.})
sns.set_style("ticks", {"xtick.major.size":8,"ytick.major.size":8})

sns.set_context("notebook",rc={"grid.linewidth": 0.1,
                            "font.family":"Serif", "axes.labelsize":10.,"xtick.labelsize":10.,
                            "ytick.labelsize":10., "legend.fontsize":9.,"title.labelsize":12.,'xtick.direction': 'in','ytick.direction': 'in'}) # https://seaborn.pydata.org/tutorial/aesthetics.html
rcParams['font.family'] = 'serif'

colors = sns.color_palette("colorblind", 8) #https://seaborn.pydata.org/tutorial/color_palettes.html
extracolors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

def main():
    args = sys.argv[1:]                      # everything after the script name
    flags = {a for a in args if a.startswith("-")}
    tags = [a for a in args if not a.startswith("-")]
    help_statement="Usage: python3 <dos_file> <bands_file> <output_file>\nOptional: -help -electron <KLABELS_file> -phonon"
    if "-help" in flags or "-h" in flags:
        print(help_statement)
        exit(2)
    if len(tags) < 3: # check the provided arguments
        print("Error: incorrect arguments")
        print(help_statement)
        exit(1)
    try:
        plot_size = (6,4)
        dos_file = tags[0]
        bands_file = tags[1]
        output_file = tags[2]
        do_electron = "-electron" in flags or "-e" in flags
        do_phonon = "-phonon" in flags or "-p" in flags or ".yaml" in bands_file
    except Exception as e:
        print(f"Failed to assign variables because of \n{e}")
    try:
        if do_electron:
            if "-electron" in args: 
                klabels_file = args[args.index('-electron')+1]
            else:
                klabels_file = args[args.index('-e')+1]
            plot_electron_dft(dos_file=dos_file,bands_file=bands_file,klabels_file=klabels_file,output_file=output_file,size=plot_size)
        if do_phonon:
            plot_phonon(dos_file=dos_file,bands_file=bands_file,output_file=output_file,size=plot_size)
    except Exception as e:
        print(f"Failed to plot because of \n{e}")

# Plot DFT Electronic DOS and Bands for a DFT system
def plot_electron_dft(dos_file,bands_file,klabels_file,output_file,size=(6,4)):
    # Read in data from file
    dos = np.array(np.loadtxt(dos_file))
    bands = np.array(np.loadtxt(bands_file))
    labels,kpath = np.loadtxt(klabels_file,dtype=str,skiprows=1,comments='*',unpack=True)
    labels[labels == 'GAMMA'] = '\u0393' # make gamma pretty
    kpath = np.array(kpath,dtype=float) # convert to numbers
    fig, axs = plt.subplots(nrows=1, ncols=2,sharey=True,figsize=size,gridspec_kw={'width_ratios': [3, 1]})
    # fig.suptitle('Electronic Results',fontsize=12)
    plt.subplot(121)
    pathLIM=[kpath[0],kpath[-1]];YLIM=np.array([np.min(bands[:,1:]),np.max(bands[:,1:])])*1.1
    if np.max(dos[:,1])/np.mean(dos[:,1][dos[:,1]>0]) > 3:
        DOSLIM=[-0.01,np.mean(dos[:,1][dos[:,1]>0])*3]
    else:
        DOSLIM=[-0.01,np.max(dos[:,1])*1.01]

    plt.vlines(kpath,YLIM[0],YLIM[1],color=colors[7],linestyle='dashed',label='Fermi')
    if bands.shape[1] == 2:
        plt.plot(bands[:,0], bands[:,1], color='k', label=None)
    else:
        for band in bands[:,1:].T:
            plt.plot(bands[:,0], band, color='k')
    #plt.text(0.19,-0.1,"Band Gap={0}".format(bandgap),backgroundcolor='white')
    plt.xlim(pathLIM);plt.ylim(YLIM)
    plt.xticks(kpath,labels)
    plt.hlines(0,pathLIM[0],pathLIM[1],color=colors[6],linestyle='dashed')
    plt.xlabel('K-Point Path')
    plt.ylabel('Energy (eV)')
    #plt.title('Band Structure', fontsize=14)

    plt.subplot(122) 
    plt.plot(dos[:,1], dos[:,0], color='k',label='DFT')
    plt.hlines(0,DOSLIM[0],DOSLIM[1],color=colors[6],linestyle='dashed',label='Fermi')
    plt.xticks([DOSLIM[0],DOSLIM[1]],[' ',' '])
    plt.xlabel('DOS (a.u.)')
    plt.xlim(DOSLIM)
    plt.ylim(YLIM)
    #plt.title('Density of States', fontsize=14)
    plt.legend()

    plt.show() 
    plt.savefig(f'{output_file.rsplit(".", 1)[0]}-DFTelectronic.jpeg',  bbox_inches='tight', pad_inches = 0.1, dpi=600)

# Plot Phonon Bands and DOS
def plot_phonon(dos_file='total_dos.dat',bands_file='band.yaml',fermi=0,output_file=None,title='Phonon Band Structure and DOS',size=[6,4]):
    # import data from files
    try: # DOS
        numeric = np.loadtxt(dos_file, comments='#', dtype=np.float64)
        dos_frequency=numeric[:,0]
        dos=numeric[:,1]
    except Exception as e:
        print(f'Failed to extract total_dos.dat from provided input files, try again. {e}')
        exit

    try: # BANDS
        with open(bands_file, 'r') as f:
            bandsdata = yaml.safe_load(f)
        distance=np.array([phonon['distance'] for phonon in bandsdata['phonon']])
        bands=np.array([[band['frequency'] for band in phonon['band']] for phonon in bandsdata['phonon']])
        segment_nqpoint=np.array([n for n in bandsdata['segment_nqpoint']])
        labels= np.array([labels for labels in bandsdata['labels']] if 'labels' in bandsdata else np.empty((segment_nqpoint.shape[0],2)))

    except Exception as e:
        print(f'Failed to extract band.yaml from provided input files, try again. {e}')
        exit
    fig, axs = plt.subplots(nrows=1, ncols=2,sharey=True,figsize=size,gridspec_kw={'width_ratios': [3, 1]})
    # fig.suptitle(title,fontsize=12)
    plt.subplot(121) # BANDS
    YLIM=[0,np.max(dos_frequency)*1.01]
    kpath=np.zeros(len(segment_nqpoint)+1)
    for k in range(0,len(segment_nqpoint)):
        kpath[k+1]=distance[np.sum(segment_nqpoint[:k])]
    for b in range(len(bands[0,:])):
        plt.plot(distance, bands[:,b]-(fermi), color=colors[0],lw=2)
    # band labels and styling
    plt.xticks(kpath,np.append(labels[:,0],labels[-1,-1]))
    plt.xlim([kpath[0],kpath[-1]])
    plt.ylim(YLIM)
    plt.xlabel('K-Point Path', labelpad = 10)
    plt.ylabel('Frequency (THz)', labelpad = 3)
    #plt.legend()
    
    # DOS
    plt.subplot(122)
    plt.plot(dos/np.max(dos), dos_frequency-(fermi), color=colors[0],lw=2)
    plt.xticks([0,1.2],['',''])
    plt.xlabel('DOS (a.u.)',  labelpad = 10)
    plt.xlim(0,1.2)
    plt.ylim(YLIM)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{output_file.rsplit(".", 1)[0]}-phonon.jpeg',  bbox_inches='tight', pad_inches = 0.1, dpi=600)

if __name__ == "__main__":
    # test the time of operation
    start = time.time()
    main()
    end = time.time()
    print(f"Done after {(end - start):.6f} seconds")