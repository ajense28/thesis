import numpy as np
import time
import sys
import yaml
import h5py # https://docs.h5py.org/en/stable/quick.html to read hdf5 files
import matplotlib.pyplot as plt
import importlib
import helpers
importlib.reload(helpers)

# CONSTANTS
NA=6.02214076e23 # units:[atom/mol] # avogadro's number
h=6.62607015e-34 # [J/Hz]  
kB=1.380649e-23 # [J/K]
eV2J=1.602177e-19 # 1 eV / 1.602177e-19 J

def main():
    args = sys.argv[1:]                      # everything after the script name
    flags = {a for a in args if a.startswith("-")}
    tags = [a for a in args if not a.startswith("-")]
    if "-help" in flags or "-h" in flags:
        print("Usage: python3 phonopy_thermal_properties.py <input_file> <output_file>")
        print("Optional: -h -write -plot -err")
        exit(2)
    if len(tags) < 2: # check the provided arguments
        print("Error: incorrect arguments")
        print("Usage: python3 phonopy_thermal_properties.py <input_file> <output_file> -write -plot -err")
        print("Optional: -h -write -plot -err")
        exit(1)

    input_file = tags[0]
    output_file = tags[1]
    do_write = "-write" in flags or "-w" in flags
    do_plot = "-plot" in flags or "-p" in flags
    do_err  = "-err" in flags or "-e" in flags

    plot_size = (6,4)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    try:
        data = helpers.get_data_hdf5(inputfile=input_file)
        # weights=data.qpoint_weight
        gv = data.group_velocity
        freq=data.frequency
        temperatures=data.temperature
        gamma = data.gamma
        kappa_unit_conversion=data.kappa_unit_conversion
        # reference values
        ref_cv = np.array(data.heat_capacity).sum(axis=1).sum(axis=1)
        ref_kappa = data.ref_thermal_conductivity
    except Exception as e:
        print(f'Failed to load data because of {e}')       
    try:
        cv,kappa = compute_kappa(freq=freq,temp=temperatures,gv=gv,gamma=gamma[:],kappa_unit_conversion=kappa_unit_conversion)
        print(r'Total calculated $\kappa_x$ at 270K = ' +f'{kappa[27,0]}')
        print(r'Total reference $\kappa_x$ at 270K  = ' +f'{ref_kappa[27,0]}')
    except Exception as e:
        print(f'Failed to compute because of {e}')        
    try:
        if do_write:
            # Open output file for writing
            with open(output_file, 'w') as out_f:
                out_f.write(r"Temperature\t$\kappa_x$\t$\kappa_y$\t$\kappa_z$\t$\kappa_y$$_z$\t$\kappa_x$$_z$\t$\kappa_x$$_y$\tC$_v$\n")
                for t,temp in enumerate(temperatures):
                    out_f.write(f"{temp}\t{'\t'.join(map(str, kappa[t]))}\t{cv[t]}")
        if do_err:
            err_kappa=np.array([compute_relative_error(kappa[5:,i],ref_kappa[5:,i]) for i in range(3)])
            print(f'Max error = {np.nanmax(err_kappa)}')
            plot_relative_error(err_kappa.T,xlabel='Temperatures',components={0: "xx", 1: "yy", 2: "zz"},xval=temperatures[5:],output_file=output_file)
        if do_plot:  
            plot_heat_capacity(temperatures,zip(cv,ref_cv),labels=[r"C$_v$",r"Ref C$_v$"],xlim=[0,1000],output_file=output_file)
            plot_tc(kappa[5:,0],ref_kappa[5:,0],temperatures[5:],ylabel="W/mK",size=plot_size,output_file=output_file)
    except Exception as e:
        print(f'Failed to write data because of {e}')    

# COMPUTE THERMAL CONDUCTIVITY 
def compute_kappa(freq,temp,gv,gamma,kappa_unit_conversion):
    # note that temp values must match the reference temperatures for accuracy with gamma -> alter results
    if isinstance(temp,int): temp = np.append(np.nan,temp)
    N_temp=temp.shape[0]
    N_qpoints=freq.shape[0]    
    N_modes=freq.shape[1]

    #  cv units eV/molK -----------------------------------
    cv=np.zeros([N_temp,N_qpoints,N_modes]) # initialize C_v
    for i,T in enumerate(temp) if not isinstance(temp,int) else (0,temp):
        if T > 0:
            for s,f in enumerate(np.split(freq,indices_or_sections=len(freq[0,:]),axis=1)): # do each mode separately (should repeat num_mode times)
                x=(h*f)/(kB*T)
                cv[i,:,s] = (kB*np.power(x,2)*(np.exp(x))/np.power(np.exp(x)-1,2)).flatten() # returned in units of J/K/point
    cv=cv/eV2J
    # gv_by_gv units Hz^2 * A^2 -----------------------------------
    gv_by_gv=np.zeros((N_qpoints,N_modes,6)) # qpt
    for q in range(N_qpoints): 
        for s in range(N_modes):
            gv_by_gv[q,s,:]=gv_by_gv[q,s,:]+ [gv[q,s,0]*gv[q,s,0],gv[q,s,1]*gv[q,s,1],gv[q,s,2]*gv[q,s,2],gv[q,s,1]*gv[q,s,2],gv[q,s,0]*gv[q,s,2],gv[q,s,0]*gv[q,s,1]]
    # kappa units W/mK -------------------------------
    mode_kappa = np.empty([N_temp,N_qpoints,N_modes,6])
    tau = np.where(np.where(gamma > 0, gamma, -1) > 0, 1.0 / (2 * np.where(gamma > 0, gamma, -1)), 0)
    for x in range(len(gv_by_gv[0,0,:])):
        mode_kappa[:,:,:,x] =  tau * cv * gv_by_gv[:,:,x] * kappa_unit_conversion * 1e-12 # unit correction for Hz
    cv_temp = cv.sum(axis=1).sum(axis=1)
    kappa = mode_kappa.sum(axis=1).sum(axis=1)/N_qpoints # normalize by mesh to compare to ref_thermal_conductivity
    return cv_temp,kappa

def plot_tc(calc_kappa,ref_kappa,temperatures=None,title=None,size=None,xlabel="Temperature (K)",ylabel=r"K W/mK",output_file=None):
    [symbols, colors] = helpers.plot_settings()
    fig, ax = plt.subplots(figsize=size or (8, 4))
    x = 10*np.arange(calc_kappa.shape[0]) if temperatures is None else temperatures
    ax.plot(x, calc_kappa[:], label=r'$\kappa_x$ (DFT)', color=colors[0 % len(colors)], linewidth=3, marker=symbols[0 % len(symbols)],markersize=0)
    ax.plot(x, ref_kappa[:], label=r'$\kappa_x$ Ref', color=colors[7 % len(colors)], linewidth=3, marker=symbols[1 % len(symbols)],markersize=1,linestyle=':')
    # plt.xscale('linear')
    plt.yscale('linear')
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.7, edgecolor="black",fontsize=9)
    fig.tight_layout()
    fig.show()
    if output_file is not None:
        plt.savefig(f'{output_file.rsplit(".", 1)[0]}-kappa.jpeg',  bbox_inches='tight', pad_inches = 0.1, dpi=600)
def plot_heat_capacity(xvals,yvals,labels,xlabel=f'Temperature (K)',ylabel=f'$C_v$ eV/K',title=None,xlim=[0,2000],ylim=None,size=[8,4],style=['-',':','--','-.', ':','--','-.', ':'],output_file=None):
    [symbols,colors] = helpers.plot_settings()

    fig = plt.figure(figsize=(size))
    for i,data in enumerate(zip(*yvals)):
        plt.plot(xvals,data,color=colors[i^3%24],label=labels[i],ls=style[i%8],lw=2.5)
    
    # Labels and styling
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    plt.title(title, pad=20)
    plt.legend()
    # Grid styling
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    if output_file is not None:
        plt.savefig(f'{output_file.rsplit(".", 1)[0]}-cv.jpeg',  bbox_inches='tight', pad_inches = 0.1, dpi=600)

 # RELATIVE ERROR FORMULATION FUNCTION  ------------
def compute_relative_error(calcdata,refdata):
    numerator = abs(refdata - calcdata)
    denom = np.where(refdata != 0, abs(refdata), abs(calcdata))
    error = np.where(denom != 0, numerator / denom, 0.0)
    return error
# RELATIVE ERROR VISUALIZATION FUNCTION ------------
def plot_relative_error(error, xlabel="X-Axis", title="Relative Error", components=None,size=None,xval=None,output_file=None):
    [symbols, colors] = helpers.plot_settings()
    x = np.arange(error.shape[0]) if xval is None else xval
    is_modes = False
    fig, ax = plt.subplots(figsize=size or (8, 4))
    if np.nanmax(error) > 10:
        plt.yscale('log')
    else:
        plt.yscale('linear')
    if len(np.shape(error)) > 1: # obviously don't iterate if there is only one set of data
        VOIGT_LABELS = {0: "xx", 1: "yy", 2: "zz", 3: "yz", 4: "xz", 5: "xy"}
        is_modes = error.shape[1] > 6 # check if data is modes or voigt xyz
        if components is None and not is_modes:
            index = list(VOIGT_LABELS.keys())
            label = list(VOIGT_LABELS.values())
        elif components is None and is_modes:
            index = np.arange(error.shape[1])
            label = [f'mode {idx+1}' for idx in index]
        elif not isinstance(components, int):
            index = components
            label = [f'mode {idx+1}' for idx in components] if is_modes else [VOIGT_LABELS[idx] for idx in components]
        else:
            index = components
            label = f'mode {components}' if is_modes else VOIGT_LABELS[components]
        for i, l, idx in zip(index,label,index) if not isinstance(index,int) else zip(index,label,index):
            ax.plot(x, error[:, idx], label=l, color=colors[i % len(colors)], linewidth=0.1, marker=symbols[i % len(symbols)])
        ax.plot(x, np.sum(error,axis=1), label='total error', color='k', linewidth=1, marker='|')
    else:
        ax.plot(x, error, label='total error', color='k', linewidth=1, marker='|')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Error")
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.7, edgecolor="black",fontsize=9,ncols = 1 if not is_modes else 2, loc='best' if not is_modes else (1,0))
    fig.tight_layout()
    plt.show()
    if output_file is not None:
        plt.savefig(f'{output_file.rsplit(".", 1)[0]}-error.jpeg',  bbox_inches='tight', pad_inches = 0.1, dpi=600)

if __name__ == "__main__":
    # test the time of operation
    start = time.time()
    main()
    end = time.time()
    print(f"Done after {(end - start):.6f} seconds")

# OTHER FUNCTIONS ------------------------
def compute_debye_heat_capacity(freq,temp):
    NA=6.02214076e23 # units:[atom/mol] # avogadro's number
    N=len(freq[0,:])/3 # num phonon modes = num 'modes' / 3 polarizations = 2
    h=6.62607015e-34 # [J/Hz]  
    kB=1.380649e-23 # [J/K]
    # Wikipedia says 645K Debye temperature for silicon crystal
    TD=495 
    cv=np.zeros(len(temp))
    # assumes temperatures are sorted low to high in numerical order for simplicity
    for i,T in enumerate(temp):
        if T>0:
            xD=TD/T # unitless Debye limit
            for f in np.split(freq,indices_or_sections=len(freq[:,0]),axis=0): # integrate each qpoint separately
                x=(h*f)/(kB*T) # this num modes*1 matrix of x values for a particular qpt and T 
                xval=x[np.logical_and(0<=x,x<=xD)] # all x values within the integral limit for a particular mode
                cv[i]=cv[i] + 9*N*kB*np.power(xD,-3) * np.trapz(np.power(xval,4)*np.exp(xval)/np.power(np.exp(xval)-1,2),x=xval)
                # CHECK THE CUBED VALUE BEFORE TRAPEZOIDAL
                # check literature for debye temp for dft phonon calc ***
    return cv*NA/len(freq[:,0]) # units of J/molK
def dulong_petit_limit(freq,temp): 
    kB=1.380649e-23 # [J/K]
    NA=6.02214076e23 # units:[atom/mol] # avogadro's number
    N=1 #len(freq[0,:])/3 # number of primitive cells / number of modes
    dulong_petit_result=3*N*kB # theoretical upper limit for cv
    #print(dulong_petit_result) # = 49.886775708919444 [2*J/K/mol]
    # returns the dulong petit result as a horizontal line (T,cv) to compare to other results
    cv=dulong_petit_result*np.ones(len(temp))
    return cv*NA # units of J/molK
# band structure and density of states plot
import matplotlib.pyplot as plt
import numpy as np
import yaml
def plot_bands_and_dos(dosfile='./data-bin/Si_total_dos.dat',bandsfile='./data-bin/Si_band.yaml',fermi=0,show=True,save=False,savefile=None,xlabel=f'T (K)',ylabel=f'Frequency (Hz)',title='Band Structure and Density of States',xlim=[None,None],ylim=[None,None],sz=[8,6]):
    # import data from files
    try: # DOS
        numeric = np.loadtxt(dosfile, comments='#', dtype=np.float64)
        dos_frequency=numeric[:,0]
        dos=numeric[:,1]
    except Exception as e:
        print(f'Failed to extract total_dos.dat from provided input files, try again. {e}')
        exit

    try: # BANDS
        with open(bandsfile, 'r') as f:
            bandsdata = yaml.safe_load(f)
        distance=np.array([phonon['distance'] for phonon in bandsdata['phonon']])
        bands=np.array([[band['frequency'] for band in phonon['band']] for phonon in bandsdata['phonon']])
        labels= np.array([labels for labels in bandsdata['labels']])
        segment_nqpoint=np.array([n for n in bandsdata['segment_nqpoint']])
    except Exception as e:
        print(f'Failed to extract band.yaml from provided input files, try again. {e}')
        exit
    [symbols, colors] = helpers.plot_settings()
    fig, axs = plt.subplots(nrows=1, ncols=2,sharey=True,figsize=sz,gridspec_kw={'width_ratios': [3, 1]})
    
    fig.suptitle(title,fontsize=14)

    plt.subplot(121) # BANDS
    
    kpath=np.zeros(len(segment_nqpoint)+1)
    for k in range(0,len(segment_nqpoint)):
        kpath[k+1]=distance[np.sum(segment_nqpoint[:k])]
    plt.hlines(0,kpath[0],kpath[-1],color=(0.984313725490196, 0.6862745098039216, 0.8941176470588236),linestyle='dashed',label='Fermi')
    for b in range(len(bands[0,:])):
        plt.plot(distance, bands[:,b]-(fermi), color='black',lw=2)
    # band labels and styling
    plt.xticks(kpath,np.append(labels[:,0],labels[-1,-1]))#kpoints=['L','\u0393','X','W','K','\u0393']
    plt.xlim([kpath[0],kpath[-1]])
    plt.ylim(ylim)
    plt.xlabel('K-Point Path', labelpad = 10)
    plt.ylabel('Frequency (THz)', labelpad = 3)
    #plt.legend()
    
    # DOS
    plt.subplot(122)
    plt.plot(dos/np.max(dos), dos_frequency-(fermi), color='black',lw=2)
    plt.hlines(0,0,1.5,color=(0.984313725490196, 0.6862745098039216, 0.8941176470588236),linestyle='dashed',label='Fermi')
    plt.xticks([0,1.2],['',''])
    plt.xlabel('DOS',  labelpad = 10)
    plt.xlim(0,1.2)
    plt.ylim(ylim)

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {savefile}")

    plt.show()
def compute_entropy(freq,temperatures):
    n=len(freq[:,0]) # num mesh pts
    S=np.zeros(len(temperatures)) # initialize
    i=len(temperatures[temperatures<=0]) # index variable because the loop iterates over T values
    for T in temperatures[temperatures>0]:
        S[i] = 1/(2*T)*np.sum(h*freq/np.tanh(h*freq/(2*kB*T))) - kB*np.sum(np.log(2*np.sinh(h*freq/(2*kB*T))))
        i=i+1
    return S*NA/n # units of J/K/mol