# helpers functions to be used in phonon thermal properties code
import numpy as np
import yaml
from dataclasses import dataclass
import h5py # https://docs.h5py.org/en/stable/quick.html to read hdf5 files
import matplotlib.pyplot as plt

@dataclass
class Data: # define data object to hold data 
    qpoints: np.ndarray = None
    qpoint_weight: np.ndarray = None
    frequency: np.ndarray = None
    gamma: np.ndarray = None
    group_velocity: np.ndarray = None
    gv_by_gv: np.ndarray = None
    lifetime: np.ndarray = None
    temperature: np.ndarray = None
    heat_capacity: np.ndarray = None
    reciprocal_lattice: np.ndarray = None
    kappa_unit_conversion: int = None
    # ref values
    ref_heat_capacity: np.ndarray = None
    ref_free_energy: np.ndarray =None
    ref_entropy: np.ndarray =None
    ref_mode_kappa: np.ndarray = None
    ref_thermal_conductivity: np.ndarray = None

# Extracts data from .txt and .dat files and writes to class Data()
def get_data(inputfile):
    data = Data()
    headers = None
    with open(inputfile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                toks = line.strip().split()
                if len(toks) > 1 and toks[1] == 'x':
                    headers = toks[1:]  # header corresponds to numeric columns
                    break
    if headers is None:
        exit
    else:
        freq_indices = [i for i, h in enumerate(headers) if h.startswith('w_') or h.startswith('Mode_')]
        group_vel_indices = [i for i, h in enumerate(headers) if h.startswith('V_g')]
        lifetime_indices = [i for i, h in enumerate(headers) if h.startswith('t_')]
    
    try: # bulk load numeric data (fast, C-backed)
        numeric = np.loadtxt(inputfile, comments='#', dtype=np.float64)
        data.qpoints = numeric[:, 0:3]
        data.frequency = numeric[:, freq_indices] if freq_indices else np.empty((numeric.shape[0], 0)) * 1e12  # UNITS OF HZ, NOT ANGULAR
        data.group_velocity = numeric[:, group_vel_indices] if group_vel_indices else np.empty((numeric.shape[0], 0))
        data.lifetime = numeric[:, lifetime_indices] if lifetime_indices else np.empty((numeric.shape[0], 0))
        return data
    except Exception as e: # fallback to python
        print(f'Exception:{e}')
        qpoints = []
        frequencies = []
        velocity = []
        lifetime = []
        with open(inputfile, 'r') as f:
            for line in f:
                if line.startswith('#') or len(line.split()) < 3:
                    continue
                parts = line.split()
                qpoints.append((float(parts[0]), float(parts[1]), float(parts[2])))
                if freq_indices:
                    frequencies.append([float(parts[i]) for i in freq_indices])
                if group_vel_indices:
                    velocity.append([float(parts[i]) for i in group_vel_indices])
                if lifetime_indices:
                    lifetime.append([float(parts[i]) for i in lifetime_indices])
        data.qpoints = np.array(qpoints, dtype=np.float64)
        data.frequency = np.array(frequencies, dtype=np.float64) * 1e12 if frequencies else np.empty((0, 0))
        data.group_velocity = np.array(velocity, dtype=np.float64) if velocity else np.empty((0, 0, 3))
        data.lifetime = np.array(lifetime, dtype=np.float64) if lifetime else np.empty((0, 0))
        return data
    
# Extracts data from .yaml files and writes to class Data()
def get_data_yaml(yamlfile):
    data = Data()
    with open(yamlfile, 'r') as f:
        yamldata = yaml.safe_load(f)
    
    try:
        data.qpoints=np.array([phonon['q-position'] for phonon in yamldata['phonon']])
        data.qpoint_weight=np.array([phonon['weight'] for phonon in yamldata['phonon']] if 'weight' in yamldata['phonon'][0] else np.empty((0, 0)))
        data.frequency=np.array([[band['frequency'] for band in phonon['band']] for phonon in yamldata['phonon']])* 1e12 if 'frequency' in yamldata['phonon'][0]['band'][0] else np.empty((0, 0)) 
        data.group_velocity=np.array([[band['group_velocity'] for band in phonon['band']] for phonon in yamldata['phonon']]) if 'group_velocity' in yamldata['phonon'][0]['band'][0] else np.empty((0, 0, 3))
        data.reciprocal_lattice=np.array([vec for vec in yamldata['reciprocal_lattice']]) if 'reciprocal_lattice' in yamldata else np.empty((3, 3))
    except Exception as e:
        print(f'Exception:{e}')
    return data

# Extracts data from .hdf5 files and writes to class Data()
def get_data_hdf5(inputfile,write=False,savefile=None,plot=False):
    data = Data()
    try:    
        hdf5data = h5py.File(inputfile, 'r')
        keys=hdf5data.keys() # read the hdf5 keys like a python dictionary
        data.frequency=np.array(hdf5data['frequency'][:] if 'frequency' in keys else None, dtype=np.float128) * 1e12 # Hz ordinal frequency NOT ANGULAR
        data.gamma=np.array([g for g in hdf5data['gamma']] if 'gamma' in keys else None, dtype=np.float128) * 1e12 # Hz ordinal frequency NOT ANGULAR
        data.qpoints=np.array(hdf5data['qpoint'][()] if 'qpoint' in keys else None, dtype=np.float64)
        data.qpoint_weight=np.array(hdf5data['weight'][:] if 'weight' in keys else None, dtype=np.float64)
        data.group_velocity=np.array([v for v in hdf5data['group_velocity']] if 'group_velocity' in keys else None, dtype=np.float64) * 1e12 # Hz*Angstrom ordinal frequency
        data.gv_by_gv=np.array([v for v in hdf5data['gv_by_gv']] if 'gv_by_gv' in keys else None, dtype=np.float64) * 1e12*1e12 # Hz^2 * Angstrom^2 ordinal frequency
        data.heat_capacity=np.array([cv for cv in hdf5data['heat_capacity']] if 'heat_capacity' in keys else None, dtype=np.float64) # eV/K NOT J
        data.temperature=np.array(hdf5data['temperature'][:] if 'temperature' in keys else None, dtype=np.float64) # Kelvin
        data.kappa_unit_conversion=np.array(hdf5data['kappa_unit_conversion'][()] if 'kappa_unit_conversion' in keys else None, dtype=np.float64)
        data.ref_mode_kappa=np.array([s for s in hdf5data['mode_kappa']] if 'mode_kappa' in keys else None, dtype=np.float64)
        data.ref_thermal_conductivity=np.array([k for k in hdf5data['kappa']] if 'kappa' in keys else None, dtype=np.float64)
        return data
    except Exception as e:
        print(f'Failed to extract data from {inputfile}:\n{e}')

# Converts fractional q-points to Cartesian BZ coords before 3D plotting
# Based on Phonopy formulation documentation (https://phonopy.github.io/phonopy/formulation.html) reciprocal lattice should already have applied 2pi 
def convert_qpts_to_bz(points,reciprocal_lattice=np.array([[  -0.18513561,   0.18513561,   0.18513559 ],[   0.18513562,  -0.18513562,   0.18513561 ],[   0.18513558,   0.18513559,  -0.18513560 ]])*2*np.pi):
    qpts=np.array([(q[0]*reciprocal_lattice[0,:]+q[1]*reciprocal_lattice[1,:]+q[2]*reciprocal_lattice[2,:]) for q in points])
    return qpts
       
# Shared marker/color styling for plotting
def plot_settings():
    import numpy as np
    import matplotlib.pyplot as plt
    # general settings for plotting
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    symbols = ['D','s','o','v','^','<','x','.','+','1','2','3']
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red','tab:cyan','tab:brown', 
              'teal', 'tab:olive','tab:pink', 'coral', 'tab:gray','gold',
               'tab:purple', 'navy', 'darkgreen', 'maroon', 'steelblue', 'seagreen',
              'indigo', 'crimson', 'forestgreen', 'tomato','slateblue', 'chocolate']
    return symbols, colors

# Plots qpoint mesh in 3D
def plot_qpoints(qpointsdata=None,qpointsfile='QPOINTS', show=True, save=False, savefile=None,size='medium'):
    if qpointsdata is not None:
        qpoints=qpointsdata
    else:
        try:
            qpoints = get_data(qpointsfile).qpoints
        except Exception as e:
            print(f'Exception:{e}')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5

    sz=[8,4] if size == 'small' else [6,8]
    fig = plt.figure(figsize=(sz))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(qpoints[:, 0], qpoints[:, 1], qpoints[:, 2],c='#2E86AB', s=18, alpha=0.4, label='Q-points', zorder=6)
    
    ax.set_xlabel('$q_x$ (a.u.)', fontsize=14, labelpad=10)
    ax.set_ylabel('$q_y$ (a.u.)', fontsize=14, labelpad=10)
    ax.set_zlabel('$q_z$ (a.u.)', fontsize=14, labelpad=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False;ax.yaxis.pane.fill = False;ax.zaxis.pane.fill = False
    plt.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {savefile}")

# Shell function to plot a thermal property, allows for setting of most parameters
def plot_thermal_property(tvals,tpvals,key,show=True,save=False,savefile=None,xlabel=f'T (K)',ylabel=f'$C_v$ (J/K/mol)',title='Thermal Property Comparison Plot',xlim=[0,2000],ylim=None,sz=[6,6],style=['-',':','--','-.', ':','--','-.', ':']):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    
    fig = plt.figure(figsize=(sz))
    
    for i in range(len(key)):
        plt.plot(tvals[key[i]],tpvals[key[i]],label=key[i],ls=style[i%8],lw=2.5)
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=14, labelpad=10)
    plt.ylabel(ylabel, fontsize=14, labelpad=10)
    plt.title(title, fontsize=16, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if show:
        plt.show()
    if save:
        fig.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {savefile}")

# Plot the phonon bands and density of states from the default Phonopy .dat and .yaml output
def plot_phonon_bands_and_dos(dosfile='./data-bin/Si_total_dos.dat',bandsfile='./data-bin/Si_band.yaml',fermi=0,show=True,save=False,savefile=None,xlabel=f'T (K)',ylabel=f'Frequency (Hz)',title='Phonon Band Structure and Density of States',xlim=[None,None],ylim=[None,None],sz=[8,6]):
    # import data from files
    try: # DOS
        numeric = np.loadtxt(dosfile, comments='#', dtype=np.float64)
        dos_frequency=numeric[:,0]
        dos=numeric[:,1]
    except Exception as e:
        print(f'Failed to extract total_dos.dat from {dosfile}:\n{e}')
        exit
    try: # BANDS
        with open(bandsfile, 'r') as f:
            bandsdata = yaml.safe_load(f)
        distance=np.array([phonon['distance'] for phonon in bandsdata['phonon']])
        bands=np.array([[band['frequency'] for band in phonon['band']] for phonon in bandsdata['phonon']])
        labels= np.array([labels for labels in bandsdata['labels']])
        segment_nqpoint=np.array([n for n in bandsdata['segment_nqpoint']])
    except Exception as e:
        print(f'Failed to extract band.yaml from {bandsfile}:\n{e}')
        exit
        
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    
    fig, axs = plt.subplots(nrows=1, ncols=2,sharey=True,figsize=sz,gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(title,fontsize=14)

    plt.subplot(121) # BANDS
    kpath=np.zeros(len(segment_nqpoint)+1)
    for k in range(0,len(segment_nqpoint)):
        kpath[k+1]=distance[np.sum(segment_nqpoint[:k])]
    # plt.vlines(kpath[1:-1],ylim[0],ylim[1],color='b',linestyle='dashed')
    for b in range(len(bands[0,:])):
        plt.plot(distance, bands[:,b]-(fermi), color='black',lw=2)
    plt.xticks(kpath,np.append(labels[:,0],labels[-1,-1]))
    plt.xlim([kpath[0],kpath[-1]])
    plt.ylim(ylim)
    plt.xlabel('K-Point Path', labelpad = 10)
    plt.ylabel('Frequency (THz)', labelpad = 3)
    
    plt.subplot(122) # DOS
    plt.plot(dos/np.max(dos), dos_frequency-(fermi), color='black',lw=2)
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

# Extracts reference Phonopy thermal property data from a .yaml or .dat file and writes to class Data()
def get_thermal_properties_refdata(inputfile: str):
    data = Data()
    try:
        if ".yaml" in inputfile:
            with open(inputfile,'r') as f:
                yamldata = yaml.safe_load(f)
            data.temperature = np.array([temp['temperature'] for temp in yamldata['thermal_properties']])
            data.ref_free_energy = np.array([temp['free_energy'] for temp in yamldata['thermal_properties']])
            data.ref_entropy = np.array([temp['entropy'] for temp in yamldata['thermal_properties']])
            data.ref_heat_capacity = np.array([temp['heat_capacity'] for temp in yamldata['thermal_properties']])
        else:
            numeric = np.loadtxt(inputfile, comments='#', dtype=np.float64)
            data.temperature = numeric[:, 0]
            data.ref_free_energy = numeric[:, 1]
            data.ref_entropy = numeric[:, 2]
            data.ref_heat_capacity = numeric[:, 3]
    except Exception as e:
        print(f'failed to get reference values from {inputfile}\n{e}')
    return data