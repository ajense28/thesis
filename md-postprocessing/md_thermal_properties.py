# script to extract data from MD LAMMPS dump file and compute heat flux and thermal conductivity
# to run this script:   
#   python md_thermal_properties.py <input_file> <output_file> <ncore>
# interactive job :     
#   salloc -A <account> -t 00:30:00 --ntasks=8
# see first x lines of file:  
#   head -x <file>
from typing import List
import numpy as np
import sys
import time # to check time
from concurrent.futures import ProcessPoolExecutor # to parallelize over multiple cores

def main(): # This 'main' executes the script and controls everything else
    if len(sys.argv) < 2: # check the provided arguments
        print("Error: incorrect arguments")
        print("Usage: python md_thermal_properties.py <input_file>")
        print("Optional: python md_thermal_properties.py <input_file> <output_file> <ncore>")
        exit(1)
    # Declare Global variables
    input_file=sys.argv[1] #"../Si-LAMMPS/dump.atom_data"
    ncore=int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit else 1 # number of cores 
    print(f"Input file: {input_file}")
    output_file=sys.argv[2] if len(sys.argv) > 2 else "OUTPUT2.csv"
    print(f"Output file: {output_file}")

    # Open output file for writing
    with open(output_file, 'w') as out_f:
        out_f.write("Timestep,Volume,J_x,J_y,J_z,Jconv_x,Jconv_y,Jconv_z,Jcond_x,Jcond_y,Jcond_z\n")
        all_flux = []
        # Get each block frame and process it
        for block_frames in frame_block_window(input_file, ncore):
            flux_result = process_block_frame(block_frames, ncore)
            all_flux += flux_result
            out_f.write("\n".join([flux.pretty_output() for flux in flux_result]) + "\n")
            out_f.flush()
            #print(f"Completed timestep(s) " + ",".join(str(flux.get_timestep()) for flux in flux_result))
            #exit() # only do one block

class Block_Frame: # define block frame object to hold LAMMPS dump data, 1 block = 1 timestep
    def __init__(self):
        self._timestep=None
        self._natom=None
        self._bounds=None
        self._atom_headers=None
        self._data_dict=None
    
    def set_timestep(self, timestep: int):
        self._timestep = timestep
    def set_natom(self, natom: int):
        self._natom = natom
    def set_bounds(self, bounds: np.array):
        self._bounds = bounds
    def set_atom_headers(self, atom_headers: list):
        self._atom_headers = atom_headers
    def set_data_dict(self, data_dict: dict):
        self._data_dict = data_dict

    def get_timestep(self):
        return self._timestep
    def get_natom(self):
        return self._natom
    def get_bounds(self):
        return self._bounds
    def get_atom_headers(self):
        return self._atom_headers
    def get_data_dict(self):
        return self._data_dict

def process_data(file: str): # extract block of data from LAMMPS dump file
    with open(file, 'r') as f:
        block_frame = Block_Frame()
        bounds = []
        data = []
        try:
            for line in f:
                if line.startswith('ITEM: TIMESTEP'):
                    block_frame.set_timestep(int(next(f).strip()))
                elif line.startswith('ITEM: NUMBER OF ATOMS'):
                    block_frame.set_natom(int(next(f).strip()))
                elif line.startswith('ITEM: BOX BOUNDS'):
                    for _ in range(3):
                        bounds.append(list(map(float, next(f).strip().split())))
                    block_frame.set_bounds(np.array(bounds))
                elif line.startswith('ITEM: ATOMS'):
                    headers = line.strip().split()[2:] 
                    block_frame.set_atom_headers(headers)
                    for _ in range(block_frame.get_natom()):
                        data.append(list(map(float, next(f).strip().split()))) 
                    data = np.array(data).T
                    data_dict = {header: data[i] for i, header in enumerate(headers)}
                    block_frame.set_data_dict(data_dict)
                    data = []
                    yield block_frame
                    block_frame = Block_Frame()
                    bounds = []
        except Exception as e:
            print(f"Error processing file: {e}")

def frame_block_window(path: str, size: int):
    block_frames = []
    for block_frame in process_data(path):
        block_frames.append(block_frame)
        if len(block_frames) == size:
            yield block_frames
            block_frames = []
    if block_frames:
        yield block_frames

def process_block_frame(block_frames: List[Block_Frame], ncore: int):
    with ProcessPoolExecutor(max_workers=ncore) as executor:
        # Pass headers and data as a single tuple
        fluxes = list(executor.map(compute_flux, block_frames))
        #print(f"Completed processing " + ",".join(str(flux.get_timestep()) for flux in fluxes) +" steps")
    return fluxes

class J_Out: # define heat flux output object
    def __init__(self, t, v, jtot, jconv, jcond):
        self._timestep=t
        self._volume=v
        self._flux_tot=jtot
        self._flux_conv=jconv
        self._flux_cond=jcond

    def pretty_output(self):    
        # Convert arrays to flat strings
        ftot_str = '\t'.join(map(str, self._flux_tot))
        fconv_str = '\t'.join(map(str, self._flux_conv))
        fcond_str = '\t'.join(map(str, self._flux_cond))
        return f"{self._timestep}\t{self._volume}\t{ftot_str}\t{fconv_str}\t{fcond_str}"
        
    def get_timestep(self):
        return self._timestep
    
    def get_volume(self):
        return self._volume
    
    def get_flux_tot(self):
        return self._flux_tot
    
def compute_flux(block_frame: Block_Frame): # compute flux, output volume Jtot=[Jx Jy Jz] J_conv=[Jx_conv Jy_conv Jz_conv] J_cond=[Jx_cond Jy_cond Jz_cond]
    # UNITS CONVERSION 1 bar = 0.000006241509 eV/Angstrom^3
    # LAMMPS method from https://docs.lammps.org/compute_heat_flux.html 
    data_dict = block_frame.get_data_dict()
    # Extract arrays (convert to 1D numpy arrays if needed)
    x = data_dict.get('x').ravel()
    y = data_dict.get('y').ravel()
    z = data_dict.get('z').ravel()
    vx = data_dict.get('vx').ravel()
    vy = data_dict.get('vy').ravel()
    vz = data_dict.get('vz').ravel()
    Sxx = data_dict.get('c_myStress[1]').ravel()
    Syy = data_dict.get('c_myStress[2]').ravel()
    Szz = data_dict.get('c_myStress[3]').ravel()
    Sxy = data_dict.get('c_myStress[4]').ravel()
    Sxz = data_dict.get('c_myStress[5]').ravel()
    Syz = data_dict.get('c_myStress[6]').ravel()
    
    # VOLUME (from box bounds) --------------------
    volume = (block_frame.get_bounds()[0,1] - block_frame.get_bounds()[0,0]) * (block_frame.get_bounds()[1,1] - block_frame.get_bounds()[1,0]) * (block_frame.get_bounds()[2,1] - block_frame.get_bounds()[2,0]);

    ## CONVECTIVE (e*v) --------------------
    energy = np.array([ke + pe for ke, pe in zip(data_dict.get('c_myKE'), data_dict.get('c_myPE'))])
    Jconv = np.array([np.sum(energy * np.transpose(vx)),np.sum(energy * np.transpose(vy)),np.sum(energy * np.transpose(vz))])

    ## CONDUCTIVE (S*v) --------------------
    Jcond=0.1*-0.000006241509*np.array([np.sum(Sxx * np.transpose(vx) + Sxy * np.transpose(vy) + Sxz * np.transpose(vz)),np.sum(Sxy * np.transpose(vx) + Syy * np.transpose(vy) + Syz * np.transpose(vz)),np.sum(Sxz * np.transpose(vx) + Syz * np.transpose(vy) + Szz * np.transpose(vz))])
    ## TOTAL FLUX --------------------
    Jtot = Jconv + Jcond # stress tensor Jcond is negative
    
    return J_Out(block_frame.get_timestep(), volume, Jtot, Jconv, Jcond) # NOT NORMALIZED BY VOLUME HERE

class K_Out: # define thermal conductivity output object
    def __init__(self, t, temp, v, ktot, acf):
        self._timestep=t
        self._temp=temp
        self._volume=v
        self._tc_tot=ktot
        self._acf=acf

    def pretty_output(self):    
        # Convert arrays to flat strings
        acf_str = '\t'.join(map(str, self._acf))
        ktot_str = '\t'.join(map(str, self._tc_tot))
        return f"{self._timestep}\t{ktot_str}\t{acf_str}\t{self._temp}\t{self._volume}"
        
    def get_timestep(self):
        return self._timestep
    
    def get_temp(self):
        return self._temp
    
    def get_acf(self):
        return self._acf
    
    def get_tc_tot(self):
        return self._tc_tot
    
def compute_tc(block_flux : J_Out, temp: int): 
    # computes thermal conductivity from heat flux data using fourier transform to approximate the autocorrelation function
    k_B=8.6173303e-5 # eV/Kelvin
    timestep = block_flux.get_timestep()
    volume = block_flux.get_volume()
    flux = block_flux.get_flux_tot()
    N = len(flux[:,0]) # number of time steps
    tc_array=np.empty(N,3)
    for i, flux_norm in enumerate(np.split(flux/volume,indices_or_sections=3,axis=1)):
        ft = np.fft.fft(flux_norm, n=2*N, axis=0) # fourier transform with N zeros at the end
        acf_J = np.real(np.fft.ifft(ft*np.conjugate(ft), axis=0)[:N,:]) # real components of inverse fourier transform of the fourier transform times complex conjugate
        acf_J_norm = acf_J/( N - np.arange(N)[:,None]) # normalized acf
        tc_array[:,i] = np.trapz(acf_J_norm, axis=0)*volume/(k_B*temp*temp) # integrate acf to get thermal conductivity
    return K_Out(timestep, temp, volume, tc_array, acf_J_norm)

if __name__ == "__main__":
    # test the time of operation
    start = time.time()
    main()
    end = time.time()
    print(f"Done after {(end - start):.6f} seconds")