import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'Dejavu sans', 'serif':['FreeSerif'], 'size':18})

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIModel


def tfim_2d(p):
    """
    Finds the ground state of the 2D transverse field ising model: 
    H = -J \sum_{<i,j>} \sigma^{x}_{i} \sigma^{x}_{j} - g \sum_{i} \sigma^{z}_{i}
    
    Input:
        p - Dictionary with all relevant parameters:
            Lx: x-Dimension of the lattice
            Ly: y-Dimension of the lattice
            J: coupling strength
            g: transverse field
            conserve: Conserve parity (number of excited qubits)? If yes, set conserve='parity', else conserve=None
            energy_convergence_goal: stop dmrg sweeps if absolute change in energy between sweeps is less than this threshold
            D: maximum bond dimension
            svd_min: discard all singular values less than this threshold in any SVD
    Returns:
        meas - Dictionary with the relevant obsrevables (here just the ground state Energy E)
        psi - the ground state wavefunction in MPS form
        M - the TFIM model as constructed by TeNPy
    """
    
    # set the conservation law
    if p['conserve'] != 'parity':
        p['conserve'] = None
    
    # print parsed parameters of the model
    if p['verbose']:
        print("Lx={Lx:d}, Ly={Ly:d}, J={J:.2f}, g={g:.2f}, conservation={conserve}".format(Lx=p['Lx'], Ly=p['Ly'], J=p['J'], g=p['g'], conserve=p['conserve']))
    
    # set model parameters
    model_params = dict(lattice='Square', order='default', Lx=p['Lx'], Ly=p['Ly'], J=p['J'], g=p['g'], bc_MPS='finite', bc_x='open', bc_y='ladder', conserve=p['conserve'])

    # create instance of model
    M = TFIModel(model_params)
        
    # initialse all qubits in "up" == "0" and create MPS
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    
    # set up DMRG parameters for ground state search
    dmrg_params = {'mixer': 'SubspaceExpansion',
                   'mixer_params': {'amplitude': 1.0e-6, 'decay': 1.2, 'disable_after': 10},
                   'max_E_err': p['energy_convergence_goal'],
                   'trunc_params': {'chi_max': p['D'], 'svd_min': p['svd_min']},
                   'combine': True,
                   'active_sites': 2
                  }
    
    # run the DMRG algorithm
    info = dmrg.run(psi, M, dmrg_params)
    
    # store GS expectation values in measurement dictionary
    meas = {}
    meas['E'] = info['E'] # energy
    
    # print GS energy and bond dimension of the state
    if p['verbose']:
        print("E = {E:.13f}".format(E=meas['E']))
        print("final bond dimensions: ", psi.chi, "\n")
        
    return meas, psi, M


### (4) Visualise the ground state energy as a function of the field

# gather all parameters in one dictionary
p = {}
p['Lx'] = 4 # x-dimension of the lattice
p['Ly'] = 4 # y-dimension of the lattice
p['J'] = 1. # coupling strength J
p['conserve'] = None # set conservation law: 'parity' (total number of excitations) or None
p['verbose'] = False

p['energy_convergence_goal'] = 1.0e-10 # stop dmrg sweeps if absolute change in energy between sweeps is less than this threshold
p['D'] = 300 # maximum bond dimension
p['svd_min'] = 1.0e-10 # discard all singular values less than this threshold in any SVD

g_array = np.arange(start=-1.0, stop=1.1, step=0.1) # array for transverse field
E = np.zeros(shape=g_array.shape, dtype=float) # initialse array for groundstate energy
for i, g in enumerate(g_array):
    print("g = {g:.2f}".format(g=g))

    # add current transverse field to dictionary
    p['g'] = g
    
    # find the groundstate
    meas, psi, M = tfim_2d(p)
    
    # store the ground state energy, X = <\sigma^_{x}> and Z = <\sigma^{z}>
    E[i] = meas['E']

# plot ground state energy and save plot
fig = plt.figure(figsize=(12,10))
ax = plt.gca()
ax.plot(g_array, E / p['J'], '.-')
ax.set_xlabel(r'$g/J$');
ax.set_ylabel(r'$E_{0}/J$')
plt.savefig('GroundstateEnergy_TFIM.pdf')
