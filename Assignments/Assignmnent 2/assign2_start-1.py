#!/usr/bin/env python3

import openmm
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
import matplotlib
matplotlib.use('Agg') # don't open a display
import matplotlib.pyplot as plt
import scipy
import argparse
import scipy.signal

def dihedral(p1, p2, p3, p4):
    '''Return dihedral angle in radians between provided points.
    This is the same calculation used by OpenMM for force calculations. '''
    v12 = np.array(p1-p2)
    v32 = np.array(p3-p2)
    v34 = np.array(p3-p4)
    
    #compute cross products 
    cp0 = np.cross(v12,v32)
    cp1 = np.cross(v32,v34)
    
    #get angle between cross products
    dot = np.dot(cp0,cp1)
    if dot != 0:
        norm1 = np.dot(cp0,cp0)
        norm2 = np.dot(cp1,cp1)
        dot /= np.sqrt(norm1*norm2)
        
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0

    if dot > 0.99 or dot < -0.99:
        #close to acos singularity, so use asin isntead
        cross = np.cross(cp0,cp1)
        scale = np.dot(cp0,cp0)*np.dot(cp1,cp1)
        angle = np.arcsin(np.sqrt(np.dot(cross,cross)/scale))
        if dot < 0.0:
            angle = np.pi - angle
    else:
        angle = np.arccos(dot)
    
    #figure out sign
    sdot = np.dot(v12,cp1)
    angle *= np.sign(sdot)
    return angle
    
    
def make_rotation_matrix(p1, p2, angle):
    '''Make a rotation matrix for rotating angle radians about the p2-p1 axis'''
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    vec = np.array((p2-p1))
    x,y,z = vec/np.linalg.norm(vec)
    cos = np.cos(angle)
    sin = np.sin(angle)
    R = np.array([[cos+x*x*(1-cos), x*y*(1-cos)-z*sin, x*z*(1-cos)+y*sin],
                  [y*x*(1-cos)+z*sin, cos+y*y*(1-cos), y*z*(1-cos)-x*sin],
                  [z*x*(1-cos)-y*sin, z*y*(1-cos)+x*sin, cos+z*z*(1-cos)]])
    return R
    
    
def moving_atoms(pdb,a=1,b=2):
    '''Identify the atoms on the b side of a dihedral with the central
    atoms a and b.  This is not the most efficient algorithm, but 
    for small molecules it really does not matter.
    A boolean mask of these atoms is returned.'''
    moving = np.zeros(pdb.topology.getNumAtoms())
    moving[b] = 1
    moving[a] = -1
    changed = True
    while changed:
        changed = False
        for b in pdb.topology.bonds():
            if (moving[b.atom1.index] + moving[b.atom2.index]) == 1:
                moving[b.atom1.index] = moving[b.atom2.index] = 1
                changed = True
    moving[1] = 0    
    return moving.astype(bool)      



parser = argparse.ArgumentParser(description='CompStruct Assignment 2')
parser.add_argument('pdb',help='input PDB file')
parser.add_argument('--aindex',help='index of first dihedral atom',type=int,default=1)
parser.add_argument('--bindex',help='index of second dihedral atom',type=int,default=2)
parser.add_argument('--state_temp',help='temperature to compare states at',type=int,default=300)
parser.add_argument('--output',help='output filename for graph',default='out.pdf')
args = parser.parse_args()

# read file
pdb = PDBFile(args.pdb)

# setup openmm system using amber 14 forcefield which defines the potential energy function
ff = ForceField('amber14-all.xml')
system = ff.createSystem(pdb.topology,ignoreExternalBonds=True)
# we aren't actually simulating, but need a simulation object to calculate energies
integrator = VerletIntegrator(1*femtosecond)
simulation = Simulation(pdb.topology, system, integrator)

# store atom positions
# note that more code is necessary for this to be a general solution
# as opposed to only working with our carefully prepared inputs
origpos = np.array(pdb.getPositions()._value)
newpos = origpos.copy()

capos = origpos[args.aindex]
cpos = origpos[args.bindex]

# a boolean mask of the atoms that should be rotated around the dihedral
mask = moving_atoms(pdb, args.aindex, args.bindex)

# we will output a pdb of the result of our rotations around the dihedral
out = open('rot.pdb','wt')
energies = []
angles = []
for d in range(0,360,1):
    # rotate atoms
    R = make_rotation_matrix(capos,cpos,np.deg2rad(d))    
    newpos[mask] = np.matmul(R,origpos[mask].T).T
     
    # setPositions to newpos in simulation (TODO)
    simulation.context.setPositions(newpos * nanometers)
    
    # get simulation.context state, fetching energy (TODO)
    state = simulation.context.getState(getEnergy=True)


    # record the energy (TODO)
    energy = state.getPotentialEnergy()  # in kJ/mol
    energies.append(energy)


    # record the dihedral
    d = dihedral(*newpos[:4])
    if d < 0: d += 2*np.pi
    angles.append(np.rad2deg(d))
    
    #write out modified structure
    PDBFile.writeModel(pdb.topology, newpos*nanometers,out,modelIndex=d)

out.close()

# make figure of energies and probabilities at different temperatures
plt.figure(figsize=(8,4),dpi=300)

E = [e._value for e in energies] # unitless for numpy manipulation

# plot energy on left y-axis
plt.plot(angles,E,color='gray')

plt.ylabel("Energy (kJ/mol)",color='gray')
plt.xlabel("Dihedral (Degrees)")
ax2 = plt.gca().twinx()
ax2.set_ylabel("Probability")

# plot probabilities for different temperatures on right y-axis
for T in (100,300,500,1000):
    T = T*kelvin
    
    # compute probabilities of each state (TODO)
    k_B = openmm.unit.constants.BOLTZMANN_CONSTANT_kB  # J/K (Boltzmann constant)
    N_A = openmm.unit.constants.AVOGADRO_CONSTANT_NA  # NA (Avogadro's number)
    beta = 1 / (k_B * T)
    probs = [np.exp(-(e/N_A)/(k_B*T)) for e in energies]
    Z=np.sum(probs)
    probs=probs/Z
    
    # plot probabilities (probs)
    ax2.plot(angles,probs,label=T,color=plt.cm.viridis(T._value/1000))
    
    # calculate average energy, entropy, and free energy (TODO)
    ave_energy = np.sum(np.array(energies) * probs)
    entropy = k_B * (np.log(Z) + beta * ave_energy)
    FE = - k_B * T * np.log(Z)

    # print out configurational partition function (Z), average energy, entropy (S), and free energy (F)
    # These should be represented as openmm Quantity objects with the right units
    # Do not change this print statement
    print(f'T={T},\tZ={Z:.3g},\t<U>={ave_energy.format("%.2f")},\tS={entropy.format("%.2f")},\tF={FE.format("%.2f")}')

plt.legend(ncol=4,loc='upper center')
plt.savefig(args.output,bbox_inches='tight')


# Now we will divide the energy landscape into two pieces and calculate
# the relative free energies of these two states.

# find all peaks
peaks = scipy.signal.find_peaks(E)[0]

# categorize our microstates based on their location relative to the two tallest peaks
states = np.array([peaks[0] <= i < peaks[1] for i in range(360)])

# state A corresponds to all positions i where states[i] is True
# state B corresponds to all positions i where states[i] is False

# for the specified temperature...
T = args.state_temp*kelvin
print(f"At {T}:")

# calculate probabilities for specified temp (TODO)
k_B = openmm.unit.constants.BOLTZMANN_CONSTANT_kB # Boltzmann constant in J/K
beta = 1 / (k_B * T)

Z_A = np.sum(np.exp(-beta * np.array(energies)[states == True]))
Z_B = np.sum(np.exp(-beta * np.array(energies)[states == False]))
pA = Z_A / (Z_A + Z_B)
pB = Z_B / (Z_A + Z_B)

print(f'Probability of state A is {pA:.4f}')
print(f'Probability of state B is {pB:.4f}')

# calculate <U>, S, and free energy of each state and print

# state A (TODO)

energies_A = np.array(energies)[states == True]
U_A = np.sum(energies_A * np.exp(-beta * energies_A)) / Z_A
S_A = k_B * (np.log(Z_A) + beta * U_A)
FE_A = -k_B * T * np.log(Z_A)



print(f'State A:\t<U>={U_A.format("%.2f")}\tS={S_A.format("%.2f")}\tF={FE_A.format("%.2f")}')

# state B (TODO)

energies_B = np.array(energies)[states == False]
U_B = np.sum(energies_B * np.exp(-beta * energies_B)) / Z_B
S_B = k_B * (np.log(Z_B) + beta * U_B)
FE_B = -k_B * T * np.log(Z_B)




print(f'State B:\t<U>={U_B.format("%.2f")}\tS={S_B.format("%.2f")}\tF={FE_B.format("%.2f")}')

# for your own edificaton, verify that F = U - TS = -k_B T ln(Z)
F_from_U_TS = U_A - T * S_A
F_from_Z = -k_B * T * np.log(Z_A)
print(f"Free energy from U - TS: {F_from_U_TS:.3f}")
print(f"Free energy from -k_B T ln(Z): {F_from_Z:.3f}")

if np.isclose(F_from_U_TS, F_from_Z, atol=1e-6):
    print("The two expressions for free energy are approximately equal.")
else:
    print("The two expressions for free energy are not equal.")

# calculate the free energy difference between A and B (TODO)
# verify that you get the same answer if you subtract the free energies or take the ratio of probabilities
deltaG = FE_A - FE_B

print('\u0394G is',deltaG.format('%.3f'))
