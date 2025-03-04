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
import os
from tqdm import tqdm

# ./assignment3_LMcDonnell.py ala.pdb --out ala_out.png 
# ./assignment3_LMcDonnell.py gly.pdb --out gly_out.png 
# ./assignment3_LMcDonnell.py phe.pdb --out phe_out.png 


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


parser = argparse.ArgumentParser(description='CompStruct Assignment 3')
parser.add_argument('pdb', help='input PDB file')
parser.add_argument('--aindex', help='index of first dihedral atom', type=int, default=1)
parser.add_argument('--bindex', help='index of second dihedral atom', type=int, default=2)
parser.add_argument('--integrator', help='integrator to use', choices=['verlet','langevin'], default='langevin')
parser.add_argument('--temp', help='temperature to simulate at', type=int, default=300)
parser.add_argument('--steps', help='number of simulation steps (1fs)', type=int, default=10000)
parser.add_argument('--output', help='output filename for graph', default='out.png')
args = parser.parse_args()

# Setup OpenMM system using amber14 forcefield
pdb = PDBFile(args.pdb)
ff = ForceField('amber14-all.xml')
system = ff.createSystem(pdb.topology, ignoreExternalBonds=True)

# Set integrator
if args.integrator == 'verlet':
    integrator = VerletIntegrator(1 * femtosecond)
else:
    integrator = LangevinIntegrator(args.temp * kelvin, 1 / picosecond, 1 * femtosecond)
    integrator.setRandomNumberSeed(42)

simulation = Simulation(pdb.topology, system, integrator)
platforms = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print("Available OpenMM platforms:", platforms)


# Store atom positions
origpos = np.array(pdb.getPositions()._value)
newpos = origpos.copy()

capos = origpos[args.aindex]
cpos = origpos[args.bindex]

# Identify which atoms move when rotating around the dihedral
mask = moving_atoms(pdb, args.aindex, args.bindex)

# We will output a pdb of the result of our rotations around the dihedral
out = open('rot.pdb','wt')
energies = []
angles = []

for deg in range(0, 360, 1):
    # rotate atoms
    R = make_rotation_matrix(capos, cpos, np.deg2rad(deg))    
    newpos[mask] = np.matmul(R, origpos[mask].T).T
     
    simulation.context.setPositions(newpos * nanometers)

    # get potential energy
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    energies.append(energy.in_units_of(kilojoule_per_mole))

    # measure actual dihedral after rotation
    dval = dihedral(*newpos[:4])
    if dval < 0:
        dval += 2 * np.pi
    angles.append(np.rad2deg(dval))
    
    # write out modified structure
    PDBFile.writeModel(pdb.topology, newpos * nanometers, out, modelIndex=deg)

out.close()


T = args.temp * kelvin
beta = 1 / (MOLAR_GAS_CONSTANT_R * T)
E_array = np.array([e._value for e in energies])

boltz = np.exp(-beta * E_array * kilojoule_per_mole)
Z = np.sum(boltz)
probs = boltz / Z

simulation.context.setPositions(pdb.getPositions())  # reset initial positions
dihedrals = []

for step in tqdm(range(args.steps), desc="Simulating"):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)

    a1 = pos[0]
    a2 = pos[1]
    a3 = pos[2]
    a4 = pos[3]
    d = dihedral(a1, a2, a3, a4)
    dihedrals.append(d)

# convert to degrees
dihedrals_deg = (np.rad2deg(dihedrals) + 360) % 360
cnts, bins = np.histogram(dihedrals_deg, range(361))

# this is what the autograder will look at (with a small time step)
print(cnts)

# make plot
plt.hist(dihedrals_deg, bins=range(361), density=True)
plt.plot(angles, probs, label=r'$\frac{1}{\hat{Z}}e^{\frac{-U}{k_BT}}$')
plt.xlim(0, 360)
plt.legend(fontsize=16)
plt.xlabel('Dihedral Angle (Degrees)')
plt.ylabel('Frequency/Probability')
plt.title(os.path.split(args.pdb)[-1][:-4].upper())
plt.savefig(args.output, bbox_inches='tight')