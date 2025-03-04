import argparse
import numpy as np
import matplotlib.pyplot as plt
from openmm.app import *
from openmm import *
from openmm.unit import *
import os
from tqdm import tqdm

# python assignment3_LMcDonnell_fig.py ala.pdb --out ala_out.png --steps 1000000
def dihedral(p1, p2, p3, p4):
    '''Calculate dihedral angle in radians'''
    v12 = np.array(p1-p2)
    v32 = np.array(p3-p2)
    v34 = np.array(p3-p4)
    
    cp0 = np.cross(v12, v32)
    cp1 = np.cross(v32, v34)
    
    dot = np.dot(cp0, cp1)
    norm1 = np.dot(cp0, cp0)
    norm2 = np.dot(cp1, cp1)
    
    dot /= np.sqrt(norm1 * norm2)
    dot = np.clip(dot, -1.0, 1.0)
    
    angle = np.arccos(dot)
    sdot = np.dot(v12, cp1)
    angle *= np.sign(sdot)
    
    return angle

def run_simulation(pdb_file, aindex, bindex, friction, steps=10000):
    '''Runs an OpenMM Langevin simulation with a given friction coefficient'''
    
    pdb = PDBFile(pdb_file)
    ff = ForceField('amber14-all.xml')
    system = ff.createSystem(pdb.topology, ignoreExternalBonds=True)
    
    # Langevin integrator with varying friction coefficient
    integrator = LangevinIntegrator(300 * kelvin, friction / picosecond, 1 * femtosecond)
    integrator.setRandomNumberSeed(42)

    # Setup simulation
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    dihedrals = []

    for step in tqdm(range(steps), desc=f"Simulating (γ={friction} ps^-1)"):
        simulation.step(1)
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)

        a1 = pos[0]
        a2 = pos[1]
        a3 = pos[2]
        a4 = pos[3]
        d = dihedral(a1, a2, a3, a4)
        dihedrals.append(d)

    return (np.rad2deg(dihedrals) + 360) % 360

# Set up argparse for command-line arguments
parser = argparse.ArgumentParser(description='Compare dihedral angle evolution under different friction coefficients.')
parser.add_argument('pdb', help='Input PDB file')
parser.add_argument('--aindex', type=int, default=1, help='Index of first dihedral atom')
parser.add_argument('--bindex', type=int, default=2, help='Index of second dihedral atom')
parser.add_argument('--steps', type=int, default=10000, help='Number of simulation steps (1fs)')
parser.add_argument('--output', default='dihedral_distribution.png', help='Output filename for graph')
args = parser.parse_args()

# Define different friction coefficients to test
friction_values = [0.1, 1.0, 10.0]  # ps^-1

# Run simulations for different friction values
results = {}
for friction in friction_values:
    results[friction] = run_simulation(args.pdb, args.aindex, args.bindex, friction, args.steps)

# Create histogram plots for each friction coefficient
plt.figure(figsize=(10, 6))

for friction, dihedrals in results.items():
    plt.hist(dihedrals, bins=range(361), density=True, alpha=0.6, label=f'γ={friction} ps$^{-1}$')

plt.xlim(0, 360)
plt.xlabel('Dihedral Angle (Degrees)')
plt.ylabel('Frequency')
plt.title('Dihedral Angle Distribution for Different Friction Coefficients')
plt.legend()
plt.savefig(args.output, bbox_inches='tight')
plt.show()
