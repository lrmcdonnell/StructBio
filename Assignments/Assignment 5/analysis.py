#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The Bio.Application")

# feel free to clean up the imports
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis import align, rms
import MDAnalysis.transformations as transformations
from MDAnalysis.analysis.distances import distance_array

warnings.filterwarnings("ignore",message="DCDReader")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, argparse

#python analysis.py --topology system.pdb --trajectory trajectory.dcd --ligand_sel "resname UNL" --selA 'not name H* and resid 61' --selB 'not name H* and byres around 4 (resid 61)' --output_prefix gtp_bound_r
#python analysis.py --topology system_gdp.pdb --trajectory trajectory_gdp.dcd --ligand_sel "resname UNL" --selA 'not name H* and resid 61' --selB 'not name H* and byres around 4 (resid 61)' --output_prefix gdp_bound_r


parser = argparse.ArgumentParser(description='Assignment 5 analysis of an MD trajectory')
parser.add_argument("--topology",required=True,help="Topology PDB file from OpenMM of ligand/receptor system")
parser.add_argument("--trajectory",required=True,help="Trajectory file of ligand/receptor system")
parser.add_argument("--step",type=int,default=100,help="Frequency to downsample simulation.")
parser.add_argument("--cutoff",type=float,default=4.0,help="Distance threshold for contacts analysis")
parser.add_argument("--ligand_sel",required=True,help="Selection of ligand for RMSD analysis")
parser.add_argument("--selA",required=True,help="Selection A for contacts analysis")
parser.add_argument("--selB",required=True,help="Selection A for contacts analysis")
parser.add_argument("--output_prefix",required=True,help="Prefix for all output files")
parser.add_argument("--output_traj",required=False,help="Base filename for outputing the processed trajectory")

args = parser.parse_args()

# KEEP all the hardcoded output file names as this is how your script will be graded.

# Preprocess trajectory
U = MDAnalysis.Universe(args.topology,args.trajectory,in_memory=True,in_memory_step=args.step)
protein = U.select_atoms('protein')
notwater = U.select_atoms('not resname HOH WAT')
transforms = [transformations.unwrap(protein),
              transformations.center_in_box(protein),
              transformations.wrap(U.select_atoms('not protein'),compound='residues')]
              
U.trajectory.add_transformations(*transforms)

u = MDAnalysis.Merge(notwater).load_new(
         AnalysisFromFunction(lambda ag: ag.positions.copy(), notwater).run().results['timeseries'],
         format=MemoryReader)
         
if args.output_traj:
    system = u.select_atoms('all')       
    system.write(f'{args.output_traj}.pdb')
    system.write(f'{args.output_traj}.dcd',frames=u.trajectory[1:])

# Align trajectory using alpha carbons to first frame
aligner = align.AlignTraj(u, u, select='name CA', in_memory=True)
aligner.run()

# RMSD Calculations
# RMSD Calculations
ligand = u.select_atoms(args.ligand_sel)
protein = u.select_atoms('protein')

# Initialize arrays for storing RMSD values
first_rmsds = []
last_rmsds = []

# Store reference positions from first frame
u.trajectory[0]
ref_first_positions = ligand.positions.copy()

# Store reference positions from last frame
u.trajectory[-1]
ref_last_positions = ligand.positions.copy()

# Iterate through trajectory once and calculate both RMSDs
u.trajectory[0]  # Reset to beginning
for ts in u.trajectory:
    # For each frame, we need to align by protein first
    # Then calculate RMSD of the ligand
    
    # Calculate RMSD to first frame (after alignment)
    rmsd_first = rms.rmsd(ligand.positions, ref_first_positions, 
                          superposition=False)  # No superposition, alignment already done
    first_rmsds.append(rmsd_first)
    
    # Calculate RMSD to last frame (after alignment)
    rmsd_last = rms.rmsd(ligand.positions, ref_last_positions, 
                         superposition=False)  # No superposition, alignment already done
    last_rmsds.append(rmsd_last)

plt.plot(first_rmsds,label='RMSD to First Frame')
plt.plot(last_rmsds,label='RMSD to Last Frame')
plt.xlabel('Frame')
plt.ylabel('RMSD (\u212B)')
plt.legend()
plt.savefig(f'{args.output_prefix}_rmsd.png',bbox_inches='tight',dpi=300)

#first_rmsds should have the rmsd of args.ligand_selection to the first frame
#and last_rmsds to the last frame in the alpha carbon aligned structure
#these files must be output with these file names for the autograder
np.save('first_rmsds.npy',first_rmsds)
np.save('last_rmsds.npy',last_rmsds)

# Contacts analysis
selA = u.select_atoms(args.selA)
selB = u.select_atoms(args.selB)

# Initialize contact matrix
contacts = np.zeros((len(selA), len(selB)))

# Calculate distances and contacts for each frame
for ts in u.trajectory:
    distances = distance_array(selA.positions, selB.positions)
    contacts += (distances <= args.cutoff).astype(float)

# Normalize by number of frames
contacts /= len(u.trajectory)

# Create labels for the heatmap
xlabels = [f"{atom.resid} {atom.resname} {atom.name}" for atom in selB]
ylabels = [f"{atom.resid} {atom.resname} {atom.name}" for atom in selA]

plt.figure(figsize=(len(xlabels)*.2+1,len(ylabels)*.2+1))
sns.heatmap(contacts,xticklabels=xlabels,yticklabels=ylabels, vmin=0,vmax=1, cmap="viridis",cbar_kws={'pad':0.01,'label':'Contact Frequency'})
plt.tight_layout()
plt.savefig(f'{args.output_prefix}_contacts.png',bbox_inches='tight',dpi=300)

# For the autograder - should be a len(selA) x len(selB) matrix of numbers
# between 0 and 1 representing contact frequencies.
np.save('contacts.npy',contacts)

