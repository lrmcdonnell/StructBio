#!/usr/bin/env python3

# feel free to clean up the imports
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.dihedrals import *
from MDAnalysis.analysis import contacts



import matplotlib.pyplot as plt
import numpy as np
import sys, argparse


parser = argparse.ArgumentParser(description='Analyze a MD trajectory using MDAnalysis')
parser.add_argument("topology",help="Topology PDB file from OpenMM")
parser.add_argument("trajectory",help="Trajectory file")
args = parser.parse_args()

# KEEP all the hardcoded output file names as this is how your script will be graded.

# Preprocess trajectory
u = MDAnalysis.Universe(args.topology, args.trajectory)
protein = u.select_atoms("protein")
protein.write("protein_only.pdb")
with MDAnalysis.Writer("protein_only.dcd", protein.n_atoms) as W:
    for ts in u.trajectory:
        W.write(protein)

protein_universe = MDAnalysis.Universe("protein_only.pdb", "protein_only.dcd")
protein_universe.transfer_to_memory()

# Align trajectory to the first frame using all backbone atoms
ref = MDAnalysis.Universe("protein_only.pdb")
backbone = protein_universe.select_atoms('backbone')
align.AlignTraj(protein_universe, ref, select="backbone", ref_frame=0, in_memory=True).run()


# RMSD Calculations
# Calculate RMSD to first frame
rmsd_first = rms.RMSD(protein_universe, ref, select="backbone", ref_frame=0)
rmsd_first.run()
rmsd_first = rmsd_first.results.rmsd.T  # Transpose for easier plotting

rmsd_last = rms.RMSD(protein_universe, protein_universe, select="backbone", ref_frame=-1)
rmsd_last.run()
rmsd_last = rmsd_last.results.rmsd.T  # Transpose for consistency

# Extract Time and RMSD values
time = rmsd_first[1]  # Column 1 = time (ps)
rmsd_first_values = rmsd_first[2]  # Column 2 = RMSD values
rmsd_last_values = rmsd_last[2]  # Column 2 = RMSD values for last frame

# Save RMSD values
np.save('rmsd_first.npy', rmsd_first_values)
np.save('rmsd_last.npy', rmsd_last_values)

# Plot RMSD
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(time, rmsd_first_values, 'b-', label="RMSD to First Frame")
ax.plot(time, rmsd_last_values, 'r-', label="RMSD to Last Frame")
ax.set_xlabel("Time (ps)")
ax.set_ylabel(r"RMSD ($\AA$)")
ax.set_title("RMSD of Protein Backbone Over Time")
ax.legend(loc="best")

# Save figure
fig.savefig("rmsd_plot.png", dpi=300)


# RMSF Calculations - Align to average structure
average = align.AverageStructure(protein_universe, ref_structure=ref, select="backbone").run()
avg_universe = average.results.universe
align.AlignTraj(protein_universe, avg_universe, select="backbone", in_memory=True).run()
'''
# After aligning to average, calculate RMSF
calphas = protein_universe.select_atoms("name CA")
rmsf_calc_ca = rms.RMSF(calphas).run()
ca_rmsf = rmsf_calc_ca.results.rmsf

np.save('ca_rmsf.npy', ca_rmsf) #alpha carbon only rmsf
protein = protein_universe.select_atoms("protein")
rmsf_calc_all = rms.RMSF(protein).run()
all_rmsf = rmsf_calc_all.results.rmsf


for i, atom in enumerate(protein.atoms):
    atom.tempfactor = all_rmsf[i]

protein.atoms.write('rmsf.pdb')

# Plot RMSF
plt.figure(figsize=(10, 6))
plt.plot(calphas.residues.resids, ca_rmsf)
plt.xlabel('Residue Number')
plt.ylabel('RMSF (Å)')
plt.title('Root Mean Square Fluctuation by Residue')
plt.savefig('rmsf_plot.png', dpi=300)
plt.close()

# Ramachandran 
residu_62 = u.select_atoms('protein and resid 62')
# Ramachandran 
rama = MDAnalysis.analysis.dihedrals.Ramachandran(residu_62).run()
rama_angles = rama.results['angles']
np.save('ramachandran.npy',rama_angles)

plt.figure(figsize=(8, 8))
# Use the built-in plot method with reference data
rama.plot(color='black', marker='.', ref=True)
plt.title('Ramachandran Plot for Residue 62')
plt.savefig('ramachandran_res62.png', dpi=300)
plt.close()

# Janin
janin = MDAnalysis.analysis.dihedrals.Janin(residu_62).run()
janin_angles = janin.results['angles']
np.save('janin.npy', janin_angles)

# Plot Janin for residue 62

plt.figure(figsize=(8, 8))
# Use the built-in plot method with reference data
janin.plot(ref=True, marker='.', color='black')
plt.title('Janin Plot for Residue 62')
plt.savefig('janin_res62.png', dpi=300)
plt.close()

res_q61 = protein_universe.select_atoms("resid 62 and name CA") 
res_binding = protein_universe.select_atoms("resid 12 and name CA") # Adjust residue ID for binding site

distances = []
for ts in protein_universe.trajectory:
    distances.append(np.linalg.norm(res_q61.positions - res_binding.positions))


plt.figure(figsize=(10, 6))
plt.plot(frames[:141], distances[:141], label="Q61 to Binding Site Distance", color="red")
plt.xlabel('Frame Number')
plt.ylabel('Distance (Å)')
plt.title('Distance Between Residue 61 and Binding Site Over Time')
plt.legend()
plt.savefig('q61_binding_distance_mut.png', dpi=300)
plt.close()
'''