import os
import pandas as pd
import matplotlib.pyplot as plt
from prody import *
import numpy as np

# Load reference structures
ref_gdp = parsePDB('gdp_bound_wt.pdb')
ref_gtp = parsePDB('gtp_bound_wt.pdb')

# Store results
results = []

# Define color and shape mappings
color_map = {
    ('WT', 'NRAS_BRAF'): 'darkblue',
    ('Q61R', 'NRAS_BRAF'): 'lightblue',
    ('WT', 'NRAS_SOS1'): 'orange',
    ('Q61R', 'NRAS_SOS1'): 'yellow',
    ('WT', 'NRAS'): 'darkgreen',
    ('Q61R', 'NRAS'): 'lightgreen',
}

shape_map = {
    'GTP': '^',
    'GDP': 's',
    'no_ligand': 'o',
}

# Loop over PDBs
for pdb_file in os.listdir():
    if pdb_file.endswith('.pdb') and '_model_0' in pdb_file:
        parts = pdb_file.replace('_model_0.pdb', '').split('_')
        nras_type = parts[1] if 'Q61R' in parts[1] else 'WT'
        complex_type = f"NRAS_{parts[2]}"
        ligand = parts[3]

        model = parsePDB(pdb_file)
        # Select only NRAS residues (first 166 residues) and CA atoms
        sel_model = model.select('resnum <= 166 and name CA')
        sel_gdp = ref_gdp.select('resnum <= 166 and name CA')
        sel_gtp = ref_gtp.select('resnum <= 166 and name CA')

        if sel_model and sel_gdp and sel_gtp:
            # Align structures before calculating RMSD
            sel_model, sel_gdp = matchAlign(sel_model, sel_gdp)
            sel_model, sel_gtp = matchAlign(sel_model, sel_gtp)
            
            # Calculate RMSD after alignment
            rmsd_gdp = calcRMSD(sel_model, sel_gdp)
            rmsd_gtp = calcRMSD(sel_model, sel_gtp)

            results.append({
                'file': pdb_file,
                'nras_type': nras_type,
                'complex': complex_type,
                'ligand': ligand,
                'rmsd_gdp': rmsd_gdp,
                'rmsd_gtp': rmsd_gtp,
                'color': color_map[(nras_type, complex_type)],
                'marker': shape_map[ligand]
            })

# Plot results
df = pd.DataFrame(results)
plt.figure(figsize=(8, 6))

for _, row in df.iterrows():
    plt.scatter(row['rmsd_gdp'], row['rmsd_gtp'],
                color=row['color'], marker=row['marker'],
                s=100, edgecolors='k', label=row['file'])

plt.xlabel('RMSD to GDP-bound WT (Å)', fontsize=12)
plt.ylabel('RMSD to GTP-bound WT (Å)', fontsize=12)
plt.title('NRAS Predicted Structures vs Reference RMSD', fontsize=14)
plt.grid(True)

# Optional: Avoid duplicate legends
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()
plt.savefig("nras_rmsd_plot.png", dpi=300)
plt.show() 