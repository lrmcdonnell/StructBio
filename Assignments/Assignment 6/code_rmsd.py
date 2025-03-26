import prody
import matplotlib.pyplot as plt
import os
import re

# Config
MODEL_DIR = "models"
GDP_REF = "gdp_bound_wt.pdb"
GTP_REF = "gtp_bound_wt.pdb"

print(f"Current working directory: {os.getcwd()}")
print(f"Looking for files in directory: {MODEL_DIR}")
print(f"Reference files: {GDP_REF}, {GTP_REF}")

# Load reference NRAS structures 
print("\nLoading reference structures...")
try:
    ref_gdp = prody.parsePDB(GDP_REF)
    print(f"✓ Loaded GDP reference: {ref_gdp.numAtoms()} atoms")
    ref_gdp = ref_gdp.select("protein and chain A")
    print(f"✓ Selected NRAS chain A: {ref_gdp.numAtoms()} atoms")
    
    ref_gtp = prody.parsePDB(GTP_REF)
    print(f"✓ Loaded GTP reference: {ref_gtp.numAtoms()} atoms")
    ref_gtp = ref_gtp.select("protein and chain A")
    print(f"✓ Selected NRAS chain A: {ref_gtp.numAtoms()} atoms")
except Exception as e:
    print(f"❌ Error loading reference structures: {e}")
    exit(1)

# Plot styling
ligand_markers = {"GDP": "^", "GTP": "s", "no_ligand": "o"}
system_colors = {
    ("NRAS:BRAF", "WT"): "#0570b0",
    ("NRAS:BRAF", "Q61R"): "#74add1",
    ("NRAS:SOS1", "WT"): "#d95f02",
    ("NRAS:SOS1", "Q61R"): "#fdc086",
    ("NRAS", "WT"): "#1b9e77",
    ("NRAS", "Q61R"): "#66c2a5",
}

# Data collection
data = []

print("\nProcessing files:")
model_files = os.listdir(MODEL_DIR)
print(f"Found {len(model_files)} files in {MODEL_DIR}")

for fname in model_files:
    if not fname.endswith(".pdb"):
        print(f"Skipping non-PDB file: {fname}")
        continue

    print(f"\nChecking file: {fname}")
    # Updated regex pattern to match our actual file names
    pattern = r"NRAS_(WT|Q61R)_NRAS_(BRAF|SOS1|)_(GDP|GTP|no_ligand)_model_0\.pdb"
    match = re.match(pattern, fname)
    if not match:
        print(f"❌ File pattern doesn't match: {fname}")
        print(f"Expected pattern: {pattern}")
        continue

    mut, system_raw, ligand_raw = match.groups()
    print(f"✓ Matched: mutation={mut}, system={system_raw}, ligand={ligand_raw}")
    
    # Convert ligand name
    ligand = {"GDP": "GDP", "GTP": "GTP", "no_ligand": "unbound"}[ligand_raw]
    
    # Convert system name
    system = {
        "BRAF": "NRAS:BRAF",
        "SOS1": "NRAS:SOS1",
        "": "NRAS"
    }[system_raw]

    filepath = os.path.join(MODEL_DIR, fname)
    try:
        pred = prody.parsePDB(filepath)
        print(f"✓ Loaded prediction: {pred.numAtoms()} atoms")
        pred = pred.select("protein and chain A")
        if not pred:
            print(f"⚠️ Skipping {fname} — NRAS chain A not found.")
            continue
        print(f"✓ Selected NRAS chain A: {pred.numAtoms()} atoms")

        _, a_gdp, b_gdp, _, _ = prody.matchAlign(pred, ref_gdp)
        rmsd_gdp = prody.calcRMSD(a_gdp, b_gdp)

        _, a_gtp, b_gtp, _, _ = prody.matchAlign(pred, ref_gtp)
        rmsd_gtp = prody.calcRMSD(a_gtp, b_gtp)

        print(f"✓ Calculated RMSDs: GDP={rmsd_gdp:.2f}, GTP={rmsd_gtp:.2f}")
        data.append({
            "system": system,
            "mut": mut,
            "ligand": ligand,
            "rmsd_gdp": rmsd_gdp,
            "rmsd_gtp": rmsd_gtp
        })

    except Exception as e:
        print(f"❌ Failed processing {fname}: {e}")
        continue

print(f"\nTotal data points collected: {len(data)}")
if len(data) == 0:
    print("No data points collected! Cannot create plot.")
    exit(1)

# Plotting
fig, ax = plt.subplots(figsize=(6.5, 6))

for point in data:
    color = system_colors.get((point["system"], point["mut"]), "gray")
    marker = ligand_markers.get(point["ligand"], "x")
    print(f"Plotting point: system={point['system']}, mut={point['mut']}, ligand={point['ligand']}")
    print(f"RMSD values: GDP={point['rmsd_gdp']:.2f}, GTP={point['rmsd_gtp']:.2f}")
    ax.scatter(point["rmsd_gdp"], point["rmsd_gtp"],
               color=color, marker=marker, edgecolor="k", s=100)

# Legend setup
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

lig_legend = [
    Line2D([0], [0], marker='^', color='w', label='GDP', markerfacecolor='gray', markeredgecolor='k', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='GTP', markerfacecolor='gray', markeredgecolor='k', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='unbound', markerfacecolor='gray', markeredgecolor='k', markersize=10)
]

sys_legend = [mpatches.Patch(color=c, label=f"{sys} {mut}") for (sys, mut), c in system_colors.items()]

ax.legend(handles=sys_legend + lig_legend, loc='upper left')

# Axes and formatting
ax.set_xlabel("RMSD to GDP Bound (Å)")
ax.set_ylabel("RMSD to GTP Bound (Å)")
ax.set_xlim(0, 10)  # Adjusted range to ensure we see all points
ax.set_ylim(0, 10)  # Adjusted range to ensure we see all points
plt.title("RMSD Comparison to GDP and GTP Reference Structures")
plt.tight_layout()

# Save figure
os.makedirs("plots_rmsd", exist_ok=True)
plt.savefig("plots_rmsd/rmsd_scatter2.png", dpi=300)
print("\nPlot saved as plots_rmsd/rmsd_scatter2.png") 