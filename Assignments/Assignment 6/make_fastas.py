import os

# Define NRAS sequences
nras_wt = """MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG
QEEYSAMRDQYMRTGEGFLCVFAINNSKSFADINLYREQIKRVKDSDDVPMVLVGNKCDL
PTRTVDTKQAHELAKSYGIPFIETSAKTRQGVEDAFYTLVREIRQYRMKKLN""".replace("\n", "")

nras_q61r = """MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG
REEYSAMRDQYMRTGEGFLCVFAINNSKSFADINLYREQIKRVKDSDDVPMVLVGNKCDL
PTRTVDTKQAHELAKSYGIPFIETSAKTRQGVEDAFYTLVREIRQYRMKKLN""".replace("\n", "")

# Define other proteins
braf = """MAALSGGGGGGAEPGQALFNGDMEPEAGAGAGAAASSAADPAIPEEVWNIKQMIKLTQEH
IEALLDKFGGEHNPPSIYLEAYEEYTSKLDALQQREQQLLESLGNGTDFSVSSSASMDTV
TSSSSSSLSVLPSSLSVFQNPTDVARSNPKSPQKPIVRVFLPNKQRTVVPARCGVTVRDS
LKKALMMRGLIPECCAVYRIQDGEKKPIGWDTDISWLTGEELHVEVLENVPLTTHNFVRK
TFFTLAFCDFCRKLLFQGFRCQTCGYKFHQRCSTEVPLMCVNYDQLDLLFVSKFFEHHPI
PQEEASLAETALTSGSSPSAPASDSIGPQILTSPSPSKSIPIPQPFRPADEDHRNQFGQR
DRSSSAPNVHINTIEPVNIDDLIRDQGFRGDGGSTTGLSATPPASLPGSLTNVKALQKSP
GPQRERKSSSSSEDRNRMKTLGRRDSSDDWEIPDGQITVGQRIGSGSFGTVYKGKWHGDV
AVKMLNVTAPTPQQLQAFKNEVGVLRKTRHVNILLFMGYSTKPQLAIVTQWCEGSSLYHH
LHIIETKFEMIKLIDIARQTAQGMDYLHAKSIIHRDLKSNNIFLHEDLTVKIGDFGLATV
KSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNIN
NRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQILASIELLARS
LPKIHRSASEPSLNRAGFQTEDFSLYACASPKTPIQAGGYGAFPVH""".replace("\n", "")

sos1 = """MQAQQLPYEFFSEENAPKWRGLLVPALKKVQGQVHPTLESNDDALQYVEELILQLLNMLC
QAQPRSASDVEERVQKSFPHPIDKWAIADAQSAIEKRKRRNPLSLPVEKIHPLLKEVLGY
KIDHQVSVYIVAVLEYISADILKLVGNYVRNIRHYEITKQDIKVAMCADKVLMDMFHQDV
EDINILSLTDEEPSTSGEQTYYDLVKAFMAEIRQYIRELNLIIKVFREPFVSNSKLFSAN
DVENIFSRIVDIHELSVKLLGHIEDTVEMTDEGSPHPLVGSCFEDLAEELAFDPYESYAR
DILRPGFHDRFLSQLSKPGAALYLQSIGEGFKEAVQYVLPRLLLAPVYHCLHYFELLKQL
EEKSEDQEDKECLKQAITALLNVQSGMEKICSKSLAKRRLSESACRFYSQQMKGKQLAIK
KMNEIQKNIDGWEGKDIGQCCNEFIMEGTLTRVGAKHERHIFLFDGLMICCKSNHGQPRL
PGASNAEYRLKEKFFMRKVQINDKDDTNEYKHAFEIILKDENSVIFSAKSAEEKNNWMAA
LISLQYRSTLERMLDVTMLQEEKEEQMRLPSADVYRFAEPDSEENIIFEENMQPKAGIPI
IKAGTVIKLIERLTYHMYADPNFVRTFLTTYRSFCKPQELLSLIIERFEIPEPEPTEADR
IAIENGDQPLSAELKRFRKEYIQPVQLRVLNVCRHWVEHHFYDFERDAYLLQRMEEFIGT
VRGKAMKKWVESITKIIQRKKIARDNGPGHNITFQSSPPTVEWHISRPGHIETFDLLTLH
PIEIARQLTLLESDLYRAVQPSELVGSVWTKEDKEINSPNLLKMIRHTTNLTLWFEKCIV
ETENLEERVAVVSRIIEILQVFQELNNFNGVLEVVSAMNSSPVYRLDHTFEQIPSRQKKI
LEEAHELSEDHYKKYLAKLRSINPPCVPFFGIYLTNILKTEEGNPEVLKRHGKELINFSK
RRKVAEITGEIQQYQNQPYCLRVESDIKRFFENLNPMGNSMEKEFTDYLFNKSLEIEPRN
PKPLPRFPKKYSYPLKSPGVRPSNPRPGTMRHPTPLQQEPRKISYSRIPESETESTASAP
NSPRTPLTPPPASGASSTTDVCSVFDSDHSSPFHSSNDTVFIQVTLPHGPRSASVSSISL
TKGTDEVPVPPPVPPRRRPESAPAESSPSKIMSKHLDSPPAIPPRQPTSKAYSPRYSISD
RTSISDPPESPPLLPPREPVRTPDVFSSSPLHLQPPPLGKKSDHGNAFFPNSPSPFTPPP
PQTPSPHGTRRHLPSPPLTQEVDLHSIAGPPVPPRQSTSQHIPKLPPKTYKREHTHPSMH
RDGPPLLENAHSS""".replace("\n", "")

GDP = """[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1c2nc([nH]c(=O)c2nc1)N)([O-])OP(=O)([O-])[O-]"""
GTP = """[P@@](=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1c2nc([nH]c(=O)c2nc1)N)([O-])O[P@@](=O)([O-])OP(=O)([O-])[O-]"""

# Define parameters
nras_types = {"NRAS_WT": nras_wt, "NRAS_Q61R": nras_q61r}
complexes = {
    "NRAS": [],
    "NRAS_SOS1": [sos1],
    "NRAS_BRAF": [braf]
}
ligands = {"no_ligand": None, "GDP": GDP, "GTP": GTP}

# Create output directory
output_dir = "fasta_files"
os.makedirs(output_dir, exist_ok=True)

# Generate FASTA files
for nras_name, nras_seq in nras_types.items():
    for complex_name, proteins in complexes.items():
        for ligand_name, ligand_smiles in ligands.items():
            # Define file name
            file_name = f"{nras_name}_{complex_name}_{ligand_name}.fasta"
            file_path = os.path.join(output_dir, file_name)

            # Open file for writing
            with open(file_path, "w") as fasta_file:
                # Write NRAS sequence (always Chain A)
                fasta_file.write(f">A|protein|empty\n{nras_seq}\n")

                # Write complex proteins if present
                for idx, prot_seq in enumerate(proteins, start=1):
                    chain_id = chr(65 + idx)  # B, C, D, etc.
                    fasta_file.write(f">{chain_id}|protein|empty\n{prot_seq}\n")

                # Write ligand if applicable
                if ligand_smiles:
                    chain_id = chr(65 + len(proteins) + 1)  # Next available letter (C or D)
                    fasta_file.write(f">{chain_id}|smiles\n{ligand_smiles}\n")

            print(f"Generated: {file_name}")

print(f"All FASTA files saved in '{output_dir}' directory.")
