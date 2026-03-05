#!/usr/bin/env python3
# For double or single chain with heme: check the tleap files: for FE, and SG residues numbers, also insert_TER_after and remove_TER_after functions.
# SDR transfers ligand and DUM2 according to vector not according to distance along Z axis.
# check:     sdr_dx, sdr_dy, sdr_dz = 0.0, -45.0, 0.0  # shift along -Y

import datetime as dt
import glob as glob
import os as os
import re as re
import shutil as shutil
import signal as signal
import subprocess as sp
import sys as sys
from lib import scripts as scripts
from lib import setup as setup
import filecmp


from lib.setup import restraints, sim_files
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix
from collections import defaultdict

#############Additional##############


def pdb_resnums_by_resname(pdb_path: str, resname: str):
    resnums = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            rn = line[17:20].strip()
            if rn != resname:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if not resnums or resnums[-1] != resi:
                resnums.append(resi)
    return resnums


def compute_othermol_resnums(
    prot_end: int, other_mol, num_chains: int, ligand_resname="UNL"
):
    """
    Returns:
      unl_resnum: int
      other_resnums: list of (name, resnum)
      hem_resnums: list of heme residue numbers (for name == "HEM")
    Assumptions: ligand UNL is exactly prot_end+1, and other_mol residues are appended after UNL.
    """
    unl_resnum = prot_end + 1

    # Expand other_mol per chain (chain A, chain B, ...)
    expanded = []
    for _ in range(int(num_chains)):
        expanded.extend(list(other_mol))

    other_resnums = []
    hem_resnums = []
    start = unl_resnum + 1
    for idx, name in enumerate(expanded):
        resnum = start + idx
        other_resnums.append((name, resnum))
        if str(name).upper() == "HEM":
            hem_resnums.append(resnum)

    return unl_resnum, other_resnums, hem_resnums


def write_atom(
    fout, idx, atom_name, resname, resid, coord, chain_id="A", occ=0.00, bfac=0.00
):
    fout.write(
        "{:<6s}{:>5d} {:<4s} {:>3s} {:1s}{:>4d}    "
        "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
            "ATOM",  # Record
            idx,  # Atom serial
            atom_name.ljust(4),  # Atom name left justified in 4 spaces
            resname,  # Residue name right justified in 3 spaces
            chain_id,  # Chain ID (you can pass or keep as "A")
            resid,  # Residue ID
            coord[0],
            coord[1],
            coord[2],  # X, Y, Z
            occ,  # Occupancy
            bfac,  # B-factor
        )
    )


def print_line_count(file_path):
    """
    Print the total number of lines in a file.
    """
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        line_count = sum(1 for _ in f)
    print(f"[INFO] Total lines in {file_path}: {line_count}")


def print_pdb_contents(pdb_path, label=None):
    """
    Print all ATOM and HETATM lines from a PDB file exactly as they appear.
    """
    if label:
        print(f"[INFO] Contents of {label} ({pdb_path}):")
    else:
        print(f"[INFO] Contents of {pdb_path}:")

    with open(pdb_path) as f:
        for line in f:
            # if line.startswith(("ATOM", "HETATM")):
            print(line.strip())


def print_pdb_residues(pdb_path, label=None):
    """
    Print all unique residues in the PDB file with resname and resnum.
    """
    if label:
        print(f"[INFO] Unique residues in {label} ({pdb_path}):")
    else:
        print(f"[INFO] Unique residues in {pdb_path}:")

    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                resnum = int(line[22:26])
                key = (resname, resnum)
                if key not in seen:
                    seen.add(key)
                    print(f"{resname}{resnum}")
    print(f"[INFO] Total residues: {len(seen)}")


def print_residues_from_pdb(pdb_path, label=None):
    """
    Print all residues from a PDB file by parsing ATOM/HETATM lines,
    without stopping at TER or END.
    """
    residues = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                resnum = line[22:26].strip()
                resid = f"{resname}{resnum}"
                if resid not in residues:
                    residues.append(resid)

    if label:
        print(f"[INFO] Residues in {label} ({pdb_path}):")
    else:
        print(f"[INFO] Residues in {pdb_path}:")

    for i, resid in enumerate(residues):
        print(f"{i+1:3}: {resid}")

    print(f"[INFO] Total residues: {len(residues)}")


def count_leading_resname(pdb_path: str, resname: str) -> int:
    """
    Count how many residues with given resname appear at the beginning of the PDB
    before the first non-matching residue.
    Assumes residues are ordered by residue number.
    """
    seen = set()
    last_resi = None
    count = 0

    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            rn = line[17:20].strip()
            try:
                resi = int(line[22:26])
            except ValueError:
                continue

            # New residue?
            if last_resi is None or resi != last_resi:
                last_resi = resi

                if rn == resname and resi not in seen:
                    seen.add(resi)
                    count += 1
                    continue

                # first non-matching residue -> stop
                break

    return count


def insert_ter_from_reference(rec_pdb_path, build_pdb_path, output_pdb_path):
    """
    Insert TER lines into build.pdb based on TER positions from rec_file.pdb.
    Assumes both PDBs are cleaned and sequential (no chain IDs), with residue numbers matching.
    """

    def extract_residues(pdb_path):
        residues = []
        seen = set()
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    resname = line[17:20].strip()
                    resnum = int(line[22:26].strip())
                    key = (resnum, resname)
                    if key not in seen:
                        residues.append(f"{resname}{resnum}")
                        seen.add(key)
        return residues

    # Step 1: Parse reference TER lines and extract TER positions
    ter_resnums = set()
    prev_resnum = None
    with open(rec_pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                prev_resnum = int(line[22:26].strip())
            elif line.startswith("TER") and prev_resnum is not None:
                ter_resnums.add(
                    prev_resnum + 1
                )  # Shift because TER follows the last atom of previous residue

    # Step 2: Count residues
    rec_residues = extract_residues(rec_pdb_path)
    build_residues = extract_residues(build_pdb_path)

    print("[INFO] Inserting TER lines based on rec_file.pdb")
    print(f"[DEBUG] Total residues in rec_pdb: {len(rec_residues)}")
    print(f"[DEBUG] Total residues in build_pdb: {len(build_residues)}")
    print(f"[DEBUG] First 5 residues in rec_pdb: {rec_residues[:5]}")
    print(f"[DEBUG] Last 5 residues in rec_pdb: {rec_residues[-5:]}")
    print(f"[DEBUG] First 5 residues in build_pdb: {build_residues[:5]}")
    print(f"[DEBUG] Last 5 residues in build_pdb: {build_residues[-5:]}")
    print(f"[DEBUG] Shifted TER positions from rec_pdb: {sorted(ter_resnums)}")

    # Step 3: Copy and insert TER
    output_lines = []
    with open(build_pdb_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        output_lines.append(line)
        if line.startswith(("ATOM", "HETATM")):
            try:
                curr_resnum = int(line[22:26].strip())
                curr_resname = line[17:20].strip()
            except ValueError:
                continue

            # Determine end of residue
            is_last_atom_of_res = False
            if i + 1 == len(lines):
                is_last_atom_of_res = True
            elif lines[i + 1].startswith(("ATOM", "HETATM")):
                next_resnum = int(lines[i + 1][22:26].strip())
                if next_resnum != curr_resnum:
                    is_last_atom_of_res = True
            else:
                is_last_atom_of_res = True

            if is_last_atom_of_res and curr_resnum in ter_resnums:
                output_lines.append("TER\n")
                print(
                    f"[DEBUG] Inserting TER after residue {curr_resname}{curr_resnum}"
                )

    if not output_lines[-1].strip() == "END":
        output_lines.append("END\n")

    with open(output_pdb_path, "w") as out:
        out.writelines(output_lines)

    print(f"[INFO] Final output written to {output_pdb_path}")
    print(f"[DEBUG] Total lines written: {len(output_lines)}")


def remove_ter_after_resnums(pdb_path="full.pdb", resnums=(119,)):
    """Remove TER records that appear right after any residue number in resnums."""
    resnums = set(int(r) for r in resnums)
    lines = open(pdb_path).read().splitlines()
    out = []
    last_resseq = None

    for l in lines:
        rec = l[:6].strip()
        if rec in ("ATOM", "HETATM"):
            # PDB resSeq is columns 23-26 (0-based [22:26])
            try:
                last_resseq = int(l[22:26])
            except ValueError:
                last_resseq = None
            out.append(l)
        elif rec == "TER":
            # Skip TER if it follows a target residue number
            if last_resseq in resnums:
                continue
            out.append(l)
        else:
            out.append(l)

    with open(pdb_path, "w") as f:
        f.write("\n".join(out) + "\n")


def remove_ter_after_resnum(pdb_path="full.pdb", resnum=119):
    """Convenience wrapper for a single residue number."""
    remove_ter_after_resnums(pdb_path, (resnum,))


def insert_ter_after_resnums(
    pdb_path="full.pdb", resnums=(119,), chains=None, output_path=None
):
    """
    Insert a TER record immediately after any residue whose PDB resSeq is in `resnums`.
    - If `chains` is None: match ALL chains that have those resSeqs.
    - If `chains` is a set/list/tuple of chain IDs: only insert for those chains.
    - If a TER is already present after that residue, no duplicate is added.
    - Writes in-place unless `output_path` is provided.
    """
    resnums = set(int(r) for r in resnums)
    chains = None if chains is None else set(chains)

    lines = open(pdb_path).read().splitlines()
    out = []

    def rec_tag(line):
        return line[:6].strip()

    def parse_chain(line):
        # PDB chain ID = col 22 (0-based index 21). May be blank.
        return (line[21] if len(line) > 21 else " ").strip() or " "

    def parse_resseq(line):
        # PDB resSeq = cols 23-26 (0-based [22:26])
        try:
            return int(line[22:26])
        except Exception:
            return None

    n = len(lines)
    i = 0
    while i < n:
        l = lines[i]
        out.append(l)
        tag = rec_tag(l)

        if tag in ("ATOM", "HETATM"):
            chain_i = parse_chain(l)
            res_i = parse_resseq(l)

            # determine if this is the last atom of the residue
            # look ahead to see if next line is a different residue or non-ATOM/HETATM
            last_of_res = False
            if i + 1 >= n:
                last_of_res = True
            else:
                ln = lines[i + 1]
                t2 = rec_tag(ln)
                if t2 not in ("ATOM", "HETATM"):
                    last_of_res = True
                else:
                    chain_n = parse_chain(ln)
                    res_n = parse_resseq(ln)
                    if (chain_n != chain_i) or (res_n != res_i):
                        last_of_res = True

            if (
                last_of_res
                and (res_i in resnums)
                and (chains is None or chain_i in chains)
            ):
                # Avoid duplicating TER if it's already there
                next_is_TER = i + 1 < n and rec_tag(lines[i + 1]) == "TER"
                if not next_is_TER:
                    out.append("TER")

        i += 1

    # Preserve END if present (and don’t add an extra)
    if out and out[-1].strip() != "END":
        out.append("END")

    dst = output_path or pdb_path
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


def insert_ter_after_resnum(
    pdb_path="full.pdb", resnum=119, chains=None, output_path=None
):
    """
    Convenience wrapper for a single residue number (optionally restricted to certain chains).
    """
    insert_ter_after_resnums(
        pdb_path=pdb_path, resnums=(resnum,), chains=chains, output_path=output_path
    )


def renumber_pdb_residues(
    pdb_in: str,
    pdb_out: str,
    start: int = 1,
    per_chain: bool = False,
    reset_on_TER: bool = False,
) -> None:
    """
    Renumber residues sequentially according to their appearance in the file,
    without changing ordering or chains.

    - If per_chain=False (default): one global counter  -> 1,2,3,... across the whole file.
    - If per_chain=True: independent counter per chain (chain ID = column 22).
    - If reset_on_TER=True: counters reset after a TER record (global or per-chain, as applicable).

    Only the PDB resSeq field (columns 23–26) is modified. Everything else is preserved.
    """

    def is_atom(line: str) -> bool:
        tag = line[:6].strip()
        return tag == "ATOM" or tag == "HETATM"

    def chain_id(line: str) -> str:
        return (line[21] if len(line) > 21 else " ").strip() or " "

    def residue_key(line: str):
        # Distinguish residues by (chain, resSeq, resName, insCode)
        resn = line[17:20].strip()
        resi = line[22:26]  # keep raw slice for boundary detection
        icode = line[26] if len(line) > 26 else " "
        return (chain_id(line), resn, resi, icode)

    with open(pdb_in) as f:
        lines = f.readlines()

    out = []

    # State for residue change detection
    prev_key = None

    # Counters
    if per_chain:
        counters = {}  # chain -> current number

        def bump(chain):
            counters[chain] = counters.get(chain, start - 1) + 1
            return counters[chain]

        def reset(chain=None):
            if chain is None:
                counters.clear()
            else:
                counters.pop(chain, None)

    else:
        counter = start - 1

        def bump(_chain=None):
            nonlocal counter
            counter += 1
            return counter

        def reset(_chain=None):
            nonlocal counter
            counter = start - 1

    # Map current residue (chain) -> assigned new number so all atoms of the residue get same value
    current_assigned = None  # assigned new residue number

    for i, line in enumerate(lines):
        tag = line[:6].strip()

        if is_atom(line):
            key = residue_key(line)  # (chain,resn,oldresi,icode)

            # new residue?
            if key != prev_key:
                ch = key[0]
                newnum = bump(ch if per_chain else None)
                current_assigned = newnum
                prev_key = key
            else:
                # same residue as previous line → reuse assigned number
                newnum = current_assigned

            # write new resSeq (cols 22:26), right-justified width 4
            new_line = line[:22] + f"{newnum:>4d}" + line[26:]
            out.append(new_line)

        elif tag == "TER":
            out.append(line)
            if reset_on_TER:
                if per_chain:
                    # reset only the chain that just ended (best-effort via prev_key)
                    if prev_key is not None:
                        reset(prev_key[0])
                else:
                    reset()
            # after TER, next ATOM/HETATM will be a new residue
            prev_key = None
            current_assigned = None

        else:
            out.append(line)
            # leave prev_key as-is for non-atom, non-TER lines

    # Ensure single END at file end
    if not out or out[-1].strip() != "END":
        out.append("END\n")

    with open(pdb_out, "w") as fo:
        fo.writelines(out)


def count_unl_residues(pdb_path: str, residue_name: str = "UNL") -> int:
    """
    Count unique residues in a PDB file matching a given residue name (default 'UNL').
    It looks at residue name (columns 17–20) and resSeq (columns 22–26).
    """
    unl_residues = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resn = line[17:20].strip()
                if resn == residue_name:
                    try:
                        resi = int(line[22:26].strip())
                        unl_residues.add(resi)
                    except ValueError:
                        continue
    return len(unl_residues)


def count_nonprotein_residues(
    pdb_path: str,
    exclude_resnames=("HOH", "WAT", "DUM"),
) -> dict:
    """
    Return a dict {resname: count} for non-protein residues.
    """
    residues = {}

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resn = line[17:20].strip().upper()
                if resn in exclude_resnames:
                    continue
                try:
                    resi = int(line[22:26].strip())
                except ValueError:
                    continue
                residues.setdefault(resn, set()).add(resi)

    return {k: len(v) for k, v in residues.items()}


###Not Necessary For Bonded Model  . For non-Bonded model##
def write_fe_heme_restraints(
    pdb_path="aligned_amber.pdb", out_path="fe_heme.rest", k=100.0, r0=2.0
):
    """
    Automatically write distance restraints between Fe (from FE3) and 4 N* atoms from HEM.
    Writes AMBER-style NMR restraints to a separate file (e.g., fe_heme.rest).

    Parameters:
        pdb_path (str): Path to the input PDB file (default: "aligned_amber.pdb")
        out_path (str): Path to the output restraint file (default: "fe_heme.rest")
        k (float): Force constant for the restraint in kcal/mol·Å² (default: 100.0)
        r0 (float): Target distance in Å between Fe and N (default: 2.0)
    """
    import os

    try:
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(
                f"❌ Input PDB file not found: {os.path.abspath(pdb_path)}"
            )

        fe_atom = None
        n_atoms = []

        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue

                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                atom_id = int(line[6:11].strip())

                if res_name == "FE3" and atom_name.upper() in ("FE", "FE3"):
                    fe_atom = atom_id

                if res_name == "HEM" and atom_name in ("NA", "NB", "NC", "ND"):
                    n_atoms.append(atom_id)

        if not fe_atom:
            raise ValueError("❌ Could not find Fe atom from FE3 in the PDB.")
        if len(n_atoms) != 4:
            raise ValueError(
                f"❌ Expected 4 coordinating N atoms from HEM, found {len(n_atoms)}."
            )

        r1 = max(0.0, r0 - 0.2)
        r2 = r0
        r3 = r0 + 0.2
        r4 = r0 + 0.4

        out_path_abs = os.path.abspath(out_path)
        with open(out_path, "a") as f:
            f.write("\n# === Fe–N Heme Restraints ===\n")
            for n_atom in n_atoms:
                f.write(
                    f"&rst iat={fe_atom},{n_atom}, "
                    f"r1={r1:.2f}, r2={r2:.2f}, r3={r3:.2f}, r4={r4:.2f}, "
                    f"rk2={k:.1f}, rk3={k:.1f} /\n"
                )

        if os.path.exists(out_path):
            print(f"✅ Added Fe–N restraints to {out_path_abs}")
        else:
            raise IOError(
                f"❌ Writing finished but output file not found: {out_path_abs}"
            )

    except Exception as e:
        print(f"🚨 Error: {e}")


def append_fe_heme_restraints_to_disang(
    disang_file="disang.rest", fe_file="fe_heme.rest"
):
    import os

    if not os.path.exists(fe_file):
        print(f"⚠️ Restraint file {fe_file} not found.")
        return

    if not os.path.exists(disang_file):
        print(f"📁 disang.rest not found. Creating a new one.")
        open(disang_file, "w").close()

    with open(disang_file, "a") as disang, open(fe_file, "r") as fe_rest:
        fe_content = fe_rest.read().strip()
        if fe_content:
            disang.write("\n# === Appended Fe–Heme Restraints ===\n")
            disang.write(fe_content + "\n")
            print(f"📎 Successfully appended {fe_file} to {disang_file}")
        else:
            print(f"⚠️ {fe_file} is empty. Nothing appended.")


###Not Necessary For Bonded Model Only for non-bonded model##


def build_equil_heme(
    pose,
    celp_st,
    mol,
    H1,
    H2,
    H3,
    calc_type,
    l1_x,
    l1_y,
    l1_z,
    l1_range,
    min_adis,
    max_adis,
    ligand_ff,
    ligand_ph,
    retain_lig_prot,
    ligand_charge,
    other_mol,
    solv_shell,
    first_cyp_equil=None,
    second_cyp_equil=None,
    first_cyp_next_equil=None,
    second_cyp_next_equil=None,
    first_cyp_previous_equil=None,
    second_cyp_previous_equil=None,
    heme_1=None,
    heme_2=None,
):

    # Not apply SDR distance when equilibrating
    sdr_dist = 0

    # Create equilibrium directory
    if not os.path.exists("equil"):
        os.makedirs("equil")
    os.chdir("equil")
    if os.path.exists("./build_files"):
        shutil.rmtree("./build_files")
    try:
        shutil.copytree("../build_files", "./build_files")
    # Directories are the same
    except shutil.Error as e:
        print("Directory not copied. Error: %s" % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print("Directory not copied. Error: %s" % e)
    os.chdir("build_files")

    if calc_type == "dock":
        shutil.copy("../../all-poses/%s_docked.pdb" % (celp_st), "./rec_file.pdb")
        shutil.copy("../../all-poses/%s.pdb" % (pose), "./")
    elif calc_type == "rank":
        shutil.copy("../../all-poses/%s.pdb" % (celp_st), "./rec_file.pdb")
        shutil.copy("../../all-poses/%s.pdb" % (pose), "./")
    elif calc_type == "crystal":
        shutil.copy("../../all-poses/%s.pdb" % (pose), "./")
        # Replace names and run initial VMD script
        with open("prep-crystal.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(
                        line.replace("MMM", mol)
                        .replace("mmm", mol.lower())
                        .replace("CCCC", pose)
                    )
        sp.call("vmd -dispdev text -e prep.tcl", shell=True)

    # Split initial receptor file
    with open("split-ini.tcl", "rt") as fin:
        with open("split.tcl", "wt") as fout:
            if other_mol:
                other_mol_vmd = " ".join(other_mol)
            else:
                other_mol_vmd = "XXX"
            for line in fin:
                if "lig" not in line:
                    fout.write(
                        line.replace("SHLL", "%4.2f" % solv_shell)
                        .replace("OTHRS", str(other_mol_vmd))
                        .replace("MMM", mol.upper())
                    )
    sp.call("vmd -dispdev text -e split.tcl", shell=True)

    # Remove possible remaining molecules
    if not other_mol:
        open("others.pdb", "w").close()

    shutil.copy("./protein.pdb", "./protein_vmd.pdb")
    sp.call("pdb4amber -i protein_vmd.pdb -o protein.pdb -y", shell=True)
    # sp.call("cp protein_vmd.pdb  protein.pdb", shell=True)

    # Get beginning and end of protein and save first residue as global variable
    with open("./protein_vmd.pdb") as myfile:
        data = myfile.readlines()
        first_res = int(data[1][22:26].strip())
    with open("./protein.pdb") as myfile:
        data = myfile.readlines()
        recep_resid_num = int(data[-2][22:26].strip())
    print("Receptor first residue: %s" % first_res)
    print("Receptor total length: %s" % recep_resid_num)

    # Adjust protein anchors to the new residue numbering
    h1_resid = H1.split("@")[0][1:]
    h2_resid = H2.split("@")[0][1:]
    h3_resid = H3.split("@")[0][1:]

    h1_atom = H1.split("@")[1]
    h2_atom = H2.split("@")[1]
    h3_atom = H3.split("@")[1]

    p1_resid = str(int(h1_resid) - int(first_res) + 2)
    p2_resid = str(int(h2_resid) - int(first_res) + 2)
    p3_resid = str(int(h3_resid) - int(first_res) + 2)

    p1_vmd = str(int(h1_resid) - int(first_res) + 1)

    P1 = ":" + p1_resid + "@" + h1_atom
    P2 = ":" + p2_resid + "@" + h2_atom
    P3 = ":" + p3_resid + "@" + h3_atom

    print("Receptor anchors:")
    print(P1)
    print(P2)
    print(P3)

    # Replace names in initial files and VMD scripts
    with open("prep-ini.tcl", "rt") as fin:
        with open("prep.tcl", "wt") as fout:
            other_mol_vmd = " ".join(other_mol)
            for line in fin:
                fout.write(
                    line.replace("MMM", mol)
                    .replace("mmm", mol.lower())
                    .replace("NN", h1_atom)
                    .replace("P1A", p1_vmd)
                    .replace("FIRST", "1")
                    .replace("LAST", str(recep_resid_num))
                    .replace("STAGE", "equil")
                    .replace("XDIS", "%4.2f" % l1_x)
                    .replace("YDIS", "%4.2f" % l1_y)
                    .replace("ZDIS", "%4.2f" % l1_z)
                    .replace("RANG", "%4.2f" % l1_range)
                    .replace("DMAX", "%4.2f" % max_adis)
                    .replace("DMIN", "%4.2f" % min_adis)
                    .replace("SDRD", "%4.2f" % sdr_dist)
                    .replace("OTHRS", str(other_mol_vmd))
                )

    # Save parameters in ff folder
    if not os.path.exists("../ff/"):
        os.makedirs("../ff/")
    for file in glob.glob("./*.mol2"):
        shutil.copy(file, "../ff/")
    for file in glob.glob("./*.frcmod"):
        shutil.copy(file, "../ff/")
    shutil.copy("./dum.mol2", "../ff/")
    shutil.copy("./dum.frcmod", "../ff/")

    # Adjust ligand files
    # Mudong's mod: optionally retain the ligand protonation state as provided in pose*.pdb, and skip Babel processing (removing H, adding H, determining total charge)
    if retain_lig_prot == "yes":
        # Determine ligand net charge by reading the rightmost column of pose*.pdb, programs such as Maestro writes atom charges there
        if ligand_charge == "nd":
            ligand_charge = 0
            with open("" + pose + ".pdb") as f_in:
                for line in f_in:
                    if "1+" in line:
                        ligand_charge += 1
                    elif "2+" in line:
                        ligand_charge += 2
                    elif "3+" in line:
                        ligand_charge += 3
                    elif "4+" in line:
                        ligand_charge += 4
                    elif "1-" in line:
                        ligand_charge += -1
                    elif "2-" in line:
                        ligand_charge += -2
                    elif "3-" in line:
                        ligand_charge += -3
                    elif "4-" in line:
                        ligand_charge += -4
        print("The net charge of the ligand is %d" % ligand_charge)
        if calc_type == "dock" or calc_type == "rank":
            shutil.copy("./" + pose + ".pdb", "./" + mol.lower() + "-h.pdb")
        elif calc_type == "crystal":
            shutil.copy("./" + mol.lower() + ".pdb", "./" + mol.lower() + "-h.pdb")
    else:
        if calc_type == "dock" or calc_type == "rank":
            sp.call(
                "obabel -i pdb " + pose + ".pdb -o pdb -O " + mol.lower() + ".pdb -d",
                shell=True,
            )  # Remove all hydrogens from the ligand
        elif calc_type == "crystal":
            sp.call(
                "obabel -i pdb "
                + mol.lower()
                + ".pdb -o pdb -O "
                + mol.lower()
                + ".pdb -d",
                shell=True,
            )  # Remove all hydrogens from crystal ligand
        sp.call(
            "obabel -i pdb "
            + mol.lower()
            + ".pdb -o pdb -O "
            + mol.lower()
            + "-h-ini.pdb -p %4.2f" % ligand_ph,
            shell=True,
        )  # Put all hydrogens back using babel
        sp.call(
            "obabel -i pdb "
            + mol.lower()
            + ".pdb -o mol2 -O "
            + mol.lower()
            + "-crg.mol2 -p %4.2f" % ligand_ph,
            shell=True,
        )
        # Clean ligand protonated pdb file
        with open(mol.lower() + "-h-ini.pdb") as oldfile, open(
            mol.lower() + "-h.pdb", "w"
        ) as newfile:
            for line in oldfile:
                if "ATOM" in line or "HETATM" in line:
                    newfile.write(line)
            newfile.close()
        if ligand_charge == "nd":
            ligand_charge = 0
            # Get ligand net charge from babel
            lig_crg = 0
            with open("%s-crg.mol2" % mol.lower()) as f_in:
                for line in f_in:
                    splitdata = line.split()
                    if len(splitdata) > 8:
                        lig_crg = lig_crg + float(splitdata[8].strip())
            print(lig_crg)
            ligand_charge = round(lig_crg)
        print("The babel protonation of the ligand is for pH %4.2f" % ligand_ph)
        print("The net charge of the ligand is %d" % ligand_charge)

    # Get ligand parameters
    if not os.path.exists("../ff/%s.mol2" % mol.lower()):
        print(
            "Antechamber parameters command: antechamber -i "
            + mol.lower()
            + "-h.pdb -fi pdb -o "
            + mol.lower()
            + ".mol2 -fo mol2 -c bcc -s 2 -at "
            + ligand_ff.lower()
            + " -nc %d" % ligand_charge
        )
        sp.call(
            "antechamber -i "
            + mol.lower()
            + "-h.pdb -fi pdb -o "
            + mol.lower()
            + ".mol2 -fo mol2 -c bcc -s 2 -at "
            + ligand_ff.lower()
            + " -nc %d" % ligand_charge,
            shell=True,
        )
        shutil.copy("./%s.mol2" % (mol.lower()), "../ff/")
    if not os.path.exists("../ff/%s.frcmod" % mol.lower()):
        if ligand_ff == "gaff":
            sp.call(
                "parmchk2 -i "
                + mol.lower()
                + ".mol2 -f mol2 -o "
                + mol.lower()
                + ".frcmod -s 1",
                shell=True,
            )
        elif ligand_ff == "gaff2":
            sp.call(
                "parmchk2 -i "
                + mol.lower()
                + ".mol2 -f mol2 -o "
                + mol.lower()
                + ".frcmod -s 2",
                shell=True,
            )
        shutil.copy("./%s.frcmod" % (mol.lower()), "../ff/")
    sp.call(
        "antechamber -i "
        + mol.lower()
        + "-h.pdb -fi pdb -o "
        + mol.lower()
        + ".pdb -fo pdb",
        shell=True,
    )

    # Create raw complex and clean it
    filenames = ["protein.pdb", "%s.pdb" % mol.lower(), "others.pdb", "crystalwat.pdb"]
    with open("./complex-merge.pdb", "w") as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    with open("complex-merge.pdb") as oldfile, open("complex.pdb", "w") as newfile:
        for line in oldfile:
            if not "CRYST1" in line and not "CONECT" in line and not "END" in line:
                newfile.write(line)

    # New work around to avoid chain swapping during alignment
    sp.call("pdb4amber -i reference.pdb -o reference_amber.pdb -y", shell=True)
    # sp.call("cp reference.pdb  reference_amber.pdb", shell=True)
    sp.call("vmd -dispdev text -e nochain.tcl", shell=True)
    sp.call(
        "./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc",
        shell=True,
    )
    sp.call("vmd -dispdev text -e measure-fit.tcl", shell=True)

    # Put in AMBER format and find ligand anchor atoms
    with open("aligned.pdb", "r") as oldfile, open("aligned-clean.pdb", "w") as newfile:
        for line in oldfile:
            splitdata = line.split()
            if len(splitdata) > 4:
                newfile.write(line)
    sp.call(
        "pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y",
        shell=True,
    )
    # sp.call("cp aligned-clean.pdb aligned_amber.pdb", shell=True)
    sp.call("vmd -dispdev text -e prep.tcl", shell=True)

    # Check size of anchor file
    anchor_file = "anchors.txt"
    if os.stat(anchor_file).st_size == 0:
        os.chdir("../")
        return "anch1"
    f = open(anchor_file, "r")
    for line in f:
        splitdata = line.split()
        if len(splitdata) < 3:
            os.rename("./anchors.txt", "anchors-" + pose + ".txt")
            os.chdir("../")
            return "anch2"
    os.rename("./anchors.txt", "anchors-" + pose + ".txt")

    os.chdir("../")

    # Create simulation directory
    if not os.path.exists(pose):
        os.makedirs(pose)
    os.chdir(pose)

    #####################Inside simulation Directory POSE#################

    dum_coords = []
    recep_coords = []
    lig_coords = []
    oth_coords = []
    dum_atomlist = []
    lig_atomlist = []
    recep_atomlist = []
    oth_atomlist = []
    dum_rsnmlist = []
    recep_rsnmlist = []
    lig_rsnmlist = []
    oth_rsnmlist = []
    dum_rsidlist = []
    recep_rsidlist = []
    lig_rsidlist = []
    oth_rsidlist = []
    dum_chainlist = []
    recep_chainlist = []
    lig_chainlist = []
    oth_chainlist = []
    dum_atom = 0
    lig_atom = 0
    recep_atom = 0
    oth_atom = 0
    total_atom = 0
    resid_lig = 0
    resname_lig = mol

    # Copy a few files
    shutil.copy("../build_files/equil-%s.pdb" % mol.lower(), "./")
    shutil.copy("../build_files/%s-noh.pdb" % mol.lower(), "./%s.pdb" % mol.lower())
    shutil.copy("../build_files/anchors-" + pose + ".txt", "./anchors.txt")

    # Read coordinates for dummy atoms
    for i in range(1, 2):
        shutil.copy("../build_files/dum" + str(i) + ".pdb", "./")
        with open("dum" + str(i) + ".pdb") as dum_in:
            lines = (line.rstrip() for line in dum_in)
            lines = list(line for line in lines if line)
            dum_coords.append(
                (
                    float(lines[1][30:38].strip()),
                    float(lines[1][38:46].strip()),
                    float(lines[1][46:54].strip()),
                )
            )
            dum_atomlist.append(lines[1][12:16].strip())
            dum_rsnmlist.append(lines[1][17:20].strip())
            dum_rsidlist.append(float(lines[1][22:26].strip()))
            dum_chainlist.append(lines[1][21].strip())
            dum_atom += 1
            total_atom += 1

    # Read coordinates from aligned system
    with open("equil-%s.pdb" % mol.lower()) as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = list(line for line in lines if line)  # Non-blank lines in a list

    # Count atoms of receptor and ligand
    for i in range(0, len(lines)):
        if (lines[i][0:6].strip() == "ATOM") or (lines[i][0:6].strip() == "HETATM"):
            if (
                (lines[i][17:20].strip() != mol)
                and (lines[i][17:20].strip() != "DUM")
                and (lines[i][17:20].strip() != "WAT")
                and (lines[i][17:20].strip() not in other_mol)
            ):
                recep_coords.append(
                    (
                        float(lines[i][30:38].strip()),
                        float(lines[i][38:46].strip()),
                        float(lines[i][46:54].strip()),
                    )
                )
                recep_atomlist.append(lines[i][12:16].strip())
                recep_rsnmlist.append(lines[i][17:20].strip())
                recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                recep_chainlist.append(lines[i][21].strip())
                recep_last = int(lines[i][22:26].strip())
                recep_atom += 1
                total_atom += 1
            elif lines[i][17:20].strip() == mol:
                lig_coords.append(
                    (
                        float(lines[i][30:38].strip()),
                        float(lines[i][38:46].strip()),
                        float(lines[i][46:54].strip()),
                    )
                )
                lig_atomlist.append(lines[i][12:16].strip())
                lig_rsnmlist.append(lines[i][17:20].strip())
                lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                lig_chainlist.append(lines[i][21].strip())
                lig_atom += 1
                total_atom += 1
            elif (lines[i][17:20].strip() == "WAT") or (
                lines[i][17:20].strip() in other_mol
            ):
                oth_coords.append(
                    (
                        float(lines[i][30:38].strip()),
                        float(lines[i][38:46].strip()),
                        float(lines[i][46:54].strip()),
                    )
                )
                oth_atomlist.append(lines[i][12:16].strip())
                oth_rsnmlist.append(lines[i][17:20].strip())
                oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                oth_chainlist.append(lines[i][21].strip())
                oth_atom += 1
                total_atom += 1

    coords = dum_coords + recep_coords + lig_coords + oth_coords
    atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
    resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
    resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
    chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
    # lig_resid = str(recep_last + dum_atom + 1) # That did assume lig is last residue before water not valid in case of other_mol
    lig_resid = str(int(lig_rsidlist[0]))  # Get actual ligand resid directly
    chain_tmp = "None"
    resid_tmp = "None"

    # Read ligand anchors obtained from VMD
    anchor_file = "anchors.txt"
    f = open(anchor_file, "r")
    for line in f:
        splitdata = line.split()
        L1 = ":" + lig_resid + "@" + splitdata[0]
        L2 = ":" + lig_resid + "@" + splitdata[1]
        L3 = ":" + lig_resid + "@" + splitdata[2]

    print("Ligand anchors:")
    print(L1)
    print(L2)
    print(L3)
    print(other_mol)

    # -----------------------------------
    # Write the new pdb file
    # -----------------------------------

    build_file = open("build.pdb", "w")

    # Reorganize atoms into groups
    dummy_atoms = []
    receptor_atoms = []
    ligand_atoms = []
    other_atoms = []

    for i in range(len(atom_namelist)):
        atom_data = {
            "idx": i + 1,
            "atom_name": atom_namelist[i],
            "resname": resname_list[i],
            "resid": int(resid_list[i]),
            "coord": coords[i],
            "chain": chain_list[i] if i < len(chain_list) else " ",
        }

        if i < dum_atom:
            dummy_atoms.append(atom_data)
        elif dum_atom <= i < dum_atom + recep_atom:
            receptor_atoms.append(atom_data)
        elif dum_atom + recep_atom <= i < dum_atom + recep_atom + lig_atom:
            ligand_atoms.append(atom_data)
        else:
            other_atoms.append(atom_data)

    # Write dummy atoms
    for atom in dummy_atoms:
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
    build_file.write("TER\n")

    # Write receptor atoms
    prev_chain = ""
    for atom in receptor_atoms:
        if (
            atom["chain"] != prev_chain
            and atom["resname"] not in other_mol
            and atom["resname"] != "WAT"
        ):
            build_file.write("TER\n")
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
        prev_chain = atom["chain"]
    build_file.write("TER\n")

    # Write ligand atoms
    for atom in ligand_atoms:
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
    build_file.write("TER\n")

    # Write other atoms (other_mol + WAT)
    prev_resid = None
    for atom in other_atoms:
        if prev_resid is not None and atom["resid"] != prev_resid:
            build_file.write("TER\n")
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
        prev_resid = atom["resid"]
    build_file.write("TER\n")
    build_file.write("END\n")

    ###Debug: Not closing the file created huge problem##########
    build_file.close()

    print("Number of lines in build.pdb before insertion")
    print_line_count("build.pdb")
    print("Inserting TER as in original pdb")
    insert_ter_from_reference(
        "../build_files/rec_file.pdb", "build.pdb", "build_ter.pdb"
    )
    # Insert TER after the first iNOS chain in build_ter.pdb (and remove any TER that corresponds
    # to the chain boundary coming from the reference numbering)
    if second_cyp_equil is not None and first_cyp_equil is not None:
        prot_len_dec = int(second_cyp_equil) - int(first_cyp_equil)

        # count dummies at start of the PDB (e.g., DUM)
        n_dum = count_leading_resname("build_ter.pdb", "DUM")

        # chain A ends at (prot_len + n_dum) in the PDB numbering
        insert_after = prot_len_dec + n_dum

        # chain B starts at the next residue
        chainB_start = insert_after + 1

        insert_ter_after_resnum("build_ter.pdb", insert_after)
        remove_ter_after_resnum("build_ter.pdb", chainB_start)

    # insert_ter_after_resnum("build_ter.pdb", 422)
    # remove_ter_after_resnum("build_ter.pdb", 504)

    shutil.move("build.pdb", "build_backup.pdb")
    shutil.move("build_ter.pdb", "build.pdb")
    print_line_count("build_backup.pdb")

    # -----------------------------------
    # Write anchors to equil-pdb
    # -----------------------------------

    with open(f"equil-{mol.lower()}.pdb", "r") as fin:
        data = fin.read().splitlines(True)

    with open(f"equil-{mol.lower()}.pdb", "w") as fout:
        fout.write(
            "%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n"
            % ("REMARK A", P1, P2, P3, L1, L2, L3, first_res, recep_last)
        )
        fout.writelines(data[1:])

    # -----------------------------------
    # Warning for missing residues
    # -----------------------------------

    if recep_last != recep_resid_num:
        print(
            "WARNING: Missing residues in the receptor protein sequence. Unless the protein is engineered this is not recommended."
        )
        print("A protein modeling tool might be required before running equilibration.")

    # === 🛠 FIX ligand_resid dynamically based on actual build.pdb ===

    # Find the first ligand atom in build.pdb
    lig_resid = None
    # === Fix misclassified atoms: separate true receptor, ligand, other ===
    fixed_recep_coords = []
    fixed_recep_atomlist = []
    fixed_recep_rsnmlist = []
    fixed_recep_rsidlist = []
    fixed_recep_chainlist = []

    fixed_lig_coords = []
    fixed_lig_atomlist = []
    fixed_lig_rsnmlist = []
    fixed_lig_rsidlist = []
    fixed_lig_chainlist = []

    fixed_oth_coords = []
    fixed_oth_atomlist = []
    fixed_oth_rsnmlist = []
    fixed_oth_rsidlist = []
    fixed_oth_chainlist = []

    lig_resid_seen = set()
    oth_resid_seen = set()
    recep_resid_seen = set()

    print("🔍 Begin atom classification from coords...")
    for i in range(dum_atom, len(coords)):
        resname = resname_list[i]
        resid = resid_list[i]
        atomname = atom_namelist[i]
        chain = chain_list[i]
        xyz = coords[i]

        tag = f"{resname}{resid}"

        if resname == mol:
            print(f"🧪 Ligand: {tag:10s} atom: {atomname}")
            fixed_lig_coords.append(xyz)
            fixed_lig_atomlist.append(atomname)
            fixed_lig_rsnmlist.append(resname)
            fixed_lig_rsidlist.append(resid)
            fixed_lig_chainlist.append(chain)
            lig_resid_seen.add(resid)

        elif resname in other_mol or resname == "WAT":
            # print(f"OtherMol: {tag:10s} atom: {atomname}")
            fixed_oth_coords.append(xyz)
            fixed_oth_atomlist.append(atomname)
            fixed_oth_rsnmlist.append(resname)
            fixed_oth_rsidlist.append(resid)
            fixed_oth_chainlist.append(chain)
            oth_resid_seen.add(resid)

    # === Post-sort diagnostics ===
    print("\n✅ Sorting Summary:")
    # print(f"Receptor residues: {sorted(recep_resid_seen)}")
    print(f"Ligand residues  : {sorted(lig_resid_seen)} ({mol})")
    print(f"Other molecules : {sorted(oth_resid_seen)} ({other_mol})")

    # Rebuild atom lists
    recep_atom = len(fixed_recep_coords)
    lig_atom = len(fixed_lig_coords)
    oth_atom = len(fixed_oth_coords)
    total_atom = dum_atom + recep_atom + lig_atom + oth_atom

    coords = dum_coords + fixed_recep_coords + fixed_lig_coords + fixed_oth_coords
    atom_namelist = (
        dum_atomlist + fixed_recep_atomlist + fixed_lig_atomlist + fixed_oth_atomlist
    )
    resid_list = (
        dum_rsidlist + fixed_recep_rsidlist + fixed_lig_rsidlist + fixed_oth_rsidlist
    )
    resname_list = (
        dum_rsnmlist + fixed_recep_rsnmlist + fixed_lig_rsnmlist + fixed_oth_rsnmlist
    )
    chain_list = (
        dum_chainlist
        + fixed_recep_chainlist
        + fixed_lig_chainlist
        + fixed_oth_chainlist
    )

    # Final check: print residue sequence order
    # print("\n🧾 DEBUG: Final residue write order in build.pdb:")
    # for idx, (rname, rid) in enumerate(zip(resname_list, resid_list)):
    #    if idx < dum_atom:
    #        label = "DUM"
    #    elif idx < dum_atom + recep_atom:
    #        label = "REC"
    #    elif idx < dum_atom + recep_atom + lig_atom:
    #        label = "LIG"
    #    else:
    #        label = "OTH"
    #    print(f"{idx+1:5d}: {label} {rname}{rid}")

    with open("build.pdb") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname == mol:  # matching ligand name
                    lig_resid = line[22:26].strip()
                    print(lig_resid)
                    break

    if lig_resid is None:
        raise ValueError("Ligand residue ID not found in build.pdb!")

    print(f"Corrected ligand residue ID found: {lig_resid}")

    # Rebuild ligand anchors using correct lig_resid
    anchor_file = "anchors.txt"
    with open(anchor_file, "r") as f:
        for line in f:
            splitdata = line.split()
            if len(splitdata) >= 3:
                L1 = ":" + lig_resid + "@" + splitdata[0]
                L2 = ":" + lig_resid + "@" + splitdata[1]
                L3 = ":" + lig_resid + "@" + splitdata[2]

    print("✅ Correct Ligand anchors after fix:")
    print(L1)
    print(L2)
    print(L3)

    # Write dry build file

    with open("build.pdb") as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = list(line for line in lines if line)  # Non-blank lines in a list
    with open("./build-dry.pdb", "w") as outfile:
        for i in range(0, len(lines)):
            if lines[i][17:20].strip() == "WAT":
                break
            outfile.write(lines[i] + "\n")

    outfile.close()

    # Append to main disang.rest used by eqnpt
    print(f"Current working directory: {os.getcwd()}")
    print(f"Copying CYP library to current directory")
    shutil.copy("../build_files/cyp.mol2", "./cyp.mol2")

    # Only for non-bonded
    # shutil.copy("../build_files/fe_heme.rest", "./fe_heme.rest")

    # print("Generating FE-Heme restraints file from full.pdb")
    # write_fe_heme_restraints("full.pdb", "fe_heme_full.rest", k=100.0, r0=2.0)
    # append_fe_heme_restraints_to_disang()

    os.chdir("../")

    return "all"


############################Original Functions###############


def build_equil(
    pose,
    celp_st,
    mol,
    H1,
    H2,
    H3,
    calc_type,
    l1_x,
    l1_y,
    l1_z,
    l1_range,
    min_adis,
    max_adis,
    ligand_ff,
    ligand_ph,
    retain_lig_prot,
    ligand_charge,
    other_mol,
    solv_shell,
):

    # Not apply SDR distance when equilibrating
    sdr_dist = 0

    # Create equilibrium directory
    if not os.path.exists("equil"):
        os.makedirs("equil")
    os.chdir("equil")
    if os.path.exists("./build_files"):
        shutil.rmtree("./build_files")
    try:
        shutil.copytree("../build_files", "./build_files")
    # Directories are the same
    except shutil.Error as e:
        print("Directory not copied. Error: %s" % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print("Directory not copied. Error: %s" % e)
    os.chdir("build_files")

    if calc_type == "dock":
        shutil.copy("../../all-poses/%s_docked.pdb" % (celp_st), "./rec_file.pdb")
        shutil.copy("../../all-poses/%s.pdb" % (pose), "./")
    elif calc_type == "rank":
        shutil.copy("../../all-poses/%s.pdb" % (celp_st), "./rec_file.pdb")
        shutil.copy("../../all-poses/%s.pdb" % (pose), "./")
    elif calc_type == "crystal":
        shutil.copy("../../all-poses/%s.pdb" % (pose), "./")
        # Replace names and run initial VMD script
        with open("prep-crystal.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(
                        line.replace("MMM", mol)
                        .replace("mmm", mol.lower())
                        .replace("CCCC", pose)
                    )
        sp.call("vmd -dispdev text -e prep.tcl", shell=True)

    # Split initial receptor file
    with open("split-ini.tcl", "rt") as fin:
        with open("split.tcl", "wt") as fout:
            if other_mol:
                other_mol_vmd = " ".join(other_mol)
            else:
                other_mol_vmd = "XXX"
            for line in fin:
                if "lig" not in line:
                    fout.write(
                        line.replace("SHLL", "%4.2f" % solv_shell)
                        .replace("OTHRS", str(other_mol_vmd))
                        .replace("MMM", mol.upper())
                    )
    sp.call("vmd -dispdev text -e split.tcl", shell=True)

    # Remove possible remaining molecules
    if not other_mol:
        open("others.pdb", "w").close()

    shutil.copy("./protein.pdb", "./protein_vmd.pdb")
    sp.call("pdb4amber -i protein_vmd.pdb -o protein.pdb -y", shell=True)
    # sp.call("cp protein_vmd.pdb  protein.pdb", shell=True)

    # Get beginning and end of protein and save first residue as global variable
    with open("./protein_vmd.pdb") as myfile:
        data = myfile.readlines()
        first_res = int(data[1][22:26].strip())
    with open("./protein.pdb") as myfile:
        data = myfile.readlines()
        recep_resid_num = int(data[-2][22:26].strip())
    print("Receptor first residue: %s" % first_res)
    print("Receptor total length: %s" % recep_resid_num)

    # Adjust protein anchors to the new residue numbering
    h1_resid = H1.split("@")[0][1:]
    h2_resid = H2.split("@")[0][1:]
    h3_resid = H3.split("@")[0][1:]

    h1_atom = H1.split("@")[1]
    h2_atom = H2.split("@")[1]
    h3_atom = H3.split("@")[1]

    p1_resid = str(int(h1_resid) - int(first_res) + 2)
    p2_resid = str(int(h2_resid) - int(first_res) + 2)
    p3_resid = str(int(h3_resid) - int(first_res) + 2)

    p1_vmd = str(int(h1_resid) - int(first_res) + 1)

    P1 = ":" + p1_resid + "@" + h1_atom
    P2 = ":" + p2_resid + "@" + h2_atom
    P3 = ":" + p3_resid + "@" + h3_atom

    print("Receptor anchors:")
    print(P1)
    print(P2)
    print(P3)

    # Replace names in initial files and VMD scripts
    with open("prep-ini.tcl", "rt") as fin:
        with open("prep.tcl", "wt") as fout:
            other_mol_vmd = " ".join(other_mol)
            for line in fin:
                fout.write(
                    line.replace("MMM", mol)
                    .replace("mmm", mol.lower())
                    .replace("NN", h1_atom)
                    .replace("P1A", p1_vmd)
                    .replace("FIRST", "1")
                    .replace("LAST", str(recep_resid_num))
                    .replace("STAGE", "equil")
                    .replace("XDIS", "%4.2f" % l1_x)
                    .replace("YDIS", "%4.2f" % l1_y)
                    .replace("ZDIS", "%4.2f" % l1_z)
                    .replace("RANG", "%4.2f" % l1_range)
                    .replace("DMAX", "%4.2f" % max_adis)
                    .replace("DMIN", "%4.2f" % min_adis)
                    .replace("SDRD", "%4.2f" % sdr_dist)
                    .replace("OTHRS", str(other_mol_vmd))
                )
    #    with open('%s.pdb' %pose) as f:
    #       data=f.read().replace('LIG','%s' %mol)
    #    with open('%s.pdb' %pose, "w") as f:
    #       f.write(data)

    # Save parameters in ff folder
    if not os.path.exists("../ff/"):
        os.makedirs("../ff/")
    for file in glob.glob("./*.mol2"):
        shutil.copy(file, "../ff/")
    for file in glob.glob("./*.frcmod"):
        shutil.copy(file, "../ff/")
    shutil.copy("./dum.mol2", "../ff/")
    shutil.copy("./dum.frcmod", "../ff/")

    # Adjust ligand files
    # Mudong's mod: optionally retain the ligand protonation state as provided in pose*.pdb, and skip Babel processing (removing H, adding H, determining total charge)
    if retain_lig_prot == "yes":
        # Determine ligand net charge by reading the rightmost column of pose*.pdb, programs such as Maestro writes atom charges there
        if ligand_charge == "nd":
            ligand_charge = 0
            with open("" + pose + ".pdb") as f_in:
                for line in f_in:
                    if "1+" in line:
                        ligand_charge += 1
                    elif "2+" in line:
                        ligand_charge += 2
                    elif "3+" in line:
                        ligand_charge += 3
                    elif "4+" in line:
                        ligand_charge += 4
                    elif "1-" in line:
                        ligand_charge += -1
                    elif "2-" in line:
                        ligand_charge += -2
                    elif "3-" in line:
                        ligand_charge += -3
                    elif "4-" in line:
                        ligand_charge += -4
        print("The net charge of the ligand is %d" % ligand_charge)
        if calc_type == "dock" or calc_type == "rank":
            shutil.copy("./" + pose + ".pdb", "./" + mol.lower() + "-h.pdb")
        elif calc_type == "crystal":
            shutil.copy("./" + mol.lower() + ".pdb", "./" + mol.lower() + "-h.pdb")
    else:
        if calc_type == "dock" or calc_type == "rank":
            sp.call(
                "obabel -i pdb " + pose + ".pdb -o pdb -O " + mol.lower() + ".pdb -d",
                shell=True,
            )  # Remove all hydrogens from the ligand
        elif calc_type == "crystal":
            sp.call(
                "obabel -i pdb "
                + mol.lower()
                + ".pdb -o pdb -O "
                + mol.lower()
                + ".pdb -d",
                shell=True,
            )  # Remove all hydrogens from crystal ligand
        sp.call(
            "obabel -i pdb "
            + mol.lower()
            + ".pdb -o pdb -O "
            + mol.lower()
            + "-h-ini.pdb -p %4.2f" % ligand_ph,
            shell=True,
        )  # Put all hydrogens back using babel
        sp.call(
            "obabel -i pdb "
            + mol.lower()
            + ".pdb -o mol2 -O "
            + mol.lower()
            + "-crg.mol2 -p %4.2f" % ligand_ph,
            shell=True,
        )
        # Clean ligand protonated pdb file
        with open(mol.lower() + "-h-ini.pdb") as oldfile, open(
            mol.lower() + "-h.pdb", "w"
        ) as newfile:
            for line in oldfile:
                if "ATOM" in line or "HETATM" in line:
                    newfile.write(line)
            newfile.close()
        if ligand_charge == "nd":
            ligand_charge = 0
            # Get ligand net charge from babel
            lig_crg = 0
            with open("%s-crg.mol2" % mol.lower()) as f_in:
                for line in f_in:
                    splitdata = line.split()
                    if len(splitdata) > 8:
                        lig_crg = lig_crg + float(splitdata[8].strip())
            print(lig_crg)
            ligand_charge = round(lig_crg)
        print("The babel protonation of the ligand is for pH %4.2f" % ligand_ph)
        print("The net charge of the ligand is %d" % ligand_charge)

    # Get ligand parameters
    if not os.path.exists("../ff/%s.mol2" % mol.lower()):
        print(
            "Antechamber parameters command: antechamber -i "
            + mol.lower()
            + "-h.pdb -fi pdb -o "
            + mol.lower()
            + ".mol2 -fo mol2 -c bcc -s 2 -at "
            + ligand_ff.lower()
            + " -nc %d" % ligand_charge
        )
        sp.call(
            "antechamber -i "
            + mol.lower()
            + "-h.pdb -fi pdb -o "
            + mol.lower()
            + ".mol2 -fo mol2 -c bcc -s 2 -at "
            + ligand_ff.lower()
            + " -nc %d" % ligand_charge,
            shell=True,
        )
        shutil.copy("./%s.mol2" % (mol.lower()), "../ff/")
    if not os.path.exists("../ff/%s.frcmod" % mol.lower()):
        if ligand_ff == "gaff":
            sp.call(
                "parmchk2 -i "
                + mol.lower()
                + ".mol2 -f mol2 -o "
                + mol.lower()
                + ".frcmod -s 1",
                shell=True,
            )
        elif ligand_ff == "gaff2":
            sp.call(
                "parmchk2 -i "
                + mol.lower()
                + ".mol2 -f mol2 -o "
                + mol.lower()
                + ".frcmod -s 2",
                shell=True,
            )
        shutil.copy("./%s.frcmod" % (mol.lower()), "../ff/")
    sp.call(
        "antechamber -i "
        + mol.lower()
        + "-h.pdb -fi pdb -o "
        + mol.lower()
        + ".pdb -fo pdb",
        shell=True,
    )

    # Create raw complex and clean it
    filenames = ["protein.pdb", "%s.pdb" % mol.lower(), "others.pdb", "crystalwat.pdb"]
    with open("./complex-merge.pdb", "w") as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    with open("complex-merge.pdb") as oldfile, open("complex.pdb", "w") as newfile:
        for line in oldfile:
            if not "CRYST1" in line and not "CONECT" in line and not "END" in line:
                newfile.write(line)

    # New work around to avoid chain swapping during alignment
    sp.call("pdb4amber -i reference.pdb -o reference_amber.pdb -y", shell=True)
    # sp.call("cp reference.pdb  reference_amber.pdb", shell=True)
    sp.call("vmd -dispdev text -e nochain.tcl", shell=True)
    sp.call(
        "./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc",
        shell=True,
    )
    sp.call("vmd -dispdev text -e measure-fit.tcl", shell=True)

    # Put in AMBER format and find ligand anchor atoms
    with open("aligned.pdb", "r") as oldfile, open("aligned-clean.pdb", "w") as newfile:
        for line in oldfile:
            splitdata = line.split()
            if len(splitdata) > 4:
                newfile.write(line)
    sp.call(
        "pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y",
        shell=True,
    )
    # sp.call("cp aligned-clean.pdb aligned_amber.pdb", shell=True)
    sp.call("vmd -dispdev text -e prep.tcl", shell=True)

    # Check size of anchor file
    anchor_file = "anchors.txt"
    if os.stat(anchor_file).st_size == 0:
        os.chdir("../")
        return "anch1"
    f = open(anchor_file, "r")
    for line in f:
        splitdata = line.split()
        if len(splitdata) < 3:
            os.rename("./anchors.txt", "anchors-" + pose + ".txt")
            os.chdir("../")
            return "anch2"
    os.rename("./anchors.txt", "anchors-" + pose + ".txt")

    os.chdir("../")

    # Create simulation directory
    if not os.path.exists(pose):
        os.makedirs(pose)
    os.chdir(pose)

    #####################Inside simulation Directory POSE#################

    dum_coords = []
    recep_coords = []
    lig_coords = []
    oth_coords = []
    dum_atomlist = []
    lig_atomlist = []
    recep_atomlist = []
    oth_atomlist = []
    dum_rsnmlist = []
    recep_rsnmlist = []
    lig_rsnmlist = []
    oth_rsnmlist = []
    dum_rsidlist = []
    recep_rsidlist = []
    lig_rsidlist = []
    oth_rsidlist = []
    dum_chainlist = []
    recep_chainlist = []
    lig_chainlist = []
    oth_chainlist = []
    dum_atom = 0
    lig_atom = 0
    recep_atom = 0
    oth_atom = 0
    total_atom = 0
    resid_lig = 0
    resname_lig = mol

    # Copy a few files
    shutil.copy("../build_files/equil-%s.pdb" % mol.lower(), "./")
    shutil.copy("../build_files/%s-noh.pdb" % mol.lower(), "./%s.pdb" % mol.lower())
    shutil.copy("../build_files/anchors-" + pose + ".txt", "./anchors.txt")

    # Read coordinates for dummy atoms
    for i in range(1, 2):
        shutil.copy("../build_files/dum" + str(i) + ".pdb", "./")
        with open("dum" + str(i) + ".pdb") as dum_in:
            lines = (line.rstrip() for line in dum_in)
            lines = list(line for line in lines if line)
            dum_coords.append(
                (
                    float(lines[1][30:38].strip()),
                    float(lines[1][38:46].strip()),
                    float(lines[1][46:54].strip()),
                )
            )
            dum_atomlist.append(lines[1][12:16].strip())
            dum_rsnmlist.append(lines[1][17:20].strip())
            dum_rsidlist.append(float(lines[1][22:26].strip()))
            dum_chainlist.append(lines[1][21].strip())
            dum_atom += 1
            total_atom += 1

    # Read coordinates from aligned system
    with open("equil-%s.pdb" % mol.lower()) as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = list(line for line in lines if line)  # Non-blank lines in a list

    # Count atoms of receptor and ligand
    for i in range(0, len(lines)):
        if (lines[i][0:6].strip() == "ATOM") or (lines[i][0:6].strip() == "HETATM"):
            if (
                (lines[i][17:20].strip() != mol)
                and (lines[i][17:20].strip() != "DUM")
                and (lines[i][17:20].strip() != "WAT")
                and (lines[i][17:20].strip() not in other_mol)
            ):
                recep_coords.append(
                    (
                        float(lines[i][30:38].strip()),
                        float(lines[i][38:46].strip()),
                        float(lines[i][46:54].strip()),
                    )
                )
                recep_atomlist.append(lines[i][12:16].strip())
                recep_rsnmlist.append(lines[i][17:20].strip())
                recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                recep_chainlist.append(lines[i][21].strip())
                recep_last = int(lines[i][22:26].strip())
                recep_atom += 1
                total_atom += 1
            elif lines[i][17:20].strip() == mol:
                lig_coords.append(
                    (
                        float(lines[i][30:38].strip()),
                        float(lines[i][38:46].strip()),
                        float(lines[i][46:54].strip()),
                    )
                )
                lig_atomlist.append(lines[i][12:16].strip())
                lig_rsnmlist.append(lines[i][17:20].strip())
                lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                lig_chainlist.append(lines[i][21].strip())
                lig_atom += 1
                total_atom += 1
            elif (lines[i][17:20].strip() == "WAT") or (
                lines[i][17:20].strip() in other_mol
            ):
                oth_coords.append(
                    (
                        float(lines[i][30:38].strip()),
                        float(lines[i][38:46].strip()),
                        float(lines[i][46:54].strip()),
                    )
                )
                oth_atomlist.append(lines[i][12:16].strip())
                oth_rsnmlist.append(lines[i][17:20].strip())
                oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                oth_chainlist.append(lines[i][21].strip())
                oth_atom += 1
                total_atom += 1

    coords = dum_coords + recep_coords + lig_coords + oth_coords
    atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
    resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
    resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
    chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
    # lig_resid = str(recep_last + dum_atom + 1) # That did assume lig is last residue before water not valid in case of other_mol
    lig_resid = str(int(lig_rsidlist[0]))  # Get actual ligand resid directly
    chain_tmp = "None"
    resid_tmp = "None"

    # Read ligand anchors obtained from VMD
    anchor_file = "anchors.txt"
    f = open(anchor_file, "r")
    for line in f:
        splitdata = line.split()
        L1 = ":" + lig_resid + "@" + splitdata[0]
        L2 = ":" + lig_resid + "@" + splitdata[1]
        L3 = ":" + lig_resid + "@" + splitdata[2]

    print("Ligand anchors:")
    print(L1)
    print(L2)
    print(L3)
    print(other_mol)

    # -----------------------------------
    # Write the new pdb file
    # -----------------------------------

    build_file = open("build.pdb", "w")

    # Reorganize atoms into groups
    dummy_atoms = []
    receptor_atoms = []
    ligand_atoms = []
    other_atoms = []

    for i in range(len(atom_namelist)):
        atom_data = {
            "idx": i + 1,
            "atom_name": atom_namelist[i],
            "resname": resname_list[i],
            "resid": int(resid_list[i]),
            "coord": coords[i],
            "chain": chain_list[i] if i < len(chain_list) else " ",
        }

        if i < dum_atom:
            dummy_atoms.append(atom_data)
        elif dum_atom <= i < dum_atom + recep_atom:
            receptor_atoms.append(atom_data)
        elif dum_atom + recep_atom <= i < dum_atom + recep_atom + lig_atom:
            ligand_atoms.append(atom_data)
        else:
            other_atoms.append(atom_data)

    # Write dummy atoms
    for atom in dummy_atoms:
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
    build_file.write("TER\n")

    # Write receptor atoms
    prev_chain = ""
    for atom in receptor_atoms:
        if (
            atom["chain"] != prev_chain
            and atom["resname"] not in other_mol
            and atom["resname"] != "WAT"
        ):
            build_file.write("TER\n")
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
        prev_chain = atom["chain"]
    build_file.write("TER\n")

    # Write ligand atoms
    for atom in ligand_atoms:
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
    build_file.write("TER\n")

    # Write other atoms (other_mol + WAT)
    prev_resid = None
    for atom in other_atoms:
        if prev_resid is not None and atom["resid"] != prev_resid:
            build_file.write("TER\n")
        write_atom(
            build_file,
            atom["idx"],
            atom["atom_name"],
            atom["resname"],
            atom["resid"],
            atom["coord"],
        )
        prev_resid = atom["resid"]
    build_file.write("TER\n")
    build_file.write("END\n")

    build_file.close()

    ###Debug: closing the file earlier is important; Not closing the file created huge problem##########

    print("Number of lines in build.pdb before insertion")
    print_line_count("build.pdb")
    print("Inserting TER as in original pdb")
    insert_ter_from_reference(
        "../build_files/rec_file.pdb", "build.pdb", "build_ter.pdb"
    )
    shutil.move("build.pdb", "build_backup.pdb")
    shutil.move("build_ter.pdb", "build.pdb")
    print_line_count("build_backup.pdb")

    # -----------------------------------
    # Write anchors to equil-pdb
    # -----------------------------------

    with open(f"equil-{mol.lower()}.pdb", "r") as fin:
        data = fin.read().splitlines(True)

    with open(f"equil-{mol.lower()}.pdb", "w") as fout:
        fout.write(
            "%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n"
            % ("REMARK A", P1, P2, P3, L1, L2, L3, first_res, recep_last)
        )
        fout.writelines(data[1:])

    # -----------------------------------
    # Warning for missing residues
    # -----------------------------------

    if recep_last != recep_resid_num:
        print(
            "WARNING: Missing residues in the receptor protein sequence. Unless the protein is engineered this is not recommended."
        )
        print("A protein modeling tool might be required before running equilibration.")

    # === FIX ligand_resid dynamically based on actual build.pdb ===

    # Find the first ligand atom in build.pdb
    lig_resid = None
    # === Fix misclassified atoms: separate true receptor, ligand, other ===
    fixed_recep_coords = []
    fixed_recep_atomlist = []
    fixed_recep_rsnmlist = []
    fixed_recep_rsidlist = []
    fixed_recep_chainlist = []

    fixed_lig_coords = []
    fixed_lig_atomlist = []
    fixed_lig_rsnmlist = []
    fixed_lig_rsidlist = []
    fixed_lig_chainlist = []

    fixed_oth_coords = []
    fixed_oth_atomlist = []
    fixed_oth_rsnmlist = []
    fixed_oth_rsidlist = []
    fixed_oth_chainlist = []

    lig_resid_seen = set()
    oth_resid_seen = set()
    recep_resid_seen = set()

    print("Begin atom classification from coords...")
    for i in range(dum_atom, len(coords)):
        resname = resname_list[i]
        resid = resid_list[i]
        atomname = atom_namelist[i]
        chain = chain_list[i]
        xyz = coords[i]

        tag = f"{resname}{resid}"

        if resname == mol:
            print(f" Ligand: {tag:10s} atom: {atomname}")
            fixed_lig_coords.append(xyz)
            fixed_lig_atomlist.append(atomname)
            fixed_lig_rsnmlist.append(resname)
            fixed_lig_rsidlist.append(resid)
            fixed_lig_chainlist.append(chain)
            lig_resid_seen.add(resid)

        elif resname in other_mol or resname == "WAT":
            # print(f"  OtherMol: {tag:10s} atom: {atomname}")
            fixed_oth_coords.append(xyz)
            fixed_oth_atomlist.append(atomname)
            fixed_oth_rsnmlist.append(resname)
            fixed_oth_rsidlist.append(resid)
            fixed_oth_chainlist.append(chain)
            oth_resid_seen.add(resid)

        else:
            print(f"Receptor: {tag:10s} atom: {atomname}")
            fixed_recep_coords.append(xyz)
            fixed_recep_atomlist.append(atomname)
            fixed_recep_rsnmlist.append(resname)
            fixed_recep_rsidlist.append(resid)
            fixed_recep_chainlist.append(chain)
            recep_resid_seen.add(resid)

    # === Post-sort diagnostics ===
    print("\n✅ Sorting Summary:")
    # print(f"Receptor residues: {sorted(recep_resid_seen)}")
    print(f"Ligand residues  : {sorted(lig_resid_seen)} ({mol})")
    print(f"Other molecules : {sorted(oth_resid_seen)} ({other_mol})")

    # Rebuild atom lists
    recep_atom = len(fixed_recep_coords)
    lig_atom = len(fixed_lig_coords)
    oth_atom = len(fixed_oth_coords)
    total_atom = dum_atom + recep_atom + lig_atom + oth_atom

    coords = dum_coords + fixed_recep_coords + fixed_lig_coords + fixed_oth_coords
    atom_namelist = (
        dum_atomlist + fixed_recep_atomlist + fixed_lig_atomlist + fixed_oth_atomlist
    )
    resid_list = (
        dum_rsidlist + fixed_recep_rsidlist + fixed_lig_rsidlist + fixed_oth_rsidlist
    )
    resname_list = (
        dum_rsnmlist + fixed_recep_rsnmlist + fixed_lig_rsnmlist + fixed_oth_rsnmlist
    )
    chain_list = (
        dum_chainlist
        + fixed_recep_chainlist
        + fixed_lig_chainlist
        + fixed_oth_chainlist
    )

    # Final check: print residue sequence order
    print("\n🧾 Final residue write order in build.pdb:")
    for idx, (rname, rid) in enumerate(zip(resname_list, resid_list)):
        if idx < dum_atom:
            label = "DUM"
        # elif idx < dum_atom + recep_atom:
        #    label = "REC"
        elif idx < dum_atom + recep_atom + lig_atom:
            label = "LIG"
        else:
            label = "OTH"
        print(f"{idx+1:5d}: {label} {rname}{rid}")

    with open("build.pdb") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname == mol:  # matching ligand name
                    lig_resid = line[22:26].strip()
                    print(lig_resid)
                    break

    if lig_resid is None:
        raise ValueError("Ligand residue ID not found in build.pdb!")

    print(f"📢 Corrected ligand residue ID found: {lig_resid}")

    # Rebuild ligand anchors using correct lig_resid
    anchor_file = "anchors.txt"
    with open(anchor_file, "r") as f:
        for line in f:
            splitdata = line.split()
            if len(splitdata) >= 3:
                L1 = ":" + lig_resid + "@" + splitdata[0]
                L2 = ":" + lig_resid + "@" + splitdata[1]
                L3 = ":" + lig_resid + "@" + splitdata[2]

    print("✅ Correct Ligand anchors after fix:")
    print(L1)
    print(L2)
    print(L3)

    # Write dry build file

    with open("build.pdb") as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = list(line for line in lines if line)  # Non-blank lines in a list
    with open("./build-dry.pdb", "w") as outfile:
        for i in range(0, len(lines)):
            if lines[i][17:20].strip() == "WAT":
                break
            outfile.write(lines[i] + "\n")

    outfile.close()

    os.chdir("../")

    return "all"


############# Building Decouple System ##################


def build_dec(
    fwin,
    hmr,
    mol,
    pose,
    molr,
    poser,
    comp,
    win,
    water_model,
    ntpr,
    ntwr,
    ntwe,
    ntwx,
    cut,
    gamma_ln,
    barostat,
    receptor_ff,
    ligand_ff,
    dt,
    sdr_dist,
    dec_method,
    l1_x,
    l1_y,
    l1_z,
    l1_range,
    min_adis,
    max_adis,
    ion_def,
    other_mol,
    solv_shell,
    first_cyp_dec=None,
    second_cyp_dec=None,
    first_cyp_next_dec=None,
    second_cyp_next_dec=None,
    first_cyp_previous_dec=None,
    second_cyp_previous_dec=None,
    heme_1=None,
    heme_2=None,
    sdr_axis=None,
):

    # --- helper: suggest SDR shift (dx, dy, dz) orthogonal to the dimer axis ---
    def _suggest_sdr_shift(rec_amber_path, sdr_dist):
        """
        Heuristic:
        - Read first TER split in rec_amber to separate chain A / chain B.
        - Compute COM of each chain (heavy atoms only to reduce bias).
        - Dimer axis = COM_B - COM_A. Pick a perpendicular axis for the SDR shift.
        - Return a pure-axis shift of magnitude sdr_dist (default positive direction).
        """
        import math

        def _parse_atoms(pdb_path):
            atoms = []
            with open(pdb_path) as f:
                for line in f:
                    rec = line[0:6].strip()
                    if rec in ("ATOM", "HETATM"):
                        # skip hydrogens for COM
                        el = line[76:78].strip() if len(line) >= 78 else ""
                        if el.upper().startswith("H"):
                            continue
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            resseq = int(line[22:26])
                        except Exception:
                            continue
                        atoms.append((resseq, x, y, z))
                    elif rec == "TER":
                        atoms.append(("TER",))  # marker
            return atoms

        def _split_at_first_ter(atoms):
            """Split atoms into two blocks at the first TER encountered."""
            chainA, chainB = [], []
            saw_ter = False
            for rec in atoms:
                if rec == ("TER",):
                    saw_ter = True
                    continue
                if not saw_ter:
                    chainA.append(rec)
                else:
                    chainB.append(rec)
            return chainA, chainB

        def _com(block):
            if not block:
                return (0.0, 0.0, 0.0)
            sx = sy = sz = 0.0
            n = 0
            for rec in block:
                if rec == ("TER",):
                    continue
                _, x, y, z = rec
                sx += x
                sy += y
                sz += z
                n += 1
            if n == 0:
                return (0.0, 0.0, 0.0)
            return (sx / n, sy / n, sz / n)

        # Fallback if rec_amber not found
        try_paths = [
            "../build_files/rec_amber.pdb",
            "../../build_files/rec_amber.pdb",
            "../build_files/rec_file.pdb",
            "../../build_files/rec_file.pdb",
        ]
        pdb_path = None
        for p in try_paths:
            if os.path.exists(p):
                pdb_path = p
                break
        if pdb_path is None:
            # Safe default: move along +Z if nothing is available
            return (0.0, 0.0, float(sdr_dist))

        atoms = _parse_atoms(pdb_path)
        chainA, chainB = _split_at_first_ter(atoms)
        Ax, Ay, Az = _com(chainA)
        Bx, By, Bz = _com(chainB)
        vx, vy, vz = (Bx - Ax, By - Ay, Bz - Az)

        # find dominant (dimer) axis
        abs_v = (abs(vx), abs(vy), abs(vz))
        max_axis = ["x", "y", "z"][abs_v.index(max(abs_v))]

        # choose a perpendicular axis for SDR shift (simple, deterministic rule)
        # if dimer axis is z -> shift along +y
        # if dimer axis is y -> shift along +z
        # if dimer axis is x -> shift along +y
        if max_axis == "z":
            return (0.0, float(sdr_dist), 0.0)
        elif max_axis == "y":
            return (0.0, 0.0, float(sdr_dist))
        else:  # "x"
            return (0.0, float(sdr_dist), 0.0)

    ########Main Builder Function############

    # Decide SDR translation (dx, dy, dz) for free ligand placement
    # sdr_dx, sdr_dy, sdr_dz = _suggest_sdr_shift(
    #    "../build_files/rec_amber.pdb", sdr_dist
    # )
    # sdr_dx, sdr_dy, sdr_dz = 0.0, -45.0, 0.0  # shift along -Y
    def _axis_to_vec(axis_spec, dist):
        """
        axis_spec examples:
          'x', '-y', 'z'
          'x+y', '-x+y', 'x-y'
          'x,y', '-x,+y'
        """
        import math

        axis_spec = axis_spec.replace(",", "+").replace(" ", "")
        parts = axis_spec.split("+")

        vx = vy = vz = 0.0

        for p in parts:
            if p == "x":
                vx += 1.0
            elif p == "-x":
                vx -= 1.0
            elif p == "y":
                vy += 1.0
            elif p == "-y":
                vy -= 1.0
            elif p == "z":
                vz += 1.0
            elif p == "-z":
                vz -= 1.0
            else:
                raise ValueError(f"Invalid sdr_axis component: '{p}'")

        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if norm == 0.0:
            raise ValueError("sdr_axis resulted in zero vector")

        # normalize & scale
        scale = float(dist) / norm
        return vx * scale, vy * scale, vz * scale

    # Decide SDR translation (dx, dy, dz)
    # Policy:
    # - If user sets sdr_axis: obey it
    # - Else: if dimer likely → auto; otherwise keep old behavior (-y)
    if sdr_axis is None or sdr_axis == "":
        # heuristic default: in your heme fork, num_chains==2 is common
        # but build_dec doesn't know num_chains; so just use auto if possible
        sdr_axis_eff = "auto"
    else:
        sdr_axis_eff = sdr_axis

    if sdr_axis_eff == "auto":
        sdr_dx, sdr_dy, sdr_dz = _suggest_sdr_shift(
            "../build_files/rec_amber.pdb", sdr_dist
        )
    else:
        sdr_dx, sdr_dy, sdr_dz = _axis_to_vec(sdr_axis_eff, sdr_dist)

    # print(
    #    f"[SDR] axis={sdr_axis_eff} dist={float(sdr_dist):.2f} -> dx={sdr_dx:.2f} dy={sdr_dy:.2f} dz={sdr_dz:.2f}",
    #    flush=True,
    # )

    if dec_method == "sdr":
        print(
            f"[SDR] axis={sdr_axis_eff} dist={float(sdr_dist):.2f} -> dx={sdr_dx:.2f} dy={sdr_dy:.2f} dz={sdr_dz:.2f}",
            flush=True,
        )
    elif dec_method == "dd":
        print(
            f"[DD] No SDR",
            flush=True,
        )
    else:
        print(
            f"[{dec_method.upper()}] dist={float(sdr_dist):.2f} -> dx={sdr_dx:.2f} dy={sdr_dy:.2f} dz={sdr_dz:.2f}",
            flush=True,
        )
    ##############Components Building#######################
    if comp == "n":
        dec_method = "sdr"

    if (
        comp == "a"
        or comp == "l"
        or comp == "t"
        or comp == "m"
        or comp == "c"
        or comp == "r"
    ):
        dec_method = "dd"

    if comp == "x":
        dec_method = "exchange"

    # Get files or finding new anchors and building some systems

    if (
        (not os.path.exists("../build_files"))
        or (dec_method == "sdr" and win == 0)
        or (dec_method == "exchange" and win == 0)
    ):
        if (dec_method == "sdr" or dec_method == "exchange") and os.path.exists(
            "../build_files"
        ):
            shutil.rmtree("../build_files")
        try:
            shutil.copytree("../../../build_files", "../build_files")
            shutil.copy(
                "../../../equil/build_files/rec_file.pdb",
                "../build_files/rec_original.pdb",
            )
        # Directories are the same
        except shutil.Error as e:
            print("Directory not copied. Error: %s" % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print("Directory not copied. Error: %s" % e)
        os.chdir("../build_files")
        # Get last state from equilibrium simulations
        shutil.copy("../../../equil/" + pose + "/md%02d.rst7" % fwin, "./")
        shutil.copy("../../../equil/" + pose + "/full.pdb", "./aligned-nc.pdb")
        for file in glob.glob("../../../equil/%s/full*.prmtop" % pose.lower()):
            shutil.copy(file, "./")
        for file in glob.glob("../../../equil/%s/vac*" % pose.lower()):
            shutil.copy(file, "./")
        sp.call(
            "cpptraj -p full.prmtop -y md%02d.rst7 -x rec_file.pdb" % fwin, shell=True
        )
        if (not os.path.exists("rec_file.pdb")) or os.path.getsize("rec_file.pdb") == 0:
            raise FileNotFoundError(
                f"cpptraj did not produce rec_file.pdb in {os.getcwd()}.\n"
                f"Expected full.prmtop and md{fwin:02d}.rst7 to be readable.\n"
                f"Check cpptraj output/logs and file paths."
            )

        # Split initial receptor file
        with open("split-ini.tcl", "rt") as fin:
            with open("split.tcl", "wt") as fout:
                if other_mol:
                    other_mol_vmd = " ".join(other_mol)
                else:
                    other_mol_vmd = "XXX"
                for line in fin:
                    fout.write(
                        line.replace("SHLL", "%4.2f" % solv_shell)
                        .replace("OTHRS", str(other_mol_vmd))
                        .replace("mmm", mol.lower())
                        .replace("MMM", mol.upper())
                    )
        sp.call("vmd -dispdev text -e split.tcl", shell=True)
        lig_pdb = f"{mol.lower()}.pdb"
        if not os.path.exists(lig_pdb):
            raise FileNotFoundError(
                f"VMD split did not produce {lig_pdb} in {os.getcwd()}. "
                f"Check split.tcl selection for ligand resname {mol}."
            )

        # Remove possible remaining molecules
        if not other_mol:
            open("others.pdb", "w").close()

        # Create raw complex and clean it
        filenames = [
            "dummy.pdb",
            "protein.pdb",
            "%s.pdb" % mol.lower(),
            "others.pdb",
            "crystalwat.pdb",
        ]
        with open("./complex-merge.pdb", "w") as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        with open("complex-merge.pdb") as oldfile, open("complex.pdb", "w") as newfile:
            for line in oldfile:
                if not "CRYST1" in line and not "CONECT" in line and not "END" in line:
                    newfile.write(line)

        # Read protein anchors and size from equilibrium
        with open("../../../equil/" + pose + "/equil-%s.pdb" % mol.lower(), "r") as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split("@")[0][1:]
        p1_atom = P1.split("@")[1]
        rec_res = int(recep_last) + 1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(
                        line.replace("MMM", mol)
                        .replace("mmm", mol.lower())
                        .replace("NN", p1_atom)
                        .replace("P1A", p1_vmd)
                        .replace("FIRST", "2")
                        .replace("LAST", str(rec_res))
                        .replace("STAGE", "fe")
                        .replace("XDIS", "%4.2f" % l1_x)
                        .replace("YDIS", "%4.2f" % l1_y)
                        .replace("ZDIS", "%4.2f" % l1_z)
                        .replace("RANG", "%4.2f" % l1_range)
                        .replace("DMAX", "%4.2f" % max_adis)
                        .replace("DMIN", "%4.2f" % min_adis)
                        .replace("SDRD", "%4.2f" % sdr_dist)
                        .replace("OTHRS", str(other_mol_vmd))
                    )

        # Align to reference (equilibrium) structure using VMD's measure fit
        sp.call("vmd -dispdev text -e measure-fit.tcl", shell=True)

        # Put in AMBER format and find ligand anchor atoms
        with open("aligned.pdb", "r") as oldfile, open(
            "aligned-clean.pdb", "w"
        ) as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        sp.call(
            "pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y",
            shell=True,
        )
        # sp.call("cp aligned-clean.pdb aligned_amber.pdb", shell=True)
        sp.call("vmd -dispdev text -e prep.tcl", shell=True)

        ##ADDITONAL code for shifting DUM2 (After Shifting the Free Ligand in SDR)

        # --- helper: read/write a single-atom PDB coordinate (keeps all other fields intact)
        def _read_first_xyz(pdb_path):
            with open(pdb_path) as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        return (
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        )
            raise RuntimeError(f"No ATOM/HETATM in {pdb_path}")

        def _rewrite_first_xyz(pdb_path, x, y, z):
            with open(pdb_path) as f:
                lines = f.readlines()
            with open(pdb_path, "w") as g:
                for line in lines:
                    if line.startswith(("ATOM", "HETATM")):
                        g.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                    else:
                        g.write(line)

        def _place_dum2_along_vector(dum1_path, dum2_path, dx, dy, dz, dist):
            # use (dx,dy,dz) as direction; normalize and scale by dist from dum1
            import math

            nx, ny, nz = dx, dy, dz
            n = math.sqrt(nx * nx + ny * ny + nz * nz)
            if n == 0.0:
                nx, ny, nz = 0.0, 0.0, 1.0  # fallback: +Z
                n = 1.0
            nx, ny, nz = nx / n, ny / n, nz / n
            x1, y1, z1 = _read_first_xyz(dum1_path)
            _rewrite_first_xyz(
                dum2_path, x1 + dist * nx, y1 + dist * ny, z1 + dist * nz
            )

        def _place_dum2_by_offset(dum1_path, dum2_path, dx, dy, dz):
            with open(dum1_path) as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        x1 = float(line[30:38].strip())
                        y1 = float(line[38:46].strip())
                        z1 = float(line[46:54].strip())
                        break
            with open(dum2_path) as f:
                lines = f.readlines()
            with open(dum2_path, "w") as g:
                for line in lines:
                    if line.startswith(("ATOM", "HETATM")):
                        g.write(
                            f"{line[:30]}{x1+dx:8.3f}{y1+dy:8.3f}{z1+dz:8.3f}{line[54:]}"
                        )
                    else:
                        g.write(line)

        # Make dum2 follow the configured direction instead of +Z
        try:
            _place_dum2_by_offset("dum1.pdb", "dum2.pdb", sdr_dx, sdr_dy, sdr_dz)
        except Exception as e:
            print("WARN: could not re-place dum2.pdb:", e)

        # End of Additional code #################

        # Check size of anchor file
        anchor_file = "anchors.txt"
        if os.stat(anchor_file).st_size == 0:
            os.chdir("../")
            return "anch1"
        f = open(anchor_file, "r")
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename("./anchors.txt", "anchors-" + pose + ".txt")
                os.chdir("../")
                return "anch2"
        os.rename("./anchors.txt", "anchors-" + pose + ".txt")

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = "anchors-" + pose + ".txt"
        f = open(anchor_file, "r")
        for line in f:
            splitdata = line.split()
            L1 = ":" + lig_resid + "@" + splitdata[0]
            L2 = ":" + lig_resid + "@" + splitdata[1]
            L3 = ":" + lig_resid + "@" + splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open("fe-%s.pdb" % mol.lower(), "r") as fin:
            data = fin.read().splitlines(True)
        with open("fe-%s.pdb" % mol.lower(), "w") as fout:
            fout.write(
                "%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n"
                % ("REMARK A", P1, P2, P3, L1, L2, L3, first_res, recep_last)
            )
            fout.writelines(data[1:])

        # Get parameters from equilibrium
        if not os.path.exists("../ff"):
            os.makedirs("../ff")
        for file in glob.glob("../../../equil/ff/*.mol2"):
            shutil.copy(file, "../ff/")
        for file in glob.glob("../../../equil/ff/*.frcmod"):
            shutil.copy(file, "../ff/")
        shutil.copy("../../../equil/ff/%s.mol2" % (mol.lower()), "../ff/")
        shutil.copy("../../../equil/ff/%s.frcmod" % (mol.lower()), "../ff/")
        shutil.copy("../../../equil/ff/dum.mol2", "../ff/")
        shutil.copy("../../../equil/ff/dum.frcmod", "../ff/")

        if comp == "v" or comp == "e" or comp == "w" or comp == "f":
            if dec_method == "dd":
                os.chdir("../dd/")
            if dec_method == "sdr" or dec_method == "exchange":
                os.chdir("../sdr/")
        elif comp != "x":
            os.chdir("../rest/")

    # Create reference for relative calculations
    if comp == "x" and win == 0:

        # Build reference ligand from last state of equilibrium simulations

        if not os.path.exists("../exchange_files"):
            shutil.copytree("../../../build_files", "../exchange_files")
        os.chdir("../exchange_files")
        shutil.copy("../../../equil/" + poser + "/md%02d.rst7" % fwin, "./")
        shutil.copy("../../../equil/" + pose + "/full.pdb", "./aligned-nc.pdb")
        for file in glob.glob("../../../equil/%s/full*.prmtop" % poser.lower()):
            shutil.copy(file, "./")
        for file in glob.glob("../../../equil/%s/vac*" % poser.lower()):
            shutil.copy(file, "./")
        sp.call(
            "cpptraj -p full.prmtop -y md%02d.rst7 -x rec_file.pdb" % fwin, shell=True
        )

        # Split initial receptor file
        with open("split-ini.tcl", "rt") as fin:
            with open("split.tcl", "wt") as fout:
                if other_mol:
                    other_mol_vmd = " ".join(other_mol)
                else:
                    other_mol_vmd = "XXX"
                for line in fin:
                    fout.write(
                        line.replace("SHLL", "%4.2f" % solv_shell)
                        .replace("OTHRS", str(other_mol_vmd))
                        .replace("mmm", molr.lower())
                        .replace("MMM", molr.upper())
                    )
        sp.call("vmd -dispdev text -e split.tcl", shell=True)

        # Remove possible remaining molecules
        if not other_mol:
            open("others.pdb", "w").close()

        # Create raw complex and clean it
        filenames = [
            "dummy.pdb",
            "protein.pdb",
            "%s.pdb" % molr.lower(),
            "others.pdb",
            "crystalwat.pdb",
        ]
        with open("./complex-merge.pdb", "w") as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        with open("complex-merge.pdb") as oldfile, open("complex.pdb", "w") as newfile:
            for line in oldfile:
                if not "CRYST1" in line and not "CONECT" in line and not "END" in line:
                    newfile.write(line)

        # Read protein anchors and size from equilibrium
        with open("../../../equil/" + poser + "/equil-%s.pdb" % molr.lower(), "r") as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split("@")[0][1:]
        p1_atom = P1.split("@")[1]
        rec_res = int(recep_last) + 1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(
                        line.replace("MMM", molr)
                        .replace("mmm", molr.lower())
                        .replace("NN", p1_atom)
                        .replace("P1A", p1_vmd)
                        .replace("FIRST", "2")
                        .replace("LAST", str(rec_res))
                        .replace("STAGE", "fe")
                        .replace("XDIS", "%4.2f" % l1_x)
                        .replace("YDIS", "%4.2f" % l1_y)
                        .replace("ZDIS", "%4.2f" % l1_z)
                        .replace("RANG", "%4.2f" % l1_range)
                        .replace("DMAX", "%4.2f" % max_adis)
                        .replace("DMIN", "%4.2f" % min_adis)
                        .replace("SDRD", "%4.2f" % sdr_dist)
                        .replace("OTHRS", str(other_mol_vmd))
                    )

        # Align to reference (equilibrium) structure using VMD's measure fit
        sp.call("vmd -dispdev text -e measure-fit.tcl", shell=True)

        # Put in AMBER format and find ligand anchor atoms
        with open("aligned.pdb", "r") as oldfile, open(
            "aligned-clean.pdb", "w"
        ) as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        sp.call("pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y", shell=True)
        # sp.call("cp aligned-clean.pdb aligned_amber.pdb", shell=True)
        sp.call("vmd -dispdev text -e prep.tcl", shell=True)

        # Check size of anchor file
        anchor_file = "anchors.txt"
        if os.stat(anchor_file).st_size == 0:
            os.chdir("../")
            return "anch1"
        f = open(anchor_file, "r")
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename("./anchors.txt", "anchors-" + poser + ".txt")
                os.chdir("../")
                return "anch2"
        os.rename("./anchors.txt", "anchors-" + poser + ".txt")

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = "anchors-" + poser + ".txt"
        f = open(anchor_file, "r")
        for line in f:
            splitdata = line.split()
            L1 = ":" + lig_resid + "@" + splitdata[0]
            L2 = ":" + lig_resid + "@" + splitdata[1]
            L3 = ":" + lig_resid + "@" + splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open("fe-%s.pdb" % molr.lower(), "r") as fin:
            data = fin.read().splitlines(True)
        with open("fe-%s.pdb" % molr.lower(), "w") as fout:
            fout.write(
                "%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n"
                % ("REMARK A", P1, P2, P3, L1, L2, L3, first_res, recep_last)
            )
            fout.writelines(data[1:])

        # Get parameters from equilibrium
        if not os.path.exists("../ff"):
            os.makedirs("../ff")
        for file in glob.glob("../../../equil/ff/*.mol2"):
            shutil.copy(file, "../ff/")
        for file in glob.glob("../../../equil/ff/*.frcmod"):
            shutil.copy(file, "../ff/")
        shutil.copy("../../../equil/ff/%s.mol2" % (molr.lower()), "../ff/")
        shutil.copy("../../../equil/ff/%s.frcmod" % (molr.lower()), "../ff/")
        shutil.copy("../../../equil/ff/dum.mol2", "../ff/")
        shutil.copy("../../../equil/ff/dum.frcmod", "../ff/")

        os.chdir("../sdr/")

    # Copy and replace simulation files for the first window
    if int(win) == 0:
        if os.path.exists("amber_files"):
            shutil.rmtree("./amber_files")
        try:
            shutil.copytree("../../../amber_files", "./amber_files")
        # Directories are the same
        except shutil.Error as e:
            print("Directory not copied. Error: %s" % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print("Directory not copied. Error: %s" % e)
        for dname, dirs, files in os.walk("./amber_files"):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                    s = (
                        s.replace("_step_", dt)
                        .replace("_ntpr_", ntpr)
                        .replace("_ntwr_", ntwr)
                        .replace("_ntwe_", ntwe)
                        .replace("_ntwx_", ntwx)
                        .replace("_cutoff_", cut)
                        .replace("_gamma_ln_", gamma_ln)
                        .replace("_barostat_", barostat)
                        .replace("_receptor_ff_", receptor_ff)
                        .replace("_ligand_ff_", ligand_ff)
                    )
                with open(fpath, "w") as f:
                    f.write(s)

        if os.path.exists("run_files"):
            shutil.rmtree("./run_files")
        try:
            shutil.copytree("../../../run_files", "./run_files")
        # Directories are the same
        except shutil.Error as e:
            print("Directory not copied. Error: %s" % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print("Directory not copied. Error: %s" % e)
        if hmr == "no":
            replacement = "full.prmtop"
            for dname, dirs, files in os.walk("./run_files"):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace("full.hmr.prmtop", replacement)
                    with open(fpath, "w") as f:
                        f.write(s)
        elif hmr == "yes":
            replacement = "full.hmr.prmtop"
            for dname, dirs, files in os.walk("./run_files"):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace("full.prmtop", replacement)
                    with open(fpath, "w") as f:
                        f.write(s)

    # Create window directory
    if not os.path.exists("%s%02d" % (comp, int(win))):
        os.makedirs("%s%02d" % (comp, int(win)))
    os.chdir("%s%02d" % (comp, int(win)))
    # Find already built system in restraint window
    altm = "None"
    altm_list = ["a00", "l00", "t00", "m00"]
    if comp == "a" or comp == "l" or comp == "t" or comp == "m":
        for i in altm_list:
            if os.path.exists("../" + i + "/full.hmr.prmtop"):
                altm = i
                break

    if int(win) == 0 and altm == "None":
        # Build new system (robustly locate build_files regardless of window depth)

        # Locate build_files directory (path depends on where this window lives)
        bf = None
        for cand in (
            "../build_files",
            "../../build_files",
            "../../../build_files",
            "./build_files",
        ):
            if os.path.isdir(cand):
                bf = cand
                break
        if bf is None:
            raise FileNotFoundError(
                f"build_dec: could not locate build_files directory from {os.getcwd()}"
            )

        # Copy vacuum ligand files
        for file in glob.glob(os.path.join(bf, "vac_ligand*")):
            shutil.copy(file, "./")

        # Copy ligand pdb (e.g., unl.pdb)
        lig_pdb = os.path.join(bf, f"{mol.lower()}.pdb")
        if not os.path.exists(lig_pdb):
            raise FileNotFoundError(
                f"build_dec: missing ligand pdb {lig_pdb} (mol={mol}). Expected it to exist in {bf}."
            )
        shutil.copy(lig_pdb, "./")

        # Copy FE build pdb
        fe_pdb = os.path.join(bf, f"fe-{mol.lower()}.pdb")
        if not os.path.exists(fe_pdb):
            raise FileNotFoundError(
                f"build_dec: missing FE pdb {fe_pdb} (mol={mol}). Expected it to exist in {bf}."
            )
        shutil.copy(fe_pdb, "./build-ini.pdb")
        shutil.copy(fe_pdb, "./")

        # Copy anchors
        anchors_txt = os.path.join(bf, f"anchors-{pose}.txt")
        if not os.path.exists(anchors_txt):
            raise FileNotFoundError(
                f"build_dec: missing anchors file {anchors_txt}. Expected it to exist in {bf}."
            )
        shutil.copy(anchors_txt, "./")

        # Copy FF/parameter files (keep original behavior/paths)
        for file in glob.glob("../../ff/*.mol2"):
            shutil.copy(file, "./")
        for file in glob.glob("../../ff/*.frcmod"):
            shutil.copy(file, "./")
        for file in glob.glob("../../ff/%s.*" % mol.lower()):
            shutil.copy(file, "./")
        for file in glob.glob("../../ff/dum.*"):
            shutil.copy(file, "./")

        # Get TER statements
        ter_atom = []
        with open("../../build_files/rec_file.pdb") as oldfile, open(
            "rec_file-clean.pdb", "w"
        ) as newfile:
            for line in oldfile:
                if not "WAT" in line:
                    newfile.write(line)
        sp.call("pdb4amber -i rec_file-clean.pdb -o rec_amber.pdb -y", shell=True)
        # sp.call("cp rec_file-clean.pdb rec_amber.pdb", shell=True)
        with open("./rec_amber.pdb") as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        for i in range(0, len(lines)):
            if lines[i][0:6].strip() == "TER":
                ter_atom.append(int(lines[i][6:11].strip()))

        dum_coords = []
        recep_coords = []
        lig_coords = []
        oth_coords = []
        dum_atomlist = []
        lig_atomlist = []
        recep_atomlist = []
        oth_atomlist = []
        dum_rsnmlist = []
        recep_rsnmlist = []
        lig_rsnmlist = []
        oth_rsnmlist = []
        dum_rsidlist = []
        recep_rsidlist = []
        lig_rsidlist = []
        oth_rsidlist = []
        dum_chainlist = []
        recep_chainlist = []
        lig_chainlist = []
        oth_chainlist = []
        dum_atom = 0
        lig_atom = 0
        recep_atom = 0
        oth_atom = 0
        total_atom = 0
        resid_lig = 0
        resname_lig = mol

        # Read coordinates for dummy atoms
        if dec_method == "sdr" or dec_method == "exchange":
            for i in range(1, 3):
                shutil.copy("../../build_files/dum" + str(i) + ".pdb", "./")
                with open("dum" + str(i) + ".pdb") as dum_in:
                    lines = (line.rstrip() for line in dum_in)
                    lines = list(line for line in lines if line)
                    dum_coords.append(
                        (
                            float(lines[1][30:38].strip()),
                            float(lines[1][38:46].strip()),
                            float(lines[1][46:54].strip()),
                        )
                    )
                    dum_atomlist.append(lines[1][12:16].strip())
                    dum_rsnmlist.append(lines[1][17:20].strip())
                    dum_rsidlist.append(float(lines[1][22:26].strip()))
                    dum_chainlist.append(lines[1][21].strip())
                    dum_atom += 1
                    total_atom += 1
        else:
            for i in range(1, 2):
                shutil.copy("../../build_files/dum" + str(i) + ".pdb", "./")
                with open("dum" + str(i) + ".pdb") as dum_in:
                    lines = (line.rstrip() for line in dum_in)
                    lines = list(line for line in lines if line)
                    dum_coords.append(
                        (
                            float(lines[1][30:38].strip()),
                            float(lines[1][38:46].strip()),
                            float(lines[1][46:54].strip()),
                        )
                    )
                    dum_atomlist.append(lines[1][12:16].strip())
                    dum_rsnmlist.append(lines[1][17:20].strip())
                    dum_rsidlist.append(float(lines[1][22:26].strip()))
                    dum_chainlist.append(lines[1][21].strip())
                    dum_atom += 1
                    total_atom += 1

        # Read coordinates from aligned system
        with open("build-ini.pdb") as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list

        # Reset only receptor, ligand, other molecules
        recep_coords = []
        lig_coords = []
        oth_coords = []

        recep_atomlist = []
        lig_atomlist = []
        oth_atomlist = []

        recep_rsnmlist = []
        lig_rsnmlist = []
        oth_rsnmlist = []

        recep_rsidlist = []
        lig_rsidlist = []
        oth_rsidlist = []

        recep_chainlist = []
        lig_chainlist = []
        oth_chainlist = []

        # Reset counters
        recep_atom = 0
        lig_atom = 0
        oth_atom = 0
        recep_last = 0
        total_atom = dum_atom  # start counting after dummies

        # Parse line-by-line
        for i in range(len(lines)):
            if (lines[i][0:6].strip() == "ATOM") or (lines[i][0:6].strip() == "HETATM"):
                resname = lines[i][17:20].strip()

                if resname == "DUM":
                    continue  # skip dummy atoms

                elif resname == mol:
                    # Ligand atoms
                    lig_coords.append(
                        (
                            float(lines[i][30:38].strip()),
                            float(lines[i][38:46].strip()),
                            float(lines[i][46:54].strip()),
                        )
                    )
                    lig_atomlist.append(lines[i][12:16].strip())
                    lig_rsnmlist.append(resname)
                    lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    lig_chainlist.append(lines[i][21].strip())
                    lig_atom += 1
                    total_atom += 1

                elif resname in other_mol or resname == "WAT":
                    # Other molecules (cofactors, waters)
                    oth_coords.append(
                        (
                            float(lines[i][30:38].strip()),
                            float(lines[i][38:46].strip()),
                            float(lines[i][46:54].strip()),
                        )
                    )
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(resname)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1

                else:
                    # Receptor (protein)
                    recep_coords.append(
                        (
                            float(lines[i][30:38].strip()),
                            float(lines[i][38:46].strip()),
                            float(lines[i][46:54].strip()),
                        )
                    )
                    recep_atomlist.append(lines[i][12:16].strip())
                    recep_rsnmlist.append(resname)
                    recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    recep_chainlist.append(lines[i][21].strip())
                    recep_last = int(lines[i][22:26].strip())
                    recep_atom += 1
                    total_atom += 1

        # Merge all coordinates and lists
        coords = dum_coords + recep_coords + lig_coords + oth_coords
        atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
        resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
        resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
        chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist

        # Ligand first atom resid
        lig_resid = recep_last + dum_atom
        oth_tmp = "None"

        # Get coordinates from reference ligand
        if comp == "x":
            shutil.copy("../../exchange_files/%s.pdb" % molr.lower(), "./")
            shutil.copy("../../exchange_files/anchors-" + poser + ".txt", "./")
            shutil.copy("../../exchange_files/vac_ligand.pdb", "./vac_reference.pdb")
            shutil.copy(
                "../../exchange_files/vac_ligand.prmtop", "./vac_reference.prmtop"
            )
            shutil.copy(
                "../../exchange_files/vac_ligand.inpcrd", "./vac_reference.inpcrd"
            )
            shutil.copy(
                "../../exchange_files/fe-%s.pdb" % molr.lower(), "./build-ref.pdb"
            )

            ref_lig_coords = []
            ref_lig_atomlist = []
            ref_lig_rsnmlist = []
            ref_lig_rsidlist = []
            ref_lig_chainlist = []
            ref_lig_atom = 0
            ref_resid_lig = 0
            resname_lig = molr

            # Read coordinates from reference system
            with open("build-ref.pdb") as f_in:
                lines = (line.rstrip() for line in f_in)
                lines = list(
                    line for line in lines if line
                )  # Non-blank lines in a list

            # Count atoms of the system
            for i in range(0, len(lines)):
                if (lines[i][0:6].strip() == "ATOM") or (
                    lines[i][0:6].strip() == "HETATM"
                ):
                    if lines[i][17:20].strip() == molr:
                        ref_lig_coords.append(
                            (
                                float(lines[i][30:38].strip()),
                                float(lines[i][38:46].strip()),
                                float(lines[i][46:54].strip()),
                            )
                        )
                        ref_lig_atomlist.append(lines[i][12:16].strip())
                        ref_lig_rsnmlist.append(lines[i][17:20].strip())
                        ref_lig_rsidlist.append(
                            float(lines[i][22:26].strip()) + dum_atom - 1
                        )
                        ref_lig_chainlist.append(lines[i][21].strip())
                        ref_lig_atom += 1

        # Write the new pdb file

        build_file = open("build.pdb", "w")

        # Positions for the dummy atoms
        for i in range(0, dum_atom):
            build_file.write(
                "%-4s  %5s %-4s %3s  %4.0f    "
                % ("ATOM", i + 1, atom_namelist[i], resname_list[i], resid_list[i])
            )
            build_file.write(
                "%8.3f%8.3f%8.3f"
                % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2]))
            )
            build_file.write("%6.2f%6.2f\n" % (0, 0))
            build_file.write("TER\n")

        # Positions of the receptor atoms
        for i in range(dum_atom, dum_atom + recep_atom):
            build_file.write(
                "%-4s  %5s %-4s %3s  %4.0f    "
                % ("ATOM", i + 1, atom_namelist[i], resname_list[i], resid_list[i])
            )
            build_file.write(
                "%8.3f%8.3f%8.3f"
                % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2]))
            )

            build_file.write("%6.2f%6.2f\n" % (0, 0))
            j = i + 2 - dum_atom
            if j in ter_atom:
                build_file.write("TER\n")

        # Positions of the ligand atoms
        for i in range(dum_atom + recep_atom, dum_atom + recep_atom + lig_atom):
            if comp == "n":
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, atom_namelist[i], mol, float(lig_resid))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    % (
                        float(coords[i][0] + sdr_dx),
                        float(coords[i][1] + sdr_dy),
                        float(coords[i][2] + sdr_dz),
                    )
                )
                build_file.write("%6.2f%6.2f\n" % (0, 0))
            elif comp != "r":
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, atom_namelist[i], mol, float(lig_resid))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2]))
                )
                build_file.write("%6.2f%6.2f\n" % (0, 0))

        if comp != "r":
            build_file.write("TER\n")

        # Extra guests for decoupling

        build_file = open("build.pdb", "a")
        if comp == "e":
            # ---- COPY #2 (pocket) : resid = lig_resid + 1 (Don't move along SDR distances and axis) ----
            for i in range(0, lig_atom):
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, lig_atomlist[i], mol, float(lig_resid + 1))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    # NEW (two occurrences)
                    % (
                        float(lig_coords[i][0]),
                        float(lig_coords[i][1]),
                        float(lig_coords[i][2]),
                    )
                )

                build_file.write("%6.2f%6.2f\n" % (0, 0))
            build_file.write("TER\n")

            if dec_method == "sdr" or dec_method == "exchange":
                # ---- COPY #3 (bulk) : resid = lig_resid + 2  ----------------------
                for i in range(0, lig_atom):
                    build_file.write(
                        "%-4s  %5s %-4s %3s  %4.0f    "
                        % ("ATOM", i + 1, lig_atomlist[i], mol, float(lig_resid + 2))
                    )
                    build_file.write(
                        "%8.3f%8.3f%8.3f"
                        % (
                            float(lig_coords[i][0] + sdr_dx),
                            float(lig_coords[i][1] + sdr_dy),
                            float(lig_coords[i][2] + sdr_dz),
                        )
                    )

                    build_file.write("%6.2f%6.2f\n" % (0, 0))
                build_file.write("TER\n")
                # -------------- COPY #4 (bulk) : resid = lig_resid + 3  -------------
                for i in range(0, lig_atom):
                    build_file.write(
                        "%-4s  %5s %-4s %3s  %4.0f    "
                        % ("ATOM", i + 1, lig_atomlist[i], mol, float(lig_resid + 3))
                    )
                    build_file.write(
                        "%8.3f%8.3f%8.3f"
                        % (
                            float(lig_coords[i][0] + sdr_dx),
                            float(lig_coords[i][1] + sdr_dy),
                            float(lig_coords[i][2] + sdr_dz),
                        )
                    )

                    build_file.write("%6.2f%6.2f\n" % (0, 0))
                build_file.write("TER\n")
            print("Creating new system for decharging...")
        if comp == "v" and (dec_method == "sdr" or dec_method == "exchange"):
            for i in range(0, lig_atom):
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, lig_atomlist[i], mol, float(lig_resid + 1))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    % (
                        float(lig_coords[i][0] + sdr_dx),
                        float(lig_coords[i][1] + sdr_dy),
                        float(lig_coords[i][2] + sdr_dz),
                    )
                )

                build_file.write("%6.2f%6.2f\n" % (0, 0))
            build_file.write("TER\n")
            print("Creating new system for vdw decoupling...")

        # Other ligands for relative calculations
        if comp == "x":
            for i in range(0, ref_lig_atom):
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, ref_lig_atomlist[i], molr, float(lig_resid + 1))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    % (
                        float(lig_coords[i][0] + sdr_dx),
                        float(lig_coords[i][1] + sdr_dy),
                        float(lig_coords[i][2] + sdr_dz),
                    )
                )

                build_file.write("%6.2f%6.2f\n" % (0, 0))
            build_file.write("TER\n")
            for i in range(0, ref_lig_atom):
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, ref_lig_atomlist[i], molr, float(lig_resid + 2))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    % (
                        float(ref_lig_coords[i][0]),
                        float(ref_lig_coords[i][1]),
                        float(ref_lig_coords[i][2]),
                    )
                )

                build_file.write("%6.2f%6.2f\n" % (0, 0))
            build_file.write("TER\n")
            for i in range(0, lig_atom):
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, lig_atomlist[i], mol, float(lig_resid + 3))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    % (
                        float(lig_coords[i][0] + sdr_dx),
                        float(lig_coords[i][1] + sdr_dy),
                        float(lig_coords[i][2] + sdr_dz),
                    )
                )

                build_file.write("%6.2f%6.2f\n" % (0, 0))
            build_file.write("TER\n")
            print("Creating new system for vdw ligand exchange...")

        # Positions of the other atoms
        for i in range(0, oth_atom):
            if oth_rsidlist[i] != oth_tmp:
                build_file.write("TER\n")
            oth_tmp = oth_rsidlist[i]
            build_file.write(
                "%-4s  %5s %-4s %3s  %4.0f    "
                % ("ATOM", i + 1, oth_atomlist[i], oth_rsnmlist[i], oth_rsidlist[i])
            )
            build_file.write(
                "%8.3f%8.3f%8.3f"
                % (
                    float(oth_coords[i][0]),
                    float(oth_coords[i][1]),
                    float(oth_coords[i][2]),
                )
            )

            build_file.write("%6.2f%6.2f\n" % (0, 0))

        build_file.write("TER\n")
        build_file.write("END\n")
        build_file.close()

        # Write dry build file

        with open("build.pdb") as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        with open("./build-dry.pdb", "w") as outfile:
            for i in range(0, len(lines)):
                if lines[i][17:20].strip() == "WAT":
                    break
                outfile.write(lines[i] + "\n")

        outfile.close()

        if comp == "f" or comp == "w" or comp == "c":
            # Create system with one or two ligands
            build_file = open("build.pdb", "w")
            for i in range(0, lig_atom):
                build_file.write(
                    "%-4s  %5s %-4s %3s  %4.0f    "
                    % ("ATOM", i + 1, lig_atomlist[i], mol, float(lig_resid))
                )
                build_file.write(
                    "%8.3f%8.3f%8.3f"
                    % (
                        float(lig_coords[i][0]),
                        float(lig_coords[i][1]),
                        float(lig_coords[i][2]),
                    )
                )

                build_file.write("%6.2f%6.2f\n" % (0, 0))
            build_file.write("TER\n")
            if comp == "f":
                for i in range(0, lig_atom):
                    build_file.write(
                        "%-4s  %5s %-4s %3s  %4.0f    "
                        % ("ATOM", i + 1, lig_atomlist[i], mol, float(lig_resid + 1))
                    )
                    build_file.write(
                        "%8.3f%8.3f%8.3f"
                        % (
                            float(lig_coords[i][0]),
                            float(lig_coords[i][1]),
                            float(lig_coords[i][2]),
                        )
                    )

                    build_file.write("%6.2f%6.2f\n" % (0, 0))
                build_file.write("TER\n")
            build_file.write("END\n")
            build_file.close()

            ###Insert TER after the first iNOS chain
            if second_cyp_dec is not None and first_cyp_dec is not None:
                prot_len_dec = int(second_cyp_dec) - int(first_cyp_dec)
                # count dummies at start of the PDB (e.g., DUM)
                n_dum = count_leading_resname("build.pdb", "DUM")
                # chain A ends at (prot_len + n_dum) in the PDB numbering
                insert_after = prot_len_dec + n_dum

                insert_ter_after_resnum("build.pdb", insert_after)
                insert_ter_after_resnum("build-dry.pdb", insert_after)

            # insert_ter_after_resnum("build.pdb", 422)
            # insert_ter_after_resnum("build-dry.pdb", 422)

            ###TLEAP for VAC.pdb and parametrs
            shutil.copy("./build.pdb", "./%s.pdb" % mol.lower())

            tleap_vac = open("tleap_vac.in", "w")
            tleap_vac.write("source leaprc." + ligand_ff + "\n\n")
            tleap_vac.write("# Load the necessary parameters\n")
            tleap_vac.write("# Load the CYP library\n")
            tleap_vac.write("CYP = loadmol2 cyp.mol2\n")
            tleap_vac.write(
                "# make CYP behave like a residue template with head/tail\n"
            )
            tleap_vac.write("set CYP head N\n")
            tleap_vac.write("set CYP tail C\n")
            tleap_vac.write("# Load the ligand parameters\n")
            tleap_vac.write("loadamberparams %s.frcmod\n" % (mol.lower()))
            tleap_vac.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
            tleap_vac.write("model = loadpdb %s.pdb\n\n" % (mol.lower()))
            # ----- Chain A -----
            tleap_vac.write("# connect the peptide bonds (chain A)\n")
            tleap_vac.write(
                f"bond model.{first_cyp_previous_dec}.C  model.{first_cyp_dec}.N\n"
            )
            tleap_vac.write(
                f"bond model.{first_cyp_dec}.C  model.{first_cyp_next_dec}.N\n"
            )

            tleap_vac.write("# Create the FE-SG bond (chain A)\n")
            tleap_vac.write(f"bond model.{first_cyp_dec}.SG model.{heme_1}.FE\n")

            # ----- Chain B (if present) -----
            if second_cyp_dec is not None:
                tleap_vac.write("# connect the peptide bonds (chain B)\n")
                tleap_vac.write(
                    f"bond model.{second_cyp_previous_dec}.C  model.{second_cyp_dec}.N\n"
                )
                tleap_vac.write(
                    f"bond model.{second_cyp_dec}.C  model.{second_cyp_next_dec}.N\n"
                )

                tleap_vac.write("# Create the FE-SG bond (chain B)\n")
                tleap_vac.write(f"bond model.{second_cyp_dec}.SG model.{heme_2}.FE\n")
            # ----- Check Model and Write PArameters -----
            tleap_vac.write("check model\n")
            tleap_vac.write("savepdb model vac.pdb\n")
            tleap_vac.write("saveamberparm model vac.prmtop vac.inpcrd\n")
            tleap_vac.write("quit\n\n")
            tleap_vac.close()

            p = sp.call("tleap -s -f tleap_vac.in > tleap_vac.log", shell=True)
    # Copy system from other attach component
    if int(win) == 0 and altm != "None":
        for file in glob.glob("../" + altm + "/*"):
            basename = os.path.basename(file)
            dst = os.path.join("./", basename)
            if os.path.abspath(file) != os.path.abspath(dst):
                shutil.copy(file, dst)
        return "altm"
    # Copy system initial window
    if win != 0:
        for file in glob.glob("../" + comp + "00/*"):
            shutil.copy(file, "./")

    return "all"


##################Create_Box_HEME Use only with build_equil (1DUM)##################################


def create_box_cyp_equil(
    comp,
    hmr,
    pose,
    mol,
    molr,
    num_waters,
    water_model,
    ion_def,
    neut,
    buffer_x,
    buffer_y,
    buffer_z,
    stage,
    ntpr,
    ntwr,
    ntwe,
    ntwx,
    cut,
    gamma_ln,
    barostat,
    receptor_ff,
    ligand_ff,
    dt,
    dec_method,
    other_mol,
    solv_shell,
    first_cyp_equil=None,
    second_cyp_equil=None,
    first_cyp_next_equil=None,
    second_cyp_next_equil=None,
    first_cyp_previous_equil=None,
    second_cyp_previous_equil=None,
    heme_1=None,
    heme_2=None,
):

    # Adjust buffers to solvation shell
    if stage == "fe" and solv_shell != 0:
        buffer_x = buffer_x - solv_shell
        buffer_y = buffer_y - solv_shell
        if buffer_z != 0:
            if (
                ((dec_method == "sdr") and (comp == "e" or comp == "v"))
                or comp == "n"
                or comp == "x"
            ):
                buffer_z = buffer_z - (solv_shell / 2)
            else:
                buffer_z = buffer_z - solv_shell

    # Copy and replace simulation files
    if stage != "fe":
        if os.path.exists("amber_files"):
            shutil.rmtree("./amber_files")
        try:
            shutil.copytree("../amber_files", "./amber_files")
        # Directories are the same
        except shutil.Error as e:
            print("Directory not copied. Error: %s" % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print("Directory not copied. Error: %s" % e)
        for dname, dirs, files in os.walk("./amber_files"):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                    s = (
                        s.replace("_step_", dt)
                        .replace("_ntpr_", ntpr)
                        .replace("_ntwr_", ntwr)
                        .replace("_ntwe_", ntwe)
                        .replace("_ntwx_", ntwx)
                        .replace("_cutoff_", cut)
                        .replace("_gamma_ln_", gamma_ln)
                        .replace("_barostat_", barostat)
                        .replace("_receptor_ff_", receptor_ff)
                        .replace("_ligand_ff_", ligand_ff)
                    )
                with open(fpath, "w") as f:
                    f.write(s)
        current_dir = os.getcwd()
        print(current_dir)
        os.chdir(pose)

    # Copy tleap files that are used for restraint generation and analysis
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac_ligand.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap.in")

    # Copy ligand parameter files
    for file in glob.glob("../ff/*"):
        shutil.copy(file, "./")

    ###Insert TER after the first iNOS chain
    if second_cyp_equil is not None and first_cyp_equil is not None:
        prot_len_equil = int(second_cyp_equil) - int(first_cyp_equil)

        # count dummies at start of the PDB (e.g., DUM)
        n_dum = count_leading_resname("build.pdb", "DUM")

        # chain A ends at (prot_len + n_dum) in the PDB numbering
        insert_after = prot_len_equil + n_dum

        insert_ter_after_resnum("build.pdb", insert_after)
        insert_ter_after_resnum("build-dry.pdb", insert_after)

    # Append tleap file for vacuum
    tleap_vac = open("tleap_vac.in", "a")
    tleap_vac.write("# Load the necessary parameters\n")
    for i in range(0, len(other_mol)):
        tleap_vac.write("loadamberparams %s.frcmod\n" % (other_mol[i].lower()))
        tleap_vac.write(
            "%s = loadmol2 %s.mol2\n" % (other_mol[i].upper(), other_mol[i].lower())
        )
    tleap_vac.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_vac.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    if comp == "x":
        tleap_vac.write("loadamberparams %s.frcmod\n" % (molr.lower()))
    if comp == "x":
        tleap_vac.write("%s = loadmol2 %s.mol2\n\n" % (molr.upper(), molr.lower()))
    tleap_vac.write("# Load the water parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_vac.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_vac.write("source leaprc.water.fb3\n\n")
    tleap_vac.write("# Load the necessary parameters\n")
    tleap_vac.write("# Load the CYP library\n")
    tleap_vac.write("CYP = loadmol2 cyp.mol2\n")
    tleap_vac.write("# make CYP behave like a residue template with head/tail\n")
    tleap_vac.write("set CYP head N\n")
    tleap_vac.write("set CYP tail C\n")
    tleap_vac.write("model = loadpdb build-dry.pdb\n\n")
    # ----- Chain A -----
    tleap_vac.write("# connect the peptide bonds (chain A)\n")
    tleap_vac.write(
        f"bond model.{first_cyp_previous_equil}.C  model.{first_cyp_equil}.N\n"
    )
    tleap_vac.write(f"bond model.{first_cyp_equil}.C  model.{first_cyp_next_equil}.N\n")

    tleap_vac.write("# Create the FE-SG bond (chain A)\n")
    tleap_vac.write(f"bond model.{first_cyp_equil}.SG model.{heme_1}.FE\n")

    # ----- Chain B (if present) -----
    if second_cyp_equil is not None:
        tleap_vac.write("# connect the peptide bonds (chain B)\n")
        tleap_vac.write(
            f"bond model.{second_cyp_previous_equil}.C  model.{second_cyp_equil}.N\n"
        )
        tleap_vac.write(
            f"bond model.{second_cyp_equil}.C  model.{second_cyp_next_equil}.N\n"
        )

        tleap_vac.write("# Create the FE-SG bond (chain B)\n")
        tleap_vac.write(f"bond model.{second_cyp_equil}.SG model.{heme_2}.FE\n")
    # ----- Check Model and Write PArameters -----
    tleap_vac.write("check model\n")
    tleap_vac.write("savepdb model vac.pdb\n")
    tleap_vac.write("saveamberparm model vac.prmtop vac.inpcrd\n")
    tleap_vac.write("quit\n")
    tleap_vac.close()

    # Append tleap file for ligand only
    tleap_vac_ligand = open("tleap_vac_ligand.in", "a")
    tleap_vac_ligand.write("# Load the ligand parameters\n")
    tleap_vac_ligand.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_vac_ligand.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    tleap_vac_ligand.write("model = loadpdb %s.pdb\n\n" % (mol.lower()))
    tleap_vac_ligand.write("check model\n")
    tleap_vac_ligand.write("savepdb model vac_ligand.pdb\n")
    tleap_vac_ligand.write("saveamberparm model vac_ligand.prmtop vac_ligand.inpcrd\n")
    tleap_vac_ligand.write("quit\n")
    tleap_vac_ligand.close()

    # Generate complex in vacuum
    p = sp.call("tleap -s -f tleap_vac.in > tleap_vac.log", shell=True)

    # Generate ligand structure in vacuum
    p = sp.call("tleap -s -f tleap_vac_ligand.in > tleap_vac_ligand.log", shell=True)

    # Find out how many cations/anions are needed for neutralization
    neu_cat = 0
    neu_ani = 0
    f = open("tleap_vac.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                neu_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                neu_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Get ligand removed charge when doing LJ calculations
    lig_cat = 0
    lig_ani = 0
    f = open("tleap_vac_ligand.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                lig_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                lig_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Adjust ions for LJ and electrostatic Calculations (avoid neutralizing plasma)
    if (comp == "v" and dec_method == "sdr") or comp == "x":
        charge_neut = neu_cat - neu_ani - 2 * lig_cat + 2 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)
    if comp == "e" and dec_method == "sdr":
        charge_neut = neu_cat - neu_ani - 3 * lig_cat + 3 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)

    # Define volume density for different water models
    ratio = 0.060
    if water_model == "TIP3P":
        water_box = water_model.upper() + "BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    elif water_model == "TIP4PEW":
        water_box = water_model.upper() + "BOX"
    elif water_model == "OPC":
        water_box = water_model.upper() + "BOX"
    elif water_model == "TIP3PF":
        water_box = water_model.upper() + "BOX"

    # Fixed number of water molecules
    if num_waters != 0:

        # Create the first box guess to get the initial number of waters and cross sectional area
        buff = 50.0
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        num_added = scripts.check_tleap()
        cross_area = scripts.cross_sectional_area()

        # First iteration to estimate box volume and number of ions
        res_diff = num_added - num_waters
        buff_diff = res_diff / (ratio * cross_area)
        buff -= buff_diff
        print(buff)
        if buff < 0:
            print(
                "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
            )
            sys.exit(1)
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # box volume already takes into account system shrinking during equilibration
        print(num_cations)

        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0

        # Update target number of residues according to the ion definitions and vacuum waters
        vac_wt = 0
        with open("./build.pdb") as myfile:
            for line in myfile:
                if "WAT" in line and " O " in line:
                    vac_wt += 1
        if neut == "no":
            target_num = int(
                num_waters - neu_cat + neu_ani + 2 * int(num_cations) - vac_wt
            )
        elif neut == "yes":
            target_num = int(num_waters + neu_cat + neu_ani - vac_wt)

        # Define a few parameters for solvation iteration
        buff = 50.0
        count = 0
        max_count = 10
        rem_limit = 16
        factor = 1
        ind = 0.90
        buff_diff = 1.0

        # Iterate to get the correct number of waters
        while num_added != target_num:
            count += 1
            if count > max_count:
                # Try different parameters
                rem_limit += 4
                if ind > 0.5:
                    ind = ind - 0.02
                else:
                    ind = 0.90
                factor = 1
                max_count = max_count + 10
            tleap_remove = None
            # Manually remove waters if inside removal limit
            if num_added > target_num and (num_added - target_num) < rem_limit:
                difference = num_added - target_num
                tleap_remove = [target_num + 1 + i for i in range(difference)]
                scripts.write_tleap(
                    mol,
                    molr,
                    comp,
                    water_model,
                    water_box,
                    buff,
                    buffer_x,
                    buffer_y,
                    other_mol,
                    tleap_remove,
                )
                scripts.check_tleap()
                break
            # Set new buffer size based on chosen water density
            res_diff = num_added - target_num - (rem_limit / 2)
            buff_diff = res_diff / (ratio * cross_area)
            buff -= buff_diff * factor
            if buff < 0:
                print(
                    "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
                )
                sys.exit(1)
            # Set relaxation factor
            factor = ind * factor
            # Get number of waters
            scripts.write_tleap(
                mol,
                molr,
                comp,
                water_model,
                water_box,
                buff,
                buffer_x,
                buffer_y,
                other_mol,
            )
            num_added = scripts.check_tleap()
        print(str(count) + " iterations for fixed water number")
    # Fixed z buffer
    elif buffer_z != 0:
        buff = buffer_z
        tleap_remove = None
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # # box volume already takes into account system shrinking during equilibration
        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0
        print(num_cations)
    ###Insert TER after the first iNOS chain
    if second_cyp_equil is not None and first_cyp_equil is not None:
        prot_len_equil = int(second_cyp_equil) - int(first_cyp_equil)

        # count dummies at start of the PDB (e.g., DUM)
        n_dum = count_leading_resname("build.pdb", "DUM")

        # chain A ends at (prot_len + n_dum) in the PDB numbering
        insert_after = prot_len_equil + n_dum

        insert_ter_after_resnum("build.pdb", insert_after)
        insert_ter_after_resnum("build-dry.pdb", insert_after)

    # insert_ter_after_resnum("build.pdb", 422)
    # insert_ter_after_resnum("build-dry.pdb", 422)

    # Write the final tleap file with the correct system size and removed water molecules
    shutil.copy("tleap.in", "tleap_solvate.in")
    tleap_solvate = open("tleap_solvate.in", "a")
    tleap_solvate.write("# Load the water and jc ion parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_solvate.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_solvate.write("source leaprc.water.fb3\n\n")
    for i in range(0, len(other_mol)):
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (other_mol[i].lower()))
        tleap_solvate.write(
            "%s = loadmol2 %s.mol2\n" % (other_mol[i].upper(), other_mol[i].lower())
        )
    tleap_solvate.write("# Load the necessary parameters\n")
    tleap_solvate.write("# Load the CYP library\n")
    tleap_solvate.write("CYP = loadmol2 cyp.mol2\n")
    tleap_solvate.write("# make CYP behave like a residue template with head/tail\n")
    tleap_solvate.write("set CYP head N\n")
    tleap_solvate.write("set CYP tail C\n")
    tleap_solvate.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    if comp == "x":
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (molr.lower()))
    if comp == "x":
        tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (molr.upper(), molr.lower()))
    tleap_solvate.write("model = loadpdb build.pdb\n\n")
    # ----- Chain A -----
    tleap_solvate.write("# connect the peptide bonds (chain A)\n")
    tleap_solvate.write(
        f"bond model.{first_cyp_previous_equil}.C  model.{first_cyp_equil}.N\n"
    )
    tleap_solvate.write(
        f"bond model.{first_cyp_equil}.C  model.{first_cyp_next_equil}.N\n"
    )

    tleap_solvate.write("# Create the FE-SG bond (chain A)\n")
    tleap_solvate.write(f"bond model.{first_cyp_equil}.SG model.{heme_1}.FE\n")

    # ----- Chain B (if present) -----
    if second_cyp_equil is not None:
        tleap_solvate.write("# connect the peptide bonds (chain B)\n")
        tleap_solvate.write(
            f"bond model.{second_cyp_previous_equil}.C  model.{second_cyp_equil}.N\n"
        )
        tleap_solvate.write(
            f"bond model.{second_cyp_equil}.C  model.{second_cyp_next_equil}.N\n"
        )

        tleap_solvate.write("# Create the FE-SG bond (chain B)\n")
        tleap_solvate.write(f"bond model.{second_cyp_equil}.SG model.{heme_2}.FE\n")
    # ----- Check Model and Write PArameters -----
    tleap_solvate.write("desc model.HEM.FE\n")
    tleap_solvate.write(f"desc model.{first_cyp_previous_equil}\n")
    tleap_solvate.write(f"desc model.{first_cyp_equil}\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_equil}\n")
    tleap_solvate.write(f"desc model.{first_cyp_previous_equil}.C\n")
    tleap_solvate.write(f"desc model.{first_cyp_equil}.N\n")
    tleap_solvate.write(f"desc model.{first_cyp_equil}.C\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_equil}.N\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_equil}.C\n")
    tleap_solvate.write("check model\n")
    tleap_solvate.write("\n")
    tleap_solvate.write("# Create water box with chosen model\n")
    tleap_solvate.write(
        "solvatebox model "
        + water_box
        + " {"
        + str(buffer_x)
        + " "
        + str(buffer_y)
        + " "
        + str(buff)
        + "}\n\n"
    )
    if tleap_remove is not None:
        tleap_solvate.write("# Remove a few waters manually\n")
        for water in tleap_remove:
            tleap_solvate.write("remove model model.%s\n" % water)
        tleap_solvate.write("\n")
    # Ionize/neutralize system
    if neut == "no":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], num_cat))
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], num_ani))
    elif neut == "yes":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        if neu_cat != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], neu_cat))
        if neu_ani != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], neu_ani))
    tleap_solvate.write("\n")
    tleap_solvate.write("desc model\n")
    tleap_solvate.write("savepdb model full.pdb\n")
    tleap_solvate.write("saveamberparm model full.prmtop full.inpcrd\n")
    tleap_solvate.write("quit")
    tleap_solvate.close()
    p = sp.call("tleap -s -f tleap_solvate.in > tleap_solvate.log", shell=True)

    f = open("tleap_solvate.log", "r")
    for line in f:
        if "Could not open file" in line:
            print("WARNING!!!")
            print(line)
            sys.exit(1)
        if "WARNING: The unperturbed charge of the unit:" in line:
            print(line)
            print("The system is not neutralized properly after solvation")
        if "addIonsRand: Argument #2 is type String must be of type: [unit]" in line:
            print("Aborted.The ion types specified in the input file could be wrong.")
            print(
                "Please check the tleap_solvate.log file, and the ion types specified in the input file.\n"
            )
            sys.exit(1)
    f.close()

    # Remove TER after residue cyp
    # Remove TER after CYP residues in EQUIL numbering
    if first_cyp_equil is not None:
        remove_ter_after_resnum("full.pdb", int(first_cyp_equil))

    if second_cyp_equil is not None:
        remove_ter_after_resnum("full.pdb", int(second_cyp_equil))

    # Apply hydrogen mass repartitioning
    print("Applying mass repartitioning...")
    shutil.copy("../amber_files/parmed-hmr.in", "./")
    sp.call("parmed -O -n -i parmed-hmr.in > parmed-hmr.log", shell=True)

    if stage != "fe":
        os.chdir("../")


##################Create_Box_HEME_FE Use only in Fe stage (2DUM)######################


def create_box_cyp_sdr(
    comp,
    hmr,
    pose,
    mol,
    molr,
    num_waters,
    water_model,
    ion_def,
    neut,
    buffer_x,
    buffer_y,
    buffer_z,
    stage,
    ntpr,
    ntwr,
    ntwe,
    ntwx,
    cut,
    gamma_ln,
    barostat,
    receptor_ff,
    ligand_ff,
    dt,
    dec_method,
    other_mol,
    solv_shell,
    first_cyp_dec=None,
    second_cyp_dec=None,
    first_cyp_next_dec=None,
    second_cyp_next_dec=None,
    first_cyp_previous_dec=None,
    second_cyp_previous_dec=None,
    heme_1=None,
    heme_2=None,
):

    # Adjust buffers to solvation shell
    if stage == "fe" and solv_shell != 0:
        buffer_x = buffer_x - solv_shell
        buffer_y = buffer_y - solv_shell
        if buffer_z != 0:
            if (
                ((dec_method == "sdr") and (comp == "e" or comp == "v"))
                or comp == "n"
                or comp == "x"
            ):
                buffer_z = buffer_z - (solv_shell / 2)
            else:
                buffer_z = buffer_z - solv_shell

    # Copy and replace simulation files
    if stage != "fe":
        if os.path.exists("amber_files"):
            shutil.rmtree("./amber_files")
        try:
            shutil.copytree("../amber_files", "./amber_files")
        # Directories are the same
        except shutil.Error as e:
            print("Directory not copied. Error: %s" % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print("Directory not copied. Error: %s" % e)
        for dname, dirs, files in os.walk("./amber_files"):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                    s = (
                        s.replace("_step_", dt)
                        .replace("_ntpr_", ntpr)
                        .replace("_ntwr_", ntwr)
                        .replace("_ntwe_", ntwe)
                        .replace("_ntwx_", ntwx)
                        .replace("_cutoff_", cut)
                        .replace("_gamma_ln_", gamma_ln)
                        .replace("_barostat_", barostat)
                        .replace("_receptor_ff_", receptor_ff)
                        .replace("_ligand_ff_", ligand_ff)
                    )
                with open(fpath, "w") as f:
                    f.write(s)
        current_dir = os.getcwd()
        print(current_dir)
        os.chdir(pose)

    # Copy tleap files that are used for restraint generation and analysis
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac_ligand.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap.in")

    # Copy ligand parameter files
    for file in glob.glob("../ff/*"):
        shutil.copy(file, "./")
    ###Insert TER after the first iNOS chain
    if second_cyp_dec is not None and first_cyp_dec is not None:
        prot_len_equil = int(second_cyp_dec) - int(first_cyp_dec)

        # count dummies at start of the PDB (e.g., DUM)
        n_dum = count_leading_resname("build.pdb", "DUM")

        # chain A ends at (prot_len + n_dum) in the PDB numbering
        insert_after = prot_len_equil + n_dum

        insert_ter_after_resnum("build.pdb", insert_after)
        insert_ter_after_resnum("build-dry.pdb", insert_after)

    # insert_ter_after_resnum("build.pdb", 423)
    # insert_ter_after_resnum("build-dry.pdb", 423)

    renumber_pdb_residues("build.pdb", "build.pdb")
    renumber_pdb_residues("build-dry.pdb", "build-dry.pdb")

    # Decide which FE residue indices to use for this component (LEaP residue indices as you had)
    # Count UNL residues to decide which heme numbering to use
    res_counts = count_nonprotein_residues("build-dry.pdb")
    print(f"[INFO] Detected {res_counts} UNL residues in build-dry.pdb", flush=True)

    # What you actually need here is: how many ligand residues exist (how many copies)
    num_unl = count_unl_residues("build-dry.pdb", residue_name=str(mol).upper())
    print(
        f"[INFO] Detected {num_unl} ligand residues ({str(mol).upper()}) in build-dry.pdb",
        flush=True,
    )
    delta = int(num_unl) - 1
    if delta < 0:
        delta = 0

    if heme_1 is None and heme_2 is None:
        raise ValueError(
            "create_box_cyp_fe requires one value at least heme_1 (EQUIL/1-dummy heme numbers)"
        )

    # Consider the extra UNL copies for heme residues
    heme_FE_1 = int(heme_1) + delta
    if heme_2 is not None:
        heme_FE_2 = int(heme_2) + delta

    # num_unl is counted in FE build-dry.pdb (inside create_box_cyp_fe)
    # delta = num_unl - 1  # 0, 1, or 3 in your cases
    # heme_FE_1 = heme_1 + delta
    # heme_FE_2 = heme_2 + delta

    # Append tleap file for vacuum
    tleap_vac = open("tleap_vac.in", "a")
    tleap_vac.write("# Load the necessary parameters\n")
    for i in range(0, len(other_mol)):
        tleap_vac.write(f"loadamberparams {other_mol[i].lower()}.frcmod\n")
        tleap_vac.write(
            f"{other_mol[i].upper()} = loadmol2 {other_mol[i].lower()}.mol2\n"
        )

    tleap_vac.write(f"loadamberparams {mol.lower()}.frcmod\n")
    tleap_vac.write(f"{mol.upper()} = loadmol2 {mol.lower()}.mol2\n\n")
    if comp == "x":
        tleap_vac.write(f"loadamberparams {molr.lower()}.frcmod\n")
        tleap_vac.write(f"{molr.upper()} = loadmol2 {molr.lower()}.mol2\n\n")

    tleap_vac.write("# Load the water parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_vac.write(f"source leaprc.water.{water_model.lower()}\n\n")
    else:
        tleap_vac.write("source leaprc.water.fb3\n\n")

    tleap_vac.write("# Load the necessary parameters\n")
    tleap_vac.write("# Load the CYP library\n")
    tleap_vac.write("CYP = loadmol2 cyp.mol2\n")
    tleap_vac.write("# make CYP behave like a residue template with head/tail\n")
    tleap_vac.write("set CYP head N\n")
    tleap_vac.write("set CYP tail C\n")
    tleap_vac.write("model = loadpdb build-dry.pdb\n\n")

    # ----- Chain A -----
    tleap_vac.write("# connect the peptide bonds (chain A)\n")
    tleap_vac.write(f"bond model.{first_cyp_previous_dec}.C  model.{first_cyp_dec}.N\n")
    tleap_vac.write(f"bond model.{first_cyp_dec}.C  model.{first_cyp_next_dec}.N\n")

    tleap_vac.write("# Create the FE-SG bond (chain A)\n")
    tleap_vac.write(f"bond model.{first_cyp_dec}.SG model.{heme_FE_1}.FE\n")

    # ----- Chain B (if present) -----
    if second_cyp_dec is not None:
        tleap_vac.write("# connect the peptide bonds (chain B)\n")
        tleap_vac.write(
            f"bond model.{second_cyp_previous_dec}.C  model.{second_cyp_dec}.N\n"
        )
        tleap_vac.write(
            f"bond model.{second_cyp_dec}.C  model.{second_cyp_next_dec}.N\n"
        )

        tleap_vac.write("# Create the FE-SG bond (chain B)\n")
        tleap_vac.write(f"bond model.{second_cyp_dec}.SG model.{heme_FE_2}.FE\n")
    # ----- Check Model and Write PArameters -----

    tleap_vac.write("check model\n")
    tleap_vac.write("savepdb model vac.pdb\n")
    tleap_vac.write("saveamberparm model vac.prmtop vac.inpcrd\n")
    tleap_vac.write("quit\n")
    tleap_vac.close()

    # Append tleap file for ligand only
    tleap_vac_ligand = open("tleap_vac_ligand.in", "a")
    tleap_vac_ligand.write("# Load the ligand parameters\n")
    tleap_vac_ligand.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_vac_ligand.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    tleap_vac_ligand.write("model = loadpdb %s.pdb\n\n" % (mol.lower()))
    tleap_vac_ligand.write("check model\n")
    tleap_vac_ligand.write("savepdb model vac_ligand.pdb\n")
    tleap_vac_ligand.write("saveamberparm model vac_ligand.prmtop vac_ligand.inpcrd\n")
    tleap_vac_ligand.write("quit\n")
    tleap_vac_ligand.close()

    # Generate complex in vacuum
    p = sp.call("tleap -s -f tleap_vac.in > tleap_vac.log", shell=True)

    # Generate ligand structure in vacuum
    p = sp.call("tleap -s -f tleap_vac_ligand.in > tleap_vac_ligand.log", shell=True)

    # Find out how many cations/anions are needed for neutralization
    neu_cat = 0
    neu_ani = 0
    f = open("tleap_vac.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                neu_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                neu_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Get ligand removed charge when doing LJ calculations
    lig_cat = 0
    lig_ani = 0
    f = open("tleap_vac_ligand.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                lig_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                lig_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Adjust ions for LJ and electrostatic Calculations (avoid neutralizing plasma)
    if (comp == "v" and dec_method == "sdr") or comp == "x":
        charge_neut = neu_cat - neu_ani - 2 * lig_cat + 2 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)
    if comp == "e" and dec_method == "sdr":
        charge_neut = neu_cat - neu_ani - 3 * lig_cat + 3 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)

    # Define volume density for different water models
    ratio = 0.060
    if water_model == "TIP3P":
        water_box = water_model.upper() + "BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    elif water_model == "TIP4PEW":
        water_box = water_model.upper() + "BOX"
    elif water_model == "OPC":
        water_box = water_model.upper() + "BOX"
    elif water_model == "TIP3PF":
        water_box = water_model.upper() + "BOX"

    # Fixed number of water molecules
    if num_waters != 0:

        # Create the first box guess to get the initial number of waters and cross sectional area
        buff = 50.0
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        num_added = scripts.check_tleap()
        cross_area = scripts.cross_sectional_area()

        # First iteration to estimate box volume and number of ions
        res_diff = num_added - num_waters
        buff_diff = res_diff / (ratio * cross_area)
        buff -= buff_diff
        print(buff)
        if buff < 0:
            print(
                "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
            )
            sys.exit(1)
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # box volume already takes into account system shrinking during equilibration
        print(num_cations)

        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0

        # Update target number of residues according to the ion definitions and vacuum waters
        vac_wt = 0
        with open("./build.pdb") as myfile:
            for line in myfile:
                if "WAT" in line and " O " in line:
                    vac_wt += 1
        if neut == "no":
            target_num = int(
                num_waters - neu_cat + neu_ani + 2 * int(num_cations) - vac_wt
            )
        elif neut == "yes":
            target_num = int(num_waters + neu_cat + neu_ani - vac_wt)

        # Define a few parameters for solvation iteration
        buff = 50.0
        count = 0
        max_count = 10
        rem_limit = 16
        factor = 1
        ind = 0.90
        buff_diff = 1.0

        # Iterate to get the correct number of waters
        while num_added != target_num:
            count += 1
            if count > max_count:
                # Try different parameters
                rem_limit += 4
                if ind > 0.5:
                    ind = ind - 0.02
                else:
                    ind = 0.90
                factor = 1
                max_count = max_count + 10
            tleap_remove = None
            # Manually remove waters if inside removal limit
            if num_added > target_num and (num_added - target_num) < rem_limit:
                difference = num_added - target_num
                tleap_remove = [target_num + 1 + i for i in range(difference)]
                scripts.write_tleap(
                    mol,
                    molr,
                    comp,
                    water_model,
                    water_box,
                    buff,
                    buffer_x,
                    buffer_y,
                    other_mol,
                    tleap_remove,
                )
                scripts.check_tleap()
                break
            # Set new buffer size based on chosen water density
            res_diff = num_added - target_num - (rem_limit / 2)
            buff_diff = res_diff / (ratio * cross_area)
            buff -= buff_diff * factor
            if buff < 0:
                print(
                    "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
                )
                sys.exit(1)
            # Set relaxation factor
            factor = ind * factor
            # Get number of waters
            scripts.write_tleap(
                mol,
                molr,
                comp,
                water_model,
                water_box,
                buff,
                buffer_x,
                buffer_y,
                other_mol,
            )
            num_added = scripts.check_tleap()
        print(str(count) + " iterations for fixed water number")
    # Fixed z buffer
    elif buffer_z != 0:
        buff = buffer_z
        tleap_remove = None
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # # box volume already takes into account system shrinking during equilibration
        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0
        print(num_cations)
    ###Insert TER after the first iNOS chain
    if second_cyp_dec is not None and first_cyp_dec is not None:
        prot_len_equil = int(second_cyp_dec) - int(first_cyp_dec)

        # count dummies at start of the PDB (e.g., DUM)
        n_dum = count_leading_resname("build.pdb", "DUM")

        # chain A ends at (prot_len + n_dum) in the PDB numbering
        insert_after = prot_len_equil + n_dum

        insert_ter_after_resnum("build.pdb", insert_after)
        insert_ter_after_resnum("build-dry.pdb", insert_after)

    # insert_ter_after_resnum("build.pdb", 423)
    # insert_ter_after_resnum("build-dry.pdb", 423)

    # Decide which FE residue indices to use for this component (LEaP residue indices as you had)
    # Count UNL residues to decide which heme numbering to use
    res_counts = count_nonprotein_residues("build-dry.pdb")
    print(f"[INFO] Detected {res_counts} UNL residues in build-dry.pdb", flush=True)

    # What you actually need here is: how many ligand residues exist (how many copies)
    num_unl = count_unl_residues("build-dry.pdb", residue_name=str(mol).upper())
    print(
        f"[INFO] Detected {num_unl} ligand residues ({str(mol).upper()}) in build-dry.pdb",
        flush=True,
    )

    # num_unl = count_unl_residues("build-dry.pdb")
    # print(f"[INFO] Detected {num_unl} UNL residues in build-dry.pdb")

    if heme_1 is None and heme_2 is None:
        raise ValueError(
            "create_box_cyp_fe requires one value at least heme_1 (EQUIL/1-dummy heme numbers)"
        )

    delta = int(num_unl) - 1
    if delta < 0:
        delta = 0

    # Consider the extra UNL copies for heme residues
    heme_FE_1 = int(heme_1) + delta
    if heme_2 is not None:
        heme_FE_2 = int(heme_2) + delta

    # Write the final tleap file with the correct system size and removed water molecules
    shutil.copy("tleap.in", "tleap_solvate.in")
    tleap_solvate = open("tleap_solvate.in", "a")
    tleap_solvate.write("# Load the water and jc ion parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_solvate.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_solvate.write("source leaprc.water.fb3\n\n")
    for i in range(0, len(other_mol)):
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (other_mol[i].lower()))
        tleap_solvate.write(
            "%s = loadmol2 %s.mol2\n" % (other_mol[i].upper(), other_mol[i].lower())
        )
    tleap_solvate.write("# Load the necessary parameters\n")
    tleap_solvate.write("# Load the CYP library\n")
    tleap_solvate.write("CYP = loadmol2 cyp.mol2\n")
    tleap_solvate.write("# make CYP behave like a residue template with head/tail\n")
    tleap_solvate.write("set CYP head N\n")
    tleap_solvate.write("set CYP tail C\n")
    tleap_solvate.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    if comp == "x":
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (molr.lower()))
    if comp == "x":
        tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (molr.upper(), molr.lower()))
    tleap_solvate.write("model = loadpdb build.pdb\n\n")
    # ----- Chain A -----
    tleap_solvate.write("# connect the peptide bonds (chain A)\n")
    tleap_solvate.write(
        f"bond model.{first_cyp_previous_dec}.C  model.{first_cyp_dec}.N\n"
    )
    tleap_solvate.write(f"bond model.{first_cyp_dec}.C  model.{first_cyp_next_dec}.N\n")

    tleap_solvate.write("# Create the FE-SG bond (chain A)\n")
    tleap_solvate.write(f"bond model.{first_cyp_dec}.SG model.{heme_FE_1}.FE\n")

    # ----- Chain B (if present) -----
    if second_cyp_dec is not None:
        tleap_solvate.write("# connect the peptide bonds (chain B)\n")
        tleap_solvate.write(
            f"bond model.{second_cyp_previous_dec}.C  model.{second_cyp_dec}.N\n"
        )
        tleap_solvate.write(
            f"bond model.{second_cyp_dec}.C  model.{second_cyp_next_dec}.N\n"
        )

        tleap_solvate.write("# Create the FE-SG bond (chain B)\n")
        tleap_solvate.write(f"bond model.{second_cyp_dec}.SG model.{heme_FE_2}.FE\n")
    # ----- Check Model and Write PArameters -----
    tleap_solvate.write("desc model.HEM.FE\n")
    tleap_solvate.write(f"desc model.{first_cyp_previous_dec}\n")
    tleap_solvate.write(f"desc model.{first_cyp_dec}\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_dec}\n")
    tleap_solvate.write(f"desc model.{first_cyp_previous_dec}.C\n")
    tleap_solvate.write(f"desc model.{first_cyp_dec}.N\n")
    tleap_solvate.write(f"desc model.{first_cyp_dec}.C\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_dec}.N\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_dec}.C\n")
    tleap_solvate.write(f"check model\n")
    tleap_solvate.write("\n")
    tleap_solvate.write("# Create water box with chosen model\n")
    tleap_solvate.write(
        "solvatebox model "
        + water_box
        + " {"
        + str(buffer_x)
        + " "
        + str(buffer_y)
        + " "
        + str(buff)
        + "}\n\n"
    )
    if tleap_remove is not None:
        tleap_solvate.write("# Remove a few waters manually\n")
        for water in tleap_remove:
            tleap_solvate.write("remove model model.%s\n" % water)
        tleap_solvate.write("\n")
    # Ionize/neutralize system
    if neut == "no":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], num_cat))
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], num_ani))
    elif neut == "yes":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        if neu_cat != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], neu_cat))
        if neu_ani != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], neu_ani))
    tleap_solvate.write("\n")
    tleap_solvate.write("desc model\n")
    tleap_solvate.write("savepdb model full.pdb\n")
    tleap_solvate.write("saveamberparm model full.prmtop full.inpcrd\n")
    tleap_solvate.write("quit")
    tleap_solvate.close()
    p = sp.call("tleap -s -f tleap_solvate.in > tleap_solvate.log", shell=True)

    f = open("tleap_solvate.log", "r")
    for line in f:
        if "Could not open file" in line:
            print("WARNING!!!")
            print(line)
            sys.exit(1)
        if "WARNING: The unperturbed charge of the unit:" in line:
            print(line)
            print("The system is not neutralized properly after solvation")
        if "addIonsRand: Argument #2 is type String must be of type: [unit]" in line:
            print("Aborted.The ion types specified in the input file could be wrong.")
            print(
                "Please check the tleap_solvate.log file, and the ion types specified in the input file.\n"
            )
            sys.exit(1)
    f.close()

    # Remove TER after CYP residues in EQUIL numbering
    if first_cyp_dec is not None:
        remove_ter_after_resnum("full.pdb", int(first_cyp_dec))

    if second_cyp_dec is not None:
        remove_ter_after_resnum("full.pdb", int(second_cyp_dec))

    # Remove TER after residue 120
    # remove_ter_after_resnum("full.pdb", 120)
    # remove_ter_after_resnum("full.pdb", 541)

    # Apply hydrogen mass repartitioning
    print("Applying mass repartitioning...")
    shutil.copy("../amber_files/parmed-hmr.in", "./")
    sp.call("parmed -O -n -i parmed-hmr.in > parmed-hmr.log", shell=True)

    if stage != "fe":
        os.chdir("../")


################## Create_Box For DD and SDR ##################################

import os, re, glob, shutil, subprocess as sp, sys

PDB_ATOM_RE = re.compile(r"^(ATOM  |HETATM)")


def _iter_pdb_atoms(pdb_path):
    with open(pdb_path, "r") as f:
        for line in f:
            if PDB_ATOM_RE.match(line):
                resname = line[17:20].strip()
                resid = int(line[22:26].strip())
                atom = line[12:16].strip()
                yield resid, resname, atom


def count_leading_resname(pdb_path, resname):
    """Count consecutive leading residues with given resname (e.g., DUM) by residue index order."""
    # Determine residue order by first occurrence in file (after renumbering, this matches residue IDs)
    seen = []
    for resid, rname, atom in _iter_pdb_atoms(pdb_path):
        if not seen or seen[-1] != resid:
            seen.append(resid)
    # Map resid -> resname (first atom occurrence)
    resid2name = {}
    for resid, rname, atom in _iter_pdb_atoms(pdb_path):
        if resid not in resid2name:
            resid2name[resid] = rname

    # Count from smallest resid upward
    lead = 0
    for resid in sorted(resid2name.keys()):
        if resid2name[resid] == resname:
            lead += 1
        else:
            break
    return lead


def _resname_at_resid(pdb_path, resid):
    for r, rn, atom in _iter_pdb_atoms(pdb_path):
        if r == resid:
            return rn
    return None


def _infer_heme_resids(pdb_path, allowed_resnames=("HEM", "HEO", "HEC", "HEA", "HEB")):
    """
    Infer heme residue indices by searching residues that contain atom name FE
    and have a heme-like residue name.
    Returns sorted unique residue ids.
    """
    hemes = set()
    # Collect per-residue atoms
    resid2 = {}
    for resid, rname, atom in _iter_pdb_atoms(pdb_path):
        if resid not in resid2:
            resid2[resid] = {"resname": rname, "atoms": set()}
        resid2[resid]["atoms"].add(atom)

    for resid, info in resid2.items():
        rn = info["resname"]
        if ("FE" in info["atoms"]) and (rn in allowed_resnames or rn.startswith("HE")):
            hemes.add(resid)

    return sorted(hemes)


def _adjust_index_if_needed(pdb_path, idx, n_dum, expected_resnames=None):
    """
    Given an index passed by user, decide if it already includes dummies or not.
    Strategy:
      - if expected_resnames is provided and resname at idx matches, keep idx
      - else if resname at idx+n_dum matches, use idx+n_dum
      - else keep idx (but it's likely already correct or user used different naming)
    """
    if idx is None:
        return None
    idx = int(idx)

    if expected_resnames:
        rn = _resname_at_resid(pdb_path, idx)
        if rn in expected_resnames:
            return idx
        rn2 = _resname_at_resid(pdb_path, idx + n_dum)
        if rn2 in expected_resnames:
            return idx + n_dum
        # try the other direction (rare, but helps when someone already added shift twice)
        rn3 = _resname_at_resid(pdb_path, max(1, idx - n_dum))
        if rn3 in expected_resnames:
            return max(1, idx - n_dum)

    return idx


def create_box_cyp_fe(
    comp,
    hmr,
    pose,
    mol,
    molr,
    num_waters,
    water_model,
    ion_def,
    neut,
    buffer_x,
    buffer_y,
    buffer_z,
    stage,
    ntpr,
    ntwr,
    ntwe,
    ntwx,
    cut,
    gamma_ln,
    barostat,
    receptor_ff,
    ligand_ff,
    dt,
    dec_method,
    other_mol,
    solv_shell,
    first_cyp_dec=None,
    second_cyp_dec=None,
    first_cyp_next_dec=None,
    second_cyp_next_dec=None,
    first_cyp_previous_dec=None,
    second_cyp_previous_dec=None,
    # heme_1/heme_2 kept only as optional sanity hints; inference is primary
    heme_1=None,
    heme_2=None,
):
    # --- buffer adjust (unchanged) ---
    if stage == "fe" and solv_shell != 0:
        buffer_x = buffer_x - solv_shell
        buffer_y = buffer_y - solv_shell
        if buffer_z != 0:
            if ((dec_method == "sdr") and (comp in ("e", "v"))) or comp in ("n", "x"):
                buffer_z = buffer_z - (solv_shell / 2)
            else:
                buffer_z = buffer_z - solv_shell

    # --- copy/replace amber_files (unchanged) ---
    if stage != "fe":
        if os.path.exists("amber_files"):
            shutil.rmtree("./amber_files")
        shutil.copytree("../amber_files", "./amber_files")
        for dname, dirs, files in os.walk("./amber_files"):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                s = (
                    s.replace("_step_", dt)
                    .replace("_ntpr_", ntpr)
                    .replace("_ntwr_", ntwr)
                    .replace("_ntwe_", ntwe)
                    .replace("_ntwx_", ntwx)
                    .replace("_cutoff_", cut)
                    .replace("_gamma_ln_", gamma_ln)
                    .replace("_barostat_", barostat)
                    .replace("_receptor_ff_", receptor_ff)
                    .replace("_ligand_ff_", ligand_ff)
                )
                with open(fpath, "w") as f:
                    f.write(s)
        os.chdir(pose)

    # Copy tleap files for restraint generation and analysis
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac_ligand.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap.in")

    # Copy ligand parameter files
    for file in glob.glob("../ff/*"):
        shutil.copy(file, "./")

    # ----- TER insertion (only once) -----
    # (your existing helper functions assumed available)
    if second_cyp_dec is not None and first_cyp_dec is not None:
        prot_len_equil = int(second_cyp_dec) - int(first_cyp_dec)
        n_dum_build = count_leading_resname("build.pdb", "DUM")
        insert_after = prot_len_equil + n_dum_build
        insert_ter_after_resnum("build.pdb", insert_after)
        insert_ter_after_resnum("build-dry.pdb", insert_after)

    # Renumber AFTER TER modifications
    renumber_pdb_residues("build.pdb", "build.pdb")
    renumber_pdb_residues("build-dry.pdb", "build-dry.pdb")

    # ----- Infer dummy count from the actual (renumbered) PDB -----
    n_dum = count_leading_resname("build-dry.pdb", "DUM")
    print(f"[INFO] Leading dummy residues (DUM) in build-dry.pdb: {n_dum}", flush=True)

    # ----- Adjust CYP residue indices only if needed -----
    # Expect your modified cysteine residue name to be CYP (or CYS if you didn’t rename)
    expected_cys = {"CYP", "CYS", "CYX", "CYM"}  # widen if needed

    first_cyp_dec = _adjust_index_if_needed(
        "build-dry.pdb", first_cyp_dec, n_dum, expected_cys
    )
    first_cyp_previous_dec = _adjust_index_if_needed(
        "build-dry.pdb", first_cyp_previous_dec, n_dum, None
    )
    first_cyp_next_dec = _adjust_index_if_needed(
        "build-dry.pdb", first_cyp_next_dec, n_dum, None
    )

    if second_cyp_dec is not None:
        second_cyp_dec = _adjust_index_if_needed(
            "build-dry.pdb", second_cyp_dec, n_dum, expected_cys
        )
        second_cyp_previous_dec = _adjust_index_if_needed(
            "build-dry.pdb", second_cyp_previous_dec, n_dum, None
        )
        second_cyp_next_dec = _adjust_index_if_needed(
            "build-dry.pdb", second_cyp_next_dec, n_dum, None
        )

    # ----- Infer heme residue indices directly (NO delta tricks) -----
    hemes = _infer_heme_resids("build-dry.pdb")
    print(f"[INFO] Inferred heme residues in build-dry.pdb: {hemes}", flush=True)

    if len(hemes) == 0:
        raise ValueError(
            "Could not infer heme residue index (no FE atom found in heme-like residue)."
        )
    if second_cyp_dec is None and len(hemes) > 1:
        # pick the first one deterministically; you can tighten this if you want
        heme_FE_1 = hemes[0]
    else:
        heme_FE_1 = hemes[0]

    heme_FE_2 = None
    if second_cyp_dec is not None:
        if len(hemes) < 2:
            raise ValueError("Expected 2 hemes (two chains), but only 1 was inferred.")
        heme_FE_2 = hemes[1]

    # --- write tleap_vac.in using inferred indices ---
    tleap_vac = open("tleap_vac.in", "a")
    tleap_vac.write("# Load the necessary parameters\n")
    for i in range(0, len(other_mol)):
        tleap_vac.write(f"loadamberparams {other_mol[i].lower()}.frcmod\n")
        tleap_vac.write(
            f"{other_mol[i].upper()} = loadmol2 {other_mol[i].lower()}.mol2\n"
        )

    tleap_vac.write(f"loadamberparams {mol.lower()}.frcmod\n")
    tleap_vac.write(f"{mol.upper()} = loadmol2 {mol.lower()}.mol2\n\n")
    if comp == "x":
        tleap_vac.write(f"loadamberparams {molr.lower()}.frcmod\n")
        tleap_vac.write(f"{molr.upper()} = loadmol2 {molr.lower()}.mol2\n\n")

    tleap_vac.write("# Load the water parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_vac.write(f"source leaprc.water.{water_model.lower()}\n\n")
    else:
        tleap_vac.write("source leaprc.water.fb3\n\n")

    tleap_vac.write("# Load the CYP library\n")
    tleap_vac.write("CYP = loadmol2 cyp.mol2\n")
    tleap_vac.write("set CYP head N\n")
    tleap_vac.write("set CYP tail C\n")
    tleap_vac.write("model = loadpdb build-dry.pdb\n\n")

    # Chain A peptide bonds + Fe-SG
    tleap_vac.write("# connect the peptide bonds (chain A)\n")
    tleap_vac.write(f"bond model.{first_cyp_previous_dec}.C  model.{first_cyp_dec}.N\n")
    tleap_vac.write(f"bond model.{first_cyp_dec}.C  model.{first_cyp_next_dec}.N\n")
    tleap_vac.write("# Create the FE-SG bond (chain A)\n")
    tleap_vac.write(f"bond model.{first_cyp_dec}.SG model.{heme_FE_1}.FE\n")

    # Chain B if present
    if second_cyp_dec is not None:
        tleap_vac.write("# connect the peptide bonds (chain B)\n")
        tleap_vac.write(
            f"bond model.{second_cyp_previous_dec}.C  model.{second_cyp_dec}.N\n"
        )
        tleap_vac.write(
            f"bond model.{second_cyp_dec}.C  model.{second_cyp_next_dec}.N\n"
        )
        tleap_vac.write("# Create the FE-SG bond (chain B)\n")
        tleap_vac.write(f"bond model.{second_cyp_dec}.SG model.{heme_FE_2}.FE\n")

    tleap_vac.write("check model\n")
    tleap_vac.write("savepdb model vac.pdb\n")
    tleap_vac.write("saveamberparm model vac.prmtop vac.inpcrd\n")
    tleap_vac.write("quit\n")
    tleap_vac.close()

    # --- keep the rest of your function (tleap_vac_ligand, solvation iteration, ions, etc.) unchanged ---
    # Important: when you later write tleap_solvate.in, use heme_FE_1/heme_FE_2 again (inferred),
    # not any delta-based computation.

    # (the rest of your original create_box_cyp_fe can follow here as-is)
    # Append tleap file for ligand only
    tleap_vac_ligand = open("tleap_vac_ligand.in", "a")
    tleap_vac_ligand.write("# Load the ligand parameters\n")
    tleap_vac_ligand.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_vac_ligand.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    tleap_vac_ligand.write("model = loadpdb %s.pdb\n\n" % (mol.lower()))
    tleap_vac_ligand.write("check model\n")
    tleap_vac_ligand.write("savepdb model vac_ligand.pdb\n")
    tleap_vac_ligand.write("saveamberparm model vac_ligand.prmtop vac_ligand.inpcrd\n")
    tleap_vac_ligand.write("quit\n")
    tleap_vac_ligand.close()

    # Generate complex in vacuum
    p = sp.call("tleap -s -f tleap_vac.in > tleap_vac.log", shell=True)

    # Generate ligand structure in vacuum
    p = sp.call("tleap -s -f tleap_vac_ligand.in > tleap_vac_ligand.log", shell=True)

    # Find out how many cations/anions are needed for neutralization
    neu_cat = 0
    neu_ani = 0
    f = open("tleap_vac.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                neu_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                neu_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Get ligand removed charge when doing LJ calculations
    lig_cat = 0
    lig_ani = 0
    f = open("tleap_vac_ligand.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                lig_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                lig_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Adjust ions for LJ and electrostatic Calculations (avoid neutralizing plasma)
    if (comp == "v" and dec_method == "sdr") or comp == "x":
        charge_neut = neu_cat - neu_ani - 2 * lig_cat + 2 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)
    if comp == "e" and dec_method == "sdr":
        charge_neut = neu_cat - neu_ani - 3 * lig_cat + 3 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)

    # Define volume density for different water models
    ratio = 0.060
    if water_model == "TIP3P":
        water_box = water_model.upper() + "BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    elif water_model == "TIP4PEW":
        water_box = water_model.upper() + "BOX"
    elif water_model == "OPC":
        water_box = water_model.upper() + "BOX"
    elif water_model == "TIP3PF":
        water_box = water_model.upper() + "BOX"

    # Fixed number of water molecules
    if num_waters != 0:

        # Create the first box guess to get the initial number of waters and cross sectional area
        buff = 50.0
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        num_added = scripts.check_tleap()
        cross_area = scripts.cross_sectional_area()

        # First iteration to estimate box volume and number of ions
        res_diff = num_added - num_waters
        buff_diff = res_diff / (ratio * cross_area)
        buff -= buff_diff
        print(buff)
        if buff < 0:
            print(
                "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
            )
            sys.exit(1)
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # box volume already takes into account system shrinking during equilibration
        print(num_cations)

        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0

        # Update target number of residues according to the ion definitions and vacuum waters
        vac_wt = 0
        with open("./build.pdb") as myfile:
            for line in myfile:
                if "WAT" in line and " O " in line:
                    vac_wt += 1
        if neut == "no":
            target_num = int(
                num_waters - neu_cat + neu_ani + 2 * int(num_cations) - vac_wt
            )
        elif neut == "yes":
            target_num = int(num_waters + neu_cat + neu_ani - vac_wt)

        # Define a few parameters for solvation iteration
        buff = 50.0
        count = 0
        max_count = 10
        rem_limit = 16
        factor = 1
        ind = 0.90
        buff_diff = 1.0

        # Iterate to get the correct number of waters
        while num_added != target_num:
            count += 1
            if count > max_count:
                # Try different parameters
                rem_limit += 4
                if ind > 0.5:
                    ind = ind - 0.02
                else:
                    ind = 0.90
                factor = 1
                max_count = max_count + 10
            tleap_remove = None
            # Manually remove waters if inside removal limit
            if num_added > target_num and (num_added - target_num) < rem_limit:
                difference = num_added - target_num
                tleap_remove = [target_num + 1 + i for i in range(difference)]
                scripts.write_tleap(
                    mol,
                    molr,
                    comp,
                    water_model,
                    water_box,
                    buff,
                    buffer_x,
                    buffer_y,
                    other_mol,
                    tleap_remove,
                )
                scripts.check_tleap()
                break
            # Set new buffer size based on chosen water density
            res_diff = num_added - target_num - (rem_limit / 2)
            buff_diff = res_diff / (ratio * cross_area)
            buff -= buff_diff * factor
            if buff < 0:
                print(
                    "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
                )
                sys.exit(1)
            # Set relaxation factor
            factor = ind * factor
            # Get number of waters
            scripts.write_tleap(
                mol,
                molr,
                comp,
                water_model,
                water_box,
                buff,
                buffer_x,
                buffer_y,
                other_mol,
            )
            num_added = scripts.check_tleap()
        print(str(count) + " iterations for fixed water number")
    # Fixed z buffer
    elif buffer_z != 0:
        buff = buffer_z
        tleap_remove = None
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # # box volume already takes into account system shrinking during equilibration
        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0
        print(num_cations)
    ###Insert TER after the first iNOS chain
    if second_cyp_dec is not None and first_cyp_dec is not None:
        prot_len_equil = int(second_cyp_dec) - int(first_cyp_dec)

        # count dummies at start of the PDB (e.g., DUM)
        n_dum = count_leading_resname("build.pdb", "DUM")

        # chain A ends at (prot_len + n_dum) in the PDB numbering
        insert_after = prot_len_equil + n_dum

        insert_ter_after_resnum("build.pdb", insert_after)
        insert_ter_after_resnum("build-dry.pdb", insert_after)

    # insert_ter_after_resnum("build.pdb", 423)
    # insert_ter_after_resnum("build-dry.pdb", 423)

    # Decide which FE residue indices to use for this component (LEaP residue indices as you had)
    # Count UNL residues to decide which heme numbering to use
    res_counts = count_nonprotein_residues("build-dry.pdb")
    print(f"[INFO] Detected {res_counts} UNL residues in build-dry.pdb", flush=True)

    # What you actually need here is: how many ligand residues exist (how many copies)
    num_unl = count_unl_residues("build-dry.pdb", residue_name=str(mol).upper())
    print(
        f"[INFO] Detected {num_unl} ligand residues ({str(mol).upper()}) in build-dry.pdb",
        flush=True,
    )

    # num_unl = count_unl_residues("build-dry.pdb")
    # print(f"[INFO] Detected {num_unl} UNL residues in build-dry.pdb")

    if heme_1 is None and heme_2 is None:
        raise ValueError(
            "create_box_cyp_fe requires one value at least heme_1 (EQUIL/1-dummy heme numbers)"
        )

    delta = int(num_unl) - 1
    if delta < 0:
        delta = 0

    # Consider the extra UNL copies for heme residues
    heme_FE_1 = int(heme_1) + delta
    if heme_2 is not None:
        heme_FE_2 = int(heme_2) + delta

    # Write the final tleap file with the correct system size and removed water molecules
    shutil.copy("tleap.in", "tleap_solvate.in")
    tleap_solvate = open("tleap_solvate.in", "a")
    tleap_solvate.write("# Load the water and jc ion parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_solvate.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_solvate.write("source leaprc.water.fb3\n\n")
    for i in range(0, len(other_mol)):
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (other_mol[i].lower()))
        tleap_solvate.write(
            "%s = loadmol2 %s.mol2\n" % (other_mol[i].upper(), other_mol[i].lower())
        )
    tleap_solvate.write("# Load the necessary parameters\n")
    tleap_solvate.write("# Load the CYP library\n")
    tleap_solvate.write("CYP = loadmol2 cyp.mol2\n")
    tleap_solvate.write("# make CYP behave like a residue template with head/tail\n")
    tleap_solvate.write("set CYP head N\n")
    tleap_solvate.write("set CYP tail C\n")
    tleap_solvate.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    if comp == "x":
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (molr.lower()))
    if comp == "x":
        tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (molr.upper(), molr.lower()))
    tleap_solvate.write("model = loadpdb build.pdb\n\n")
    # ----- Chain A -----
    tleap_solvate.write("# connect the peptide bonds (chain A)\n")
    tleap_solvate.write(
        f"bond model.{first_cyp_previous_dec}.C  model.{first_cyp_dec}.N\n"
    )
    tleap_solvate.write(f"bond model.{first_cyp_dec}.C  model.{first_cyp_next_dec}.N\n")

    tleap_solvate.write("# Create the FE-SG bond (chain A)\n")
    tleap_solvate.write(f"bond model.{first_cyp_dec}.SG model.{heme_FE_1}.FE\n")

    # ----- Chain B (if present) -----
    if second_cyp_dec is not None:
        tleap_solvate.write("# connect the peptide bonds (chain B)\n")
        tleap_solvate.write(
            f"bond model.{second_cyp_previous_dec}.C  model.{second_cyp_dec}.N\n"
        )
        tleap_solvate.write(
            f"bond model.{second_cyp_dec}.C  model.{second_cyp_next_dec}.N\n"
        )

        tleap_solvate.write("# Create the FE-SG bond (chain B)\n")
        tleap_solvate.write(f"bond model.{second_cyp_dec}.SG model.{heme_FE_2}.FE\n")
    # ----- Check Model and Write PArameters -----
    tleap_solvate.write("desc model.HEM.FE\n")
    tleap_solvate.write(f"desc model.{first_cyp_previous_dec}\n")
    tleap_solvate.write(f"desc model.{first_cyp_dec}\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_dec}\n")
    tleap_solvate.write(f"desc model.{first_cyp_previous_dec}.C\n")
    tleap_solvate.write(f"desc model.{first_cyp_dec}.N\n")
    tleap_solvate.write(f"desc model.{first_cyp_dec}.C\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_dec}.N\n")
    tleap_solvate.write(f"desc model.{first_cyp_next_dec}.C\n")
    tleap_solvate.write(f"check model\n")
    tleap_solvate.write("\n")
    tleap_solvate.write("# Create water box with chosen model\n")
    tleap_solvate.write(
        "solvatebox model "
        + water_box
        + " {"
        + str(buffer_x)
        + " "
        + str(buffer_y)
        + " "
        + str(buff)
        + "}\n\n"
    )
    if tleap_remove is not None:
        tleap_solvate.write("# Remove a few waters manually\n")
        for water in tleap_remove:
            tleap_solvate.write("remove model model.%s\n" % water)
        tleap_solvate.write("\n")
    # Ionize/neutralize system
    if neut == "no":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], num_cat))
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], num_ani))
    elif neut == "yes":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        if neu_cat != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], neu_cat))
        if neu_ani != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], neu_ani))
    tleap_solvate.write("\n")
    tleap_solvate.write("desc model\n")
    tleap_solvate.write("savepdb model full.pdb\n")
    tleap_solvate.write("saveamberparm model full.prmtop full.inpcrd\n")
    tleap_solvate.write("quit")
    tleap_solvate.close()
    p = sp.call("tleap -s -f tleap_solvate.in > tleap_solvate.log", shell=True)

    f = open("tleap_solvate.log", "r")
    for line in f:
        if "Could not open file" in line:
            print("WARNING!!!")
            print(line)
            sys.exit(1)
        if "WARNING: The unperturbed charge of the unit:" in line:
            print(line)
            print("The system is not neutralized properly after solvation")
        if "addIonsRand: Argument #2 is type String must be of type: [unit]" in line:
            print("Aborted.The ion types specified in the input file could be wrong.")
            print(
                "Please check the tleap_solvate.log file, and the ion types specified in the input file.\n"
            )
            sys.exit(1)
    f.close()

    # Remove TER after CYP residues in EQUIL numbering
    if first_cyp_dec is not None:
        remove_ter_after_resnum("full.pdb", int(first_cyp_dec))

    if second_cyp_dec is not None:
        remove_ter_after_resnum("full.pdb", int(second_cyp_dec))

    # Remove TER after residue 120
    # remove_ter_after_resnum("full.pdb", 120)
    # remove_ter_after_resnum("full.pdb", 541)

    # Apply hydrogen mass repartitioning
    print("Applying mass repartitioning...")
    shutil.copy("../amber_files/parmed-hmr.in", "./")
    sp.call("parmed -O -n -i parmed-hmr.in > parmed-hmr.log", shell=True)

    if stage != "fe":
        os.chdir("../")


##################Create_Box Original Function##################################


def create_box(
    comp,
    hmr,
    pose,
    mol,
    molr,
    num_waters,
    water_model,
    ion_def,
    neut,
    buffer_x,
    buffer_y,
    buffer_z,
    stage,
    ntpr,
    ntwr,
    ntwe,
    ntwx,
    cut,
    gamma_ln,
    barostat,
    receptor_ff,
    ligand_ff,
    dt,
    dec_method,
    other_mol,
    solv_shell,
):

    # Adjust buffers to solvation shell
    if stage == "fe" and solv_shell != 0:
        buffer_x = buffer_x - solv_shell
        buffer_y = buffer_y - solv_shell
        if buffer_z != 0:
            if (
                ((dec_method == "sdr") and (comp == "e" or comp == "v"))
                or comp == "n"
                or comp == "x"
            ):
                buffer_z = buffer_z - (solv_shell / 2)
            else:
                buffer_z = buffer_z - solv_shell

    # Copy and replace simulation files
    if stage != "fe":
        if os.path.exists("amber_files"):
            shutil.rmtree("./amber_files")
        try:
            shutil.copytree("../amber_files", "./amber_files")
        # Directories are the same
        except shutil.Error as e:
            print("Directory not copied. Error: %s" % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print("Directory not copied. Error: %s" % e)
        for dname, dirs, files in os.walk("./amber_files"):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                    s = (
                        s.replace("_step_", dt)
                        .replace("_ntpr_", ntpr)
                        .replace("_ntwr_", ntwr)
                        .replace("_ntwe_", ntwe)
                        .replace("_ntwx_", ntwx)
                        .replace("_cutoff_", cut)
                        .replace("_gamma_ln_", gamma_ln)
                        .replace("_barostat_", barostat)
                        .replace("_receptor_ff_", receptor_ff)
                        .replace("_ligand_ff_", ligand_ff)
                    )
                with open(fpath, "w") as f:
                    f.write(s)
        current_dir = os.getcwd()
        print(current_dir)
        os.chdir(pose)

    # Copy tleap files that are used for restraint generation and analysis
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap_vac_ligand.in")
    shutil.copy("../amber_files/tleap.in.amber16", "tleap.in")

    # Copy ligand parameter files
    for file in glob.glob("../ff/*"):
        shutil.copy(file, "./")

    # Append tleap file for vacuum
    tleap_vac = open("tleap_vac.in", "a")
    tleap_vac.write("# Load the necessary parameters\n")
    for i in range(0, len(other_mol)):
        tleap_vac.write("loadamberparams %s.frcmod\n" % (other_mol[i].lower()))
        tleap_vac.write(
            "%s = loadmol2 %s.mol2\n" % (other_mol[i].upper(), other_mol[i].lower())
        )
    tleap_vac.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_vac.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    if comp == "x":
        tleap_vac.write("loadamberparams %s.frcmod\n" % (molr.lower()))
    if comp == "x":
        tleap_vac.write("%s = loadmol2 %s.mol2\n\n" % (molr.upper(), molr.lower()))
    tleap_vac.write("# Load the water parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_vac.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_vac.write("source leaprc.water.fb3\n\n")
    tleap_vac.write("model = loadpdb build-dry.pdb\n\n")
    tleap_vac.write("check model\n")
    tleap_vac.write("savepdb model vac.pdb\n")
    tleap_vac.write("saveamberparm model vac.prmtop vac.inpcrd\n")
    tleap_vac.write("quit\n")
    tleap_vac.close()

    # Append tleap file for ligand only
    tleap_vac_ligand = open("tleap_vac_ligand.in", "a")
    tleap_vac_ligand.write("# Load the ligand parameters\n")
    tleap_vac_ligand.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_vac_ligand.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    tleap_vac_ligand.write("model = loadpdb %s.pdb\n\n" % (mol.lower()))
    tleap_vac_ligand.write("check model\n")
    tleap_vac_ligand.write("savepdb model vac_ligand.pdb\n")
    tleap_vac_ligand.write("saveamberparm model vac_ligand.prmtop vac_ligand.inpcrd\n")
    tleap_vac_ligand.write("quit\n")
    tleap_vac_ligand.close()

    # Generate complex in vacuum
    p = sp.call("tleap -s -f tleap_vac.in > tleap_vac.log", shell=True)

    # Generate ligand structure in vacuum
    p = sp.call("tleap -s -f tleap_vac_ligand.in > tleap_vac_ligand.log", shell=True)

    # Find out how many cations/anions are needed for neutralization
    neu_cat = 0
    neu_ani = 0
    f = open("tleap_vac.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                neu_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                neu_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Get ligand removed charge when doing LJ calculations
    lig_cat = 0
    lig_ani = 0
    f = open("tleap_vac_ligand.log", "r")
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip("'\",.:;#()][")) < 0:
                lig_cat = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
            elif float(splitline[6].strip("'\",.:;#()][")) > 0:
                lig_ani = round(
                    float(re.sub("[+-]", "", splitline[6].strip("'\"-,.:;#()][")))
                )
    f.close()

    # Adjust ions for LJ and electrostatic Calculations (avoid neutralizing plasma)
    if (comp == "v" and dec_method == "sdr") or comp == "x":
        charge_neut = neu_cat - neu_ani - 2 * lig_cat + 2 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)
    if comp == "e" and dec_method == "sdr":
        charge_neut = neu_cat - neu_ani - 3 * lig_cat + 3 * lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)

    # Define volume density for different water models
    ratio = 0.060
    if water_model == "TIP3P":
        water_box = water_model.upper() + "BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    elif water_model == "TIP4PEW":
        water_box = water_model.upper() + "BOX"
    elif water_model == "OPC":
        water_box = water_model.upper() + "BOX"
    elif water_model == "TIP3PF":
        water_box = water_model.upper() + "BOX"

    # Fixed number of water molecules
    if num_waters != 0:

        # Create the first box guess to get the initial number of waters and cross sectional area
        buff = 50.0
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        num_added = scripts.check_tleap()
        cross_area = scripts.cross_sectional_area()

        # First iteration to estimate box volume and number of ions
        res_diff = num_added - num_waters
        buff_diff = res_diff / (ratio * cross_area)
        buff -= buff_diff
        print(buff)
        if buff < 0:
            print(
                "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
            )
            sys.exit(1)
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # box volume already takes into account system shrinking during equilibration
        print(num_cations)

        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0

        # Update target number of residues according to the ion definitions and vacuum waters
        vac_wt = 0
        with open("./build.pdb") as myfile:
            for line in myfile:
                if "WAT" in line and " O " in line:
                    vac_wt += 1
        if neut == "no":
            target_num = int(
                num_waters - neu_cat + neu_ani + 2 * int(num_cations) - vac_wt
            )
        elif neut == "yes":
            target_num = int(num_waters + neu_cat + neu_ani - vac_wt)

        # Define a few parameters for solvation iteration
        buff = 50.0
        count = 0
        max_count = 10
        rem_limit = 16
        factor = 1
        ind = 0.90
        buff_diff = 1.0

        # Iterate to get the correct number of waters
        while num_added != target_num:
            count += 1
            if count > max_count:
                # Try different parameters
                rem_limit += 4
                if ind > 0.5:
                    ind = ind - 0.02
                else:
                    ind = 0.90
                factor = 1
                max_count = max_count + 10
            tleap_remove = None
            # Manually remove waters if inside removal limit
            if num_added > target_num and (num_added - target_num) < rem_limit:
                difference = num_added - target_num
                tleap_remove = [target_num + 1 + i for i in range(difference)]
                scripts.write_tleap(
                    mol,
                    molr,
                    comp,
                    water_model,
                    water_box,
                    buff,
                    buffer_x,
                    buffer_y,
                    other_mol,
                    tleap_remove,
                )
                scripts.check_tleap()
                break
            # Set new buffer size based on chosen water density
            res_diff = num_added - target_num - (rem_limit / 2)
            buff_diff = res_diff / (ratio * cross_area)
            buff -= buff_diff * factor
            if buff < 0:
                print(
                    "Not enough water molecules to fill the system in the z direction, please increase the number of water molecules"
                )
                sys.exit(1)
            # Set relaxation factor
            factor = ind * factor
            # Get number of waters
            scripts.write_tleap(
                mol,
                molr,
                comp,
                water_model,
                water_box,
                buff,
                buffer_x,
                buffer_y,
                other_mol,
            )
            num_added = scripts.check_tleap()
        print(str(count) + " iterations for fixed water number")
    # Fixed z buffer
    elif buffer_z != 0:
        buff = buffer_z
        tleap_remove = None
        # Get box volume and number of added ions
        scripts.write_tleap(
            mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol
        )
        box_volume = scripts.box_volume()
        print(box_volume)
        num_cations = round(
            ion_def[2] * 6.02e23 * box_volume * 1e-27
        )  # # box volume already takes into account system shrinking during equilibration
        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0
        print(num_cations)

    # Write the final tleap file with the correct system size and removed water molecules
    shutil.copy("tleap.in", "tleap_solvate.in")
    tleap_solvate = open("tleap_solvate.in", "a")
    tleap_solvate.write("# Load the water and jc ion parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_solvate.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_solvate.write("source leaprc.water.fb3\n\n")
    for i in range(0, len(other_mol)):
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (other_mol[i].lower()))
        tleap_solvate.write(
            "%s = loadmol2 %s.mol2\n" % (other_mol[i].upper(), other_mol[i].lower())
        )
    tleap_solvate.write("# Load the necessary parameters\n")
    tleap_solvate.write("# Load the CYP library\n")
    tleap_solvate.write("CYP = loadmol2 cyp.mol2\n")
    tleap_solvate.write("# make CYP behave like a residue template with head/tail\n")
    tleap_solvate.write("set CYP head N\n")
    tleap_solvate.write("set CYP tail C\n")
    tleap_solvate.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    if comp == "x":
        tleap_solvate.write("loadamberparams %s.frcmod\n" % (molr.lower()))
    if comp == "x":
        tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (molr.upper(), molr.lower()))
    tleap_solvate.write("model = loadpdb build.pdb\n\n")
    tleap_solvate.write("\n")
    tleap_solvate.write("desc model.HEM.FE\n")
    tleap_solvate.write("desc model.118\n")
    tleap_solvate.write("desc model.119\n")
    tleap_solvate.write("desc model.120\n")
    tleap_solvate.write("desc model.118.C\n")
    tleap_solvate.write("desc model.119.N\n")
    tleap_solvate.write("desc model.119.C\n")
    tleap_solvate.write("desc model.120.N\n")
    tleap_solvate.write("desc model.120.C\n")
    tleap_solvate.write("check model\n")
    tleap_solvate.write("\n")
    tleap_solvate.write("# Create water box with chosen model\n")
    tleap_solvate.write(
        "solvatebox model "
        + water_box
        + " {"
        + str(buffer_x)
        + " "
        + str(buffer_y)
        + " "
        + str(buff)
        + "}\n\n"
    )
    if tleap_remove is not None:
        tleap_solvate.write("# Remove a few waters manually\n")
        for water in tleap_remove:
            tleap_solvate.write("remove model model.%s\n" % water)
        tleap_solvate.write("\n")
    # Ionize/neutralize system
    if neut == "no":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], num_cat))
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], num_ani))
    elif neut == "yes":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        if neu_cat != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], neu_cat))
        if neu_ani != 0:
            tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[1], neu_ani))
    tleap_solvate.write("\n")
    tleap_solvate.write("desc model\n")
    tleap_solvate.write("savepdb model full.pdb\n")
    tleap_solvate.write("saveamberparm model full.prmtop full.inpcrd\n")
    tleap_solvate.write("quit")
    tleap_solvate.close()
    p = sp.call("tleap -s -f tleap_solvate.in > tleap_solvate.log", shell=True)

    f = open("tleap_solvate.log", "r")
    for line in f:
        if "Could not open file" in line:
            print("WARNING!!!")
            print(line)
            sys.exit(1)
        if "WARNING: The unperturbed charge of the unit:" in line:
            print(line)
            print("The system is not neutralized properly after solvation")
        if "addIonsRand: Argument #2 is type String must be of type: [unit]" in line:
            print("Aborted.The ion types specified in the input file could be wrong.")
            print(
                "Please check the tleap_solvate.log file, and the ion types specified in the input file.\n"
            )
            sys.exit(1)
    f.close()

    # remove spurious TER between CYP 119 and ILE 120
    # Remove TER after CYP residues in EQUIL numbering
    # if first_cyp_equil is not None:
    #    remove_ter_after_resnum("full.pdb", int(first_cyp_equil))

    # if num_chains == 2 and second_cyp_equil is not None:
    #    remove_ter_after_resnum("full.pdb", int(second_cyp_equil))

    # remove_ter_after_resnum("full.pdb", 119)

    # Apply hydrogen mass repartitioning
    print("Applying mass repartitioning...")
    shutil.copy("../amber_files/parmed-hmr.in", "./")
    sp.call("parmed -O -n -i parmed-hmr.in > parmed-hmr.log", shell=True)

    if stage != "fe":
        os.chdir("../")


######################################################################################


def ligand_box(mol, lig_buffer, water_model, neut, ion_def, comp, ligand_ff):
    # Define volume density for different water models
    if water_model == "TIP3P":
        water_box = water_model.upper() + "BOX"
    elif water_model == "SPCE":
        water_box = "SPCBOX"
    elif water_model == "TIP4PEW":
        water_box = water_model.upper() + "BOX"
    elif water_model == "OPC":
        water_box = water_model.upper() + "BOX"
    elif water_model == "TIP3PF":
        water_box = water_model.upper() + "BOX"

    # Copy ligand parameter files
    for file in glob.glob("../../ff/%s.*" % mol.lower()):
        shutil.copy(file, "./")

    # Write and run preliminary tleap file
    tleap_solvate = open("tmp_tleap.in", "w")
    tleap_solvate.write("source leaprc." + ligand_ff + "\n\n")
    tleap_solvate.write("# Load the ligand parameters\n")
    tleap_solvate.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    tleap_solvate.write("model = loadpdb %s.pdb\n\n" % (mol.lower()))
    tleap_solvate.write("# Load the water and jc ion parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_solvate.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_solvate.write("source leaprc.water.fb3\n\n")
    tleap_solvate.write("check model\n")
    tleap_solvate.write("savepdb model vac.pdb\n")
    tleap_solvate.write("saveamberparm model vac.prmtop vac.inpcrd\n\n")
    tleap_solvate.write("# Create water box with chosen model\n")
    tleap_solvate.write(
        "solvatebox model " + water_box + " " + str(lig_buffer) + "\n\n"
    )
    tleap_solvate.write("quit\n")
    tleap_solvate.close()

    # Get box volume and number of added ions
    box_volume = scripts.box_volume()
    print(box_volume)
    num_cations = round(
        ion_def[2] * 6.02e23 * box_volume * 1e-27
    )  # box volume already takes into account system shrinking during equilibration
    print(num_cations)

    # Write and run tleap file
    tleap_solvate = open("tleap_solvate.in", "a")
    tleap_solvate.write("source leaprc." + ligand_ff + "\n\n")
    tleap_solvate.write("# Load the ligand parameters\n")
    tleap_solvate.write("loadamberparams %s.frcmod\n" % (mol.lower()))
    tleap_solvate.write("%s = loadmol2 %s.mol2\n\n" % (mol.upper(), mol.lower()))
    tleap_solvate.write("model = loadpdb %s.pdb\n\n" % (mol.lower()))
    tleap_solvate.write("# Load the water and jc ion parameters\n")
    if water_model.lower() != "tip3pf":
        tleap_solvate.write("source leaprc.water.%s\n\n" % (water_model.lower()))
    else:
        tleap_solvate.write("source leaprc.water.fb3\n\n")
    tleap_solvate.write("check model\n")
    tleap_solvate.write("savepdb model vac.pdb\n")
    tleap_solvate.write("saveamberparm model vac.prmtop vac.inpcrd\n\n")
    tleap_solvate.write("# Create water box with chosen model\n")
    tleap_solvate.write(
        "solvatebox model " + water_box + " " + str(lig_buffer) + "\n\n"
    )
    if neut == "no":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        tleap_solvate.write("addionsrand model %s %d\n" % (ion_def[0], num_cations))
        tleap_solvate.write("addionsrand model %s 0\n" % (ion_def[1]))
    elif neut == "yes":
        tleap_solvate.write("# Add ions for neutralization/ionization\n")
        tleap_solvate.write("addionsrand model %s 0\n" % (ion_def[0]))
        tleap_solvate.write("addionsrand model %s 0\n" % (ion_def[1]))
    tleap_solvate.write("\n")
    tleap_solvate.write("desc model\n")
    tleap_solvate.write("savepdb model full.pdb\n")
    tleap_solvate.write("saveamberparm model full.prmtop full.inpcrd\n")
    tleap_solvate.write("quit\n")
    tleap_solvate.close()
    p = sp.call("tleap -s -f tleap_solvate.in > tleap_solvate.log", shell=True)

    # Apply hydrogen mass repartitioning
    print("Applying mass repartitioning...")
    shutil.copy("../amber_files/parmed-hmr.in", "./")
    sp.call("parmed -O -n -i parmed-hmr.in > parmed-hmr.log", shell=True)

    # Copy a few files for consistency
    if comp != "f" and comp != "w":
        shutil.copy("./vac.pdb", "./vac_ligand.pdb")
        shutil.copy("./vac.prmtop", "./vac_ligand.prmtop")


###################################################################################################


###################################################################################################
