#!/usr/bin/env python3

import glob as glob
import os as os
import re
import shutil as shutil
import signal as signal
import subprocess as sp
import sys as sys
from lib import build
from lib import scripts
from lib import setup
from lib import analysis
import numpy as np


# ----------------------------
# Heme protein residue mapping
# ----------------------------
# in ORIGINAL protein residue numbering (pre-AMBER renumbering).
# After setup, protein starts at residue 1.
# Additionally, equil has +1 dummy residue; FE/decouple has +2 dummy residues.

import re

_RES_ATOM_RE = re.compile(r"^:(\d+)(@.+)?$")  # supports ":352@CA" or ":352"


def _map_res_orig_to_amber(orig_res: int, prot_shift: int) -> int:
    mapped = orig_res - prot_shift
    if mapped < 1:
        raise ValueError(
            f"Residue mapping produced <1: orig={orig_res} shift={prot_shift} -> {mapped}"
        )
    return mapped


def map_mask(mask: str, prot_shift: int, add_dummy: int = 0) -> str:
    mask = str(mask).strip()
    m = _RES_ATOM_RE.match(mask)
    if not m:
        raise ValueError(f"Unsupported mask format (expected ':<res>@<atom>'): {mask}")
    orig_res = int(m.group(1))
    atom_part = m.group(2) or ""
    amber_res = _map_res_orig_to_amber(orig_res, prot_shift) + add_dummy
    return f":{amber_res}{atom_part}"


def map_res_list(res_list, prot_shift: int, add_dummy: int = 0):
    # map a list like [379, 400] or an int
    out = []
    for r in res_list:
        out.append(_map_res_orig_to_amber(int(r), prot_shift) + add_dummy)
    return out


def log(msg):
    print(msg, flush=True)


######reading the input parameters####


ion_def = []
poses_list = []
ligand_list = []
poses_def = []
release_eq = []
attach_rest = []
lambdas = []
weights = []
components = []
aa1_poses = []
aa2_poses = []
other_mol = []
bb_start = []
bb_end = []
mols = []
celp_st = []

# Defaults

a_steps1 = 0
a_steps2 = 0
l_steps1 = 0
l_steps2 = 0
t_steps1 = 0
t_steps2 = 0
m_steps1 = 0
m_steps2 = 0
n_steps1 = 0
n_steps2 = 0
c_steps1 = 0
c_steps2 = 0
r_steps1 = 0
r_steps2 = 0
e_steps1 = 0
e_steps2 = 0
v_steps1 = 0
v_steps2 = 0
f_steps1 = 0
f_steps2 = 0
w_steps1 = 0
w_steps2 = 0
x_steps1 = 0
x_steps2 = 0

a_itera1 = 0
a_itera2 = 0
l_itera1 = 0
l_itera2 = 0
t_itera1 = 0
t_itera2 = 0
m_itera1 = 0
m_itera2 = 0
n_itera1 = 0
n_itera2 = 0
c_itera1 = 0
c_itera2 = 0
r_itera1 = 0
r_itera2 = 0
e_itera1 = 0
e_itera2 = 0
v_itera1 = 0
v_itera2 = 0
f_itera1 = 0
f_itera2 = 0
w_itera1 = 0
w_itera2 = 0
x_itera1 = 0
x_itera2 = 0

sdr_dist = 0
rng = 0
rec_dihcf_force = 0
buffer_z = 0
num_waters = 0
ion_conc = 0.0
retain_lig_prot = "no"
ligand_ph = 7.0
ligand_charge = "nd"
software = "amber"
solv_shell = 0.0
dlambda = 0.001

ntpr = "1000"
ntwr = "10000"
ntwe = "0"
ntwx = "2500"
cut = "9.0"
barostat = "2"
ti_points = 0

# Before the loop (defaults)
protein_type = "standard"
prot_first_orig = None
prot_last_orig = None
cyp_residue_orig = None
num_chains = 1
heme_1_orig = None
heme_2_orig = None
sdr_axis = ""  # default

# Read arguments that define input file and stage
if len(sys.argv) < 5:
    scripts.help_message()
    sys.exit(0)
for i in [1, 3]:
    if "-i" == sys.argv[i].lower():
        input_file = sys.argv[i + 1]
    elif "-s" == sys.argv[i].lower():
        stage = sys.argv[i + 1]
    else:
        scripts.help_message()
        sys.exit(1)

# Open input file
with open(input_file) as f_in:
    # Remove spaces and tabs
    lines = (line.strip(" \t\n\r") for line in f_in)
    lines = list(line for line in lines if line)  # Non-blank lines in a list

for i in range(0, len(lines)):
    # split line using the equal sign, and remove text after #
    if not lines[i][0] == "#":
        lines[i] = lines[i].split("#")[0].split("=")

# Read parameters from input file
for i in range(0, len(lines)):
    if not lines[i][0] == "#":
        lines[i][0] = lines[i][0].strip().lower()
        lines[i][1] = lines[i][1].strip()
        key = lines[i][0].strip().lower()
        val = lines[i][1].strip()
        ## Residue input for heme containing proteins
        if key == "protein_type":
            protein_type = val.strip().strip("'\"").lower()
            print("protein type:", protein_type)

        elif key == "prot_first":
            prot_first_orig = scripts.check_input("int", val, input_file, key)
            print("prot_first_orig:", prot_first_orig)

        elif key == "prot_last":
            prot_last_orig = scripts.check_input("int", val, input_file, key)
            print("prot_last_orig:", prot_last_orig)

        elif key == "cyp_residue":
            cyp_residue_orig = scripts.check_input("int", val, input_file, key)
            print("cyp_residue_orig:", cyp_residue_orig)
        elif key == "num_chains":
            num_chains = scripts.check_input("int", val, input_file, key)
            print("num_chains:", num_chains)

        elif key == "heme_1":
            heme_1_orig = scripts.check_input("int", val, input_file, key)
            print("heme_1_orig:", heme_1_orig)

        elif key == "heme_2":
            heme_2_orig = scripts.check_input("int", val, input_file, key)
            print("heme_2_orig:", heme_2_orig)

        ###########Simulation Parameters################
        if lines[i][0] == "temperature":
            temperature = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "eq_steps1":
            eq_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "eq_steps2":
            eq_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "a_steps1":
            a_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "a_steps2":
            a_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "l_steps1":
            l_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "l_steps2":
            l_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "t_steps1":
            t_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "t_steps2":
            t_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "m_steps1":
            m_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "m_steps2":
            m_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "n_steps1":
            n_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "n_steps2":
            n_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "c_steps1":
            c_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "c_steps2":
            c_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "r_steps1":
            r_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "r_steps2":
            r_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "e_steps1":
            e_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "e_steps2":
            e_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "v_steps1":
            v_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "v_steps2":
            v_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "f_steps1":
            f_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "f_steps2":
            f_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "w_steps1":
            w_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "w_steps2":
            w_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "x_steps1":
            x_steps1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "x_steps2":
            x_steps2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        # OpenMM only
        elif lines[i][0] == "a_itera1":
            a_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "a_itera2":
            a_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "l_itera1":
            l_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "l_itera2":
            l_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "t_itera1":
            t_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "t_itera2":
            t_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "m_itera1":
            m_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "m_itera2":
            m_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "n_itera1":
            n_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "n_itera2":
            n_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "c_itera1":
            c_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "c_itera2":
            c_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "r_itera1":
            r_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "r_itera2":
            r_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "e_itera1":
            e_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "e_itera2":
            e_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "v_itera1":
            v_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "v_itera2":
            v_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "f_itera1":
            f_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "f_itera2":
            f_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "w_itera1":
            w_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "w_itera2":
            w_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "x_itera1":
            x_itera1 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "x_itera2":
            x_itera2 = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "itera_steps":
            itera_steps = scripts.check_input(
                "int", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "itcheck":
            itcheck = lines[i][1]
        ####
        elif lines[i][0] == "ti_points":
            ti_points = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "poses_list":
            newline = lines[i][1].strip("'\"-,.:;#()][").split(",")
            for j in range(0, len(newline)):
                poses_list.append(
                    scripts.check_input("int", newline[j], input_file, lines[i][0])
                )
        elif lines[i][0] == "ligand_list":
            newline = lines[i][1].strip("'\"-,.:;#()][").split(",")
            for j in range(0, len(newline)):
                ligand_list.append(newline[j])
        elif lines[i][0] == "other_mol":
            newline = lines[i][1].strip("'\"-,.:;#()][").split(",")
            for j in range(0, len(newline)):
                other_mol.append(newline[j])
        elif lines[i][0] == "calc_type":
            if lines[i][1].lower() == "dock":
                calc_type = lines[i][1].lower()
            elif lines[i][1].lower() == "rank":
                calc_type = lines[i][1].lower()
            elif lines[i][1].lower() == "crystal":
                calc_type = lines[i][1].lower()
            else:
                print("Please choose dock, rank or crystal for the calculation type")
                sys.exit(1)
        elif lines[i][0] == "retain_lig_prot":
            retain_lig_prot = lines[i][1].lower()
        elif lines[i][0] == "celpp_receptor":
            newline = lines[i][1].strip("'\"-,.:;#()][").split(",")
            for j in range(0, len(newline)):
                celp_st.append(newline[j])
        elif lines[i][0] == "p1":
            H1 = lines[i][1]
        elif lines[i][0] == "p2":
            H2 = lines[i][1]
        elif lines[i][0] == "p3":
            H3 = lines[i][1]
        elif lines[i][0] == "ligand_name":
            newline = lines[i][1].strip("'\"-,.:;#()][").split(",")
            for j in range(0, len(newline)):
                mols.append(newline[j])
        elif lines[i][0] == "fe_type":
            if lines[i][1].lower() == "rest":
                fe_type = lines[i][1].lower()
            elif lines[i][1].lower() == "dd":
                fe_type = lines[i][1].lower()
            elif lines[i][1].lower() == "sdr":
                fe_type = lines[i][1].lower()
            elif lines[i][1].lower() == "sdr-rest":
                fe_type = lines[i][1].lower()
            elif lines[i][1].lower() == "express":
                fe_type = lines[i][1].lower()
            elif lines[i][1].lower() == "dd-rest":
                fe_type = lines[i][1].lower()
            elif lines[i][1].lower() == "relative":
                fe_type = lines[i][1].lower()
            elif lines[i][1].lower() == "custom":
                fe_type = lines[i][1].lower()
            else:
                print(
                    "Free energy type not recognized, please choose rest (restraints only), dd (double decoupling only), sdr (simultaneous decoupling-recoupling only), express (sdr with simultaneous restraints), dd-rest (dd with restraints), sdr-rest (sdr with restraints), relative (using merged restraints) or custom."
                )
                sys.exit(1)
        elif lines[i][0] == "dec_int":
            if lines[i][1].lower() == "mbar":
                dec_int = lines[i][1].lower()
            elif lines[i][1].lower() == "ti":
                dec_int = lines[i][1].lower()
            else:
                print(
                    "Decoupling integration method not recognized, please choose ti or mbar"
                )
                sys.exit(1)
        elif lines[i][0] == "dec_method":
            if lines[i][1].lower() == "dd":
                dec_method = lines[i][1].lower()
            elif lines[i][1].lower() == "sdr":
                dec_method = lines[i][1].lower()
            elif lines[i][1].lower() == "exchange":
                dec_method = lines[i][1].lower()
            else:
                print("Decoupling method not recognized, please choose dd or sdr")
                sys.exit(1)

        elif lines[i][0] == "sdr_axis":
            # Store always; only USED later when dec_method == "sdr"
            sdr_axis = lines[i][1].strip().strip("'\"").lower()

        elif lines[i][0] == "blocks":
            blocks = scripts.check_input("int", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "hmr":
            if lines[i][1].lower() == "yes":
                hmr = "yes"
            elif lines[i][1].lower() == "no":
                hmr = "no"
            else:
                print(
                    "Wrong input! Please use yes or no to indicate whether hydrogen mass repartitioning "
                    "will be used."
                )
                sys.exit(1)
        elif lines[i][0] == "water_model":
            if lines[i][1].lower() == "tip3p":
                water_model = lines[i][1].upper()
            elif lines[i][1].lower() == "tip4pew":
                water_model = lines[i][1].upper()
            elif lines[i][1].lower() == "spce":
                water_model = lines[i][1].upper()
            elif lines[i][1].lower() == "opc":
                water_model = lines[i][1].upper()
            elif lines[i][1].lower() == "tip3pf":
                water_model = lines[i][1].upper()
            else:
                print(
                    "Water model not supported. Please choose TIP3P, TIP4PEW, SPCE, OPC or TIP3PF"
                )
                sys.exit(1)
        elif lines[i][0] == "num_waters":
            num_waters = scripts.check_input(
                "int", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "neutralize_only":
            if lines[i][1].lower() == "yes":
                neut = "yes"
            elif lines[i][1].lower() == "no":
                neut = "no"
            else:
                print(
                    "Wrong input! Please choose neutralization only or add extra ions"
                )
                sys.exit(1)
        elif lines[i][0] == "cation":
            cation = lines[i][1]
        elif lines[i][0] == "anion":
            anion = lines[i][1]
        elif lines[i][0] == "ion_conc":
            ion_conc = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "buffer_x":
            buffer_x = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "buffer_y":
            buffer_y = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "buffer_z":
            buffer_z = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "lig_buffer":
            lig_buffer = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "rec_dihcf_force":
            rec_dihcf_force = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "rec_discf_force":
            rec_discf_force = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "lig_distance_force":
            lig_distance_force = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "lig_angle_force":
            lig_angle_force = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "lig_dihcf_force":
            lig_dihcf_force = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "rec_com_force":
            rec_com_force = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "lig_com_force":
            lig_com_force = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "sdr_dist":
            sdr_dist = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "l1_x":
            l1_x = scripts.check_input("float", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "l1_y":
            l1_y = scripts.check_input("float", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "l1_z":
            l1_z = scripts.check_input("float", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "l1_range":
            l1_range = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "min_adis":
            min_adis = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "max_adis":
            max_adis = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "solv_shell":
            solv_shell = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "dlambda":
            dlambda = scripts.check_input("float", lines[i][1], input_file, lines[i][0])
        elif lines[i][0] == "rec_bb":
            if lines[i][1].lower() == "yes":
                rec_bb = "yes"
            elif lines[i][1].lower() == "no":
                rec_bb = "no"
            else:
                print(
                    "Wrong input! Please use yes or no to indicate whether protein backbone restraints "
                    "will be used."
                )
                sys.exit(1)
        elif lines[i][0] == "bb_start":
            newline = lines[i][1].strip("'\"-,.:;#()][").split(",")
            for j in range(0, len(newline)):
                bb_start.append(
                    scripts.check_input("int", newline[j], input_file, lines[i][0])
                )
        elif lines[i][0] == "bb_end":
            newline = lines[i][1].strip("'\"-,.:;#()][").split(",")
            for j in range(0, len(newline)):
                bb_end.append(
                    scripts.check_input("int", newline[j], input_file, lines[i][0])
                )
            if len(bb_start) != len(bb_end):
                print(
                    "Wrong input! Please use arrays of the same size for bb_start and bb_end"
                )
                sys.exit(1)
        elif lines[i][0] == "bb_equil":
            if lines[i][1].lower() == "yes":
                bb_equil = lines[i][1].lower()
            else:
                bb_equil = "no"
        elif lines[i][0] == "release_eq":
            strip_line = lines[i][1].strip("'\"-,.:;#()][").split()
            for j in range(0, len(strip_line)):
                release_eq.append(
                    scripts.check_input("float", strip_line[j], input_file, lines[i][0])
                )
        elif lines[i][0] == "attach_rest":
            strip_line = lines[i][1].strip("'\"-,.:;#()][").split()
            for j in range(0, len(strip_line)):
                attach_rest.append(
                    scripts.check_input("float", strip_line[j], input_file, lines[i][0])
                )
        elif lines[i][0] == "lambdas":
            strip_line = lines[i][1].strip("'\"-,.:;#()][").split()
            for j in range(0, len(strip_line)):
                lambdas.append(
                    scripts.check_input("float", strip_line[j], input_file, lines[i][0])
                )
        elif lines[i][0] == "components":
            strip_line = lines[i][1].strip("'\"-,.:;#()][").split()
            for j in range(0, len(strip_line)):
                components.append(strip_line[j])
        elif lines[i][0] == "ntpr":
            ntpr = lines[i][1]
        elif lines[i][0] == "ntwr":
            ntwr = lines[i][1]
        elif lines[i][0] == "ntwe":
            ntwe = lines[i][1]
        elif lines[i][0] == "ntwx":
            ntwx = lines[i][1]
        elif lines[i][0] == "cut":
            cut = lines[i][1]
        elif lines[i][0] == "gamma_ln":
            gamma_ln = lines[i][1]
        elif lines[i][0] == "barostat":
            barostat = lines[i][1]
        elif lines[i][0] == "receptor_ff":
            receptor_ff = lines[i][1]
        elif lines[i][0] == "ligand_ff":
            if lines[i][1].lower() == "gaff":
                ligand_ff = "gaff"
            elif lines[i][1].lower() == "gaff2":
                ligand_ff = "gaff2"
            else:
                print(
                    "Wrong input! Available options for ligand force-field are gaff and gaff2"
                )
                sys.exit(1)
        elif lines[i][0] == "ligand_ph":
            ligand_ph = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "ligand_charge":
            ligand_charge = scripts.check_input(
                "float", lines[i][1], input_file, lines[i][0]
            )
        elif lines[i][0] == "dt":
            dt = lines[i][1]
        elif lines[i][0] == "software":
            if lines[i][1].lower() == "openmm":
                software = lines[i][1].lower()
            elif lines[i][1].lower() == "amber":
                software = lines[i][1].lower()
            else:
                print(
                    "Simulation software not recognized, please choose openmm or amber"
                )
                sys.exit(1)

if dec_method != "sdr":
    sdr_axis = None

# ------------------------------------------------------------
# Residue mapping for heme-containing proteins (supports 1–2 chains)
# Assumptions:
#   - prot_first/prot_last/cyp_residue are in ORIGINAL per-chain numbering
#   - After mapping, chain A starts at residue 1 in AMBER numbering
#   - Chain B continues numbering after chain A, with a +1 TER boundary
#   - In EQUIL, 1 dummy residue is added before the protein region
#   - In FE,   2 dummy residues are added before the protein region
#   - UNL (ligand) is appended immediately after protein residues
#   - other_mol residues are appended after UNL; for num_chains=2 they appear twice
#     in the same order as other_mol (A then B)
# ------------------------------------------------------------
if protein_type == "heme":
    if prot_first_orig is None or prot_last_orig is None or cyp_residue_orig is None:
        raise ValueError(
            "protein_type=heme but prot_first/prot_last/cyp_residue missing in input."
        )

    if num_chains not in (1, 2):
        raise ValueError("Currently only num_chains = 1 or 2 is supported.")

    # --- Per-chain length in original numbering ---
    prot_len = prot_last_orig - prot_first_orig + 1
    if prot_len <= 0:
        raise ValueError(
            f"Invalid prot range: prot_first={prot_first_orig}, prot_last={prot_last_orig}"
        )

    prot_shift = prot_first_orig - 1  # so prot_first_orig -> 1

    # ---------- Chain A (AMBER numbering, no dummies) ----------
    prot_first_amber_A = _map_res_orig_to_amber(
        prot_first_orig, prot_shift
    )  # should be 1
    prot_last_amber_A = _map_res_orig_to_amber(prot_last_orig, prot_shift)
    cyp_residue_amber_A = _map_res_orig_to_amber(cyp_residue_orig, prot_shift)

    # ---------- Chain B (AMBER numbering, no dummies) ----------
    # +1 because reconstruction places chain B after a TER boundary
    prot_first_amber_B = prot_last_amber_B = cyp_residue_amber_B = None
    if num_chains == 2:
        chainB_offset = prot_len  # to be reviewed later: chainB_offset = prot_len + 1
        prot_first_amber_B = prot_first_amber_A + chainB_offset
        prot_last_amber_B = prot_last_amber_A + chainB_offset
        cyp_residue_amber_B = cyp_residue_amber_A + chainB_offset

    # ---------- Stage dummy shifts ----------
    NDUMMY_1 = 1
    NDUMMY_2 = 2

    # Convenience: "first/second CYP" in no-dummy AMBER numbering
    first_cyp = cyp_residue_amber_A
    second_cyp = cyp_residue_amber_B if num_chains == 2 else None

    # CYP residue indices in stage-specific numbering (dummies added before protein)
    first_cyp_1dum = first_cyp + NDUMMY_1
    first_cyp_2dum = first_cyp + NDUMMY_2
    second_cyp_1dum = (second_cyp + NDUMMY_1) if second_cyp is not None else None
    second_cyp_2dum = (second_cyp + NDUMMY_2) if second_cyp is not None else None

    # Neighbor residues (used for peptide connectivity in tleap)
    first_cyp_next_1dum = first_cyp_1dum + 1
    first_cyp_next_2dum = first_cyp_2dum + 1
    second_cyp_next_1dum = (
        (second_cyp_1dum + 1) if second_cyp_1dum is not None else None
    )
    second_cyp_next_2dum = (
        (second_cyp_2dum + 1) if second_cyp_2dum is not None else None
    )

    # Previous residue can be None if CYP is residue 1 in that numbering
    first_cyp_previous_1dum = (first_cyp_1dum - 1) if first_cyp_1dum > 1 else None
    first_cyp_previous_2dum = (first_cyp_2dum - 1) if first_cyp_2dum > 1 else None
    second_cyp_previous_1dum = (
        (second_cyp_1dum - 1)
        if (second_cyp_1dum is not None and second_cyp_1dum > 1)
        else None
    )
    second_cyp_previous_2dum = (
        (second_cyp_2dum - 1)
        if (second_cyp_2dum is not None and second_cyp_2dum > 1)
        else None
    )

    # ---------- Protein end in stage numbering (needed to place UNL/other_mol) ----------
    # End of protein without dummies:
    prot_end_nodum = prot_last_amber_B if num_chains == 2 else prot_last_amber_A
    # End of protein with stage dummies:
    prot_end_1dum = prot_end_nodum + NDUMMY_1
    prot_end_2dum = prot_end_nodum + NDUMMY_2

    # ---------- Heme residue numbers ----------
    # Policy:
    #   - heme_1/heme_2 in the input are OPTIONAL overrides
    #   - they MUST match the EQUIL numbering used by tleap scripts in equil
    #   - if they look wrong, we ignore them and infer from other_mol

    heme_amber_A = heme_1_orig  # already int or None from parsing
    heme_amber_B = heme_2_orig  # already int or None from parsing

    def _infer_hemes_from_othermol(prot_end_stage: int, other_mol_list, nchains: int):
        unl_res = prot_end_stage + 1
        start = unl_res + 1
        expanded = []
        for _ in range(int(nchains)):
            expanded.extend(list(other_mol_list))
        hem_res = []
        for idx, name in enumerate(expanded):
            if str(name).upper() == "HEM":
                hem_res.append(start + idx)
        return hem_res

    # ---------- EQUIL (1 dummy) ----------
    hem_1dum = _infer_hemes_from_othermol(prot_end_1dum, other_mol, num_chains)

    # Validate overrides against inferred EQUIL numbering
    if heme_amber_A is not None and len(hem_1dum) >= 1 and heme_amber_A != hem_1dum[0]:
        print(
            "WARNING: heme_1 override does not match inferred EQUIL numbering; ignoring override.",
            flush=True,
        )
        heme_amber_A = None

    if (
        num_chains == 2
        and heme_amber_B is not None
        and len(hem_1dum) >= 2
        and heme_amber_B != hem_1dum[1]
    ):
        print(
            "WARNING: heme_2 override does not match inferred EQUIL numbering; ignoring override.",
            flush=True,
        )
        heme_amber_B = None

    # Fill missing from inference (EQUIL)
    if heme_amber_A is None:
        heme_amber_A = hem_1dum[0] if len(hem_1dum) >= 1 else None
    if num_chains == 2 and heme_amber_B is None:
        heme_amber_B = hem_1dum[1] if len(hem_1dum) >= 2 else None

    # Hard error if still missing
    if heme_amber_A is None:
        raise ValueError(
            "Could not infer heme_1 from other_mol for equil; include 'HEM' in other_mol or provide heme_1."
        )
    if num_chains == 2 and heme_amber_B is None:
        raise ValueError(
            "Could not infer heme_2 from other_mol for equil; include 'HEM' in other_mol or provide heme_2."
        )

    # ---------- Expose heme numbers ----------
    # EQUIL (1 dummy)
    heme_1_1dum = heme_amber_A
    heme_2_1dum = heme_amber_B if num_chains == 2 else None

    # FE (2 dummies): shift by +1 relative to EQUIL
    # (because FE has one extra dummy compared to EQUIL)
    heme_1_2dum = heme_1_1dum + 1
    heme_2_2dum = heme_2_1dum + 1 if heme_2_1dum is not None else None

    # Backward-compatible names (if existing code expects these)
    heme_1 = heme_1_1dum
    heme_2 = heme_2_1dum

    log("Heme mapping summary:")
    log(f"  num_chains={num_chains}, prot_len_per_chain={prot_len}")
    log(
        f"  Chain A: prot {prot_first_amber_A}..{prot_last_amber_A}, cyp={cyp_residue_amber_A}"
    )
    log(
        f"  Chain B: prot {prot_first_amber_B}..{prot_last_amber_B}, cyp={cyp_residue_amber_B}"
    )
    log(
        f"  Equil dummy(+1): first_cyp_1dum={first_cyp_1dum}, second_cyp_1dum={second_cyp_1dum}, first heme_1dum={heme_1_1dum}, second heme_1dum={heme_2_1dum},"
    )
    log(
        f"  FE dummy(+2):    first_cyp_2dum={first_cyp_2dum},    second_cyp_2dum={second_cyp_2dum}, first heme_1dum={heme_2_2dum}, second heme_2dum={heme_2_2dum},"
    )


####Water models and box################
if num_waters == 0 and buffer_z == 0:
    print(
        "Wrong input! Please choose either a number of water molecules or a z buffer value."
    )
    sys.exit(1)

if num_waters != 0 and buffer_z != 0:
    print(
        "Wrong input! Please choose either a number of water molecules or a z buffer value."
    )
    sys.exit(1)

if buffer_x <= solv_shell or buffer_y <= solv_shell:
    print(
        "Wrong input! Solvation buffers cannot be smaller than the solv_shell variable."
    )
    sys.exit(1)

if buffer_z != 0 and buffer_z <= solv_shell:
    print(
        "Wrong input! Solvation buffers cannot be smaller than the solv_shell variable."
    )
    sys.exit(1)

if other_mol == [""]:
    other_mol = []

# Number of simulations, 1 equilibrium and 1 production
apr_sim = 2

# Define free energy components
if fe_type == "custom":
    try:
        dec_method
    except NameError:
        print(
            "Wrong input! Please choose a decoupling method (dd, sdr or exchange) when using the custom option."
        )
        sys.exit(1)
elif fe_type == "rest":
    components = ["c", "a", "l", "t", "r"]
    dec_method = "dd"
elif fe_type == "sdr":
    components = ["e", "v"]
    dec_method = "sdr"
elif fe_type == "dd":
    components = ["e", "v", "f", "w"]
    dec_method = "dd"
elif fe_type == "sdr-rest":
    components = ["c", "a", "l", "t", "r", "e", "v"]
    dec_method = "sdr"
elif fe_type == "express":
    components = ["m", "n", "e", "v"]
    dec_method = "sdr"
elif fe_type == "dd-rest":
    components = ["c", "a", "l", "t", "r", "e", "v", "f", "w"]
    dec_method = "dd"
elif fe_type == "relative":
    components = ["x", "e", "n", "m"]
    dec_method = "exchange"

if (dec_method == "sdr" or dec_method == "exchange") and sdr_dist == 0:
    print(
        "Wrong input! Please choose a positive value for the sdr_dist variable when performing sdr or exchange."
    )
    sys.exit(1)

for i in components:
    if i == "n" and sdr_dist == 0:
        print(
            "Wrong input! Please choose a positive value for the sdr_dist variable when using the n component."
        )
        sys.exit(1)

# Do not apply protein backbone restraints
if rec_bb == "no":
    bb_start = [1]
    bb_end = [0]
    bb_equil = "no"


# Create poses definitions
if calc_type == "dock":
    celp_st = celp_st[0]
    for i in range(0, len(poses_list)):
        poses_def.append("pose" + str(poses_list[i]))
elif calc_type == "rank":
    celp_st = celp_st[0]
    for i in range(0, len(ligand_list)):
        poses_def.append(ligand_list[i])
elif calc_type == "crystal":
    for i in range(0, len(celp_st)):
        poses_def.append(celp_st[i])

# Obtain all ligand names
if calc_type != "crystal":
    mols = []
    for i in range(0, len(poses_def)):
        with open("./all-poses/%s.pdb" % poses_def[i].lower()) as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
            for j in range(0, len(lines)):
                if (lines[j][0:6].strip() == "ATOM") or (
                    lines[j][0:6].strip() == "HETATM"
                ):
                    lig_name = lines[j][17:20].strip()
                    mols.append(lig_name)
                    break

print("Receptor/complex structures:")
print(celp_st)
print("Ligand names")
print(mols)
print("Cobinders names:")
print(other_mol)

for i in range(0, len(mols)):
    if mols[i] in other_mol:
        print(
            "Same residue name ("
            + mols[i]
            + ") found in ligand name and cobinders, please change one of them"
        )
        sys.exit(1)


# Create restraint definitions
rest = [
    rec_dihcf_force,
    rec_discf_force,
    lig_distance_force,
    lig_angle_force,
    lig_dihcf_force,
    rec_com_force,
    lig_com_force,
]

# Create ion definitions
ion_def = [cation, anion, ion_conc]

# Define number of steps for all stages (amber)
dic_steps1 = {}
dic_steps2 = {}
dic_steps1["a"] = a_steps1
dic_steps2["a"] = a_steps2
dic_steps1["l"] = l_steps1
dic_steps2["l"] = l_steps2
dic_steps1["t"] = t_steps1
dic_steps2["t"] = t_steps2
dic_steps1["m"] = m_steps1
dic_steps2["m"] = m_steps2
dic_steps1["n"] = n_steps1
dic_steps2["n"] = n_steps2
dic_steps1["c"] = c_steps1
dic_steps2["c"] = c_steps2
dic_steps1["r"] = r_steps1
dic_steps2["r"] = r_steps2
dic_steps1["v"] = v_steps1
dic_steps2["v"] = v_steps2
dic_steps1["e"] = e_steps1
dic_steps2["e"] = e_steps2
dic_steps1["w"] = w_steps1
dic_steps2["w"] = w_steps2
dic_steps1["f"] = f_steps1
dic_steps2["f"] = f_steps2
dic_steps1["x"] = x_steps1
dic_steps2["x"] = x_steps2

# Define number of steps for all stages (openmm)
dic_itera1 = {}
dic_itera2 = {}
dic_itera1["a"] = a_itera1
dic_itera2["a"] = a_itera2
dic_itera1["l"] = l_itera1
dic_itera2["l"] = l_itera2
dic_itera1["t"] = t_itera1
dic_itera2["t"] = t_itera2
dic_itera1["m"] = m_itera1
dic_itera2["m"] = m_itera2
dic_itera1["n"] = n_itera1
dic_itera2["n"] = n_itera2
dic_itera1["c"] = c_itera1
dic_itera2["c"] = c_itera2
dic_itera1["r"] = r_itera1
dic_itera2["r"] = r_itera2
dic_itera1["v"] = v_itera1
dic_itera2["v"] = v_itera2
dic_itera1["e"] = e_itera1
dic_itera2["e"] = e_itera2
dic_itera1["w"] = w_itera1
dic_itera2["w"] = w_itera2
dic_itera1["f"] = f_itera1
dic_itera2["f"] = f_itera2
dic_itera1["x"] = x_itera1
dic_itera2["x"] = x_itera2

# Obtain Gaussian Quadrature lambdas and weights

if dec_int == "ti":
    if ti_points != 0:
        lambdas = []
        weights = []
        x, y = np.polynomial.legendre.leggauss(ti_points)
        # Adjust Gaussian lambdas
        for i in range(0, len(x)):
            lambdas.append(float((x[i] + 1) / 2))
        # Adjust Gaussian weights
        for i in range(0, len(y)):
            weights.append(float(y[i] / 2))
    else:
        print(
            "Wrong input! Please choose a positive integer for the ti_points variable when using the TI-GQ method"
        )
        sys.exit(1)
    print("lambda values:", lambdas)
    print("Gaussian weights:", weights)
elif dec_int == "mbar":
    if lambdas == []:
        print(
            "Wrong input! Please choose a set of lambda values when using the MBAR method"
        )
        sys.exit(1)
    if ti_points != 0:
        print(
            "Wrong input! Do not define the ti_points variable when applying the MBAR method, instead choose a set of lambda values"
        )
        sys.exit(1)
    print("lambda values:", lambdas)


# Adjust components and windows for OpenMM

if software == "openmm" and stage == "fe":
    components_inp = list(components)
    print(components_inp)
    if sdr_dist == 0:
        dec_method_inp = dec_method
        components = ["t", "c"]
    elif dec_method != "exchange":
        dec_method_inp = dec_method
        print(dec_method_inp)
        dec_method = "sdr"
        components = ["t", "c", "n", "v"]
    else:
        dec_method_inp = dec_method
        print(dec_method_inp)
        components = ["t", "c", "n", "v", "x"]
    attach_rest_inp = list(attach_rest)
    print(attach_rest_inp)
    attach_rest = [100.0]
    lambdas_inp = list(lambdas)
    print(lambdas_inp)
    lambdas = [0.0]
    dt = str(float(dt) * 1000)
    print(dt)
    cut = str(float(cut) / 10)
    print(cut)

    # Convert equil output file
    os.chdir("equil")
    for i in range(0, len(poses_def)):
        pose = poses_def[i]
        rng = len(release_eq) - 1
        if os.path.exists(pose):
            os.chdir(pose)
            convert_file = open("convert.in", "w")
            convert_file.write("parm full.prmtop\n")
            convert_file.write("trajin md%02d.dcd\n" % rng)
            convert_file.write("trajout md%02d.rst7 onlyframes 10\n" % rng)
            convert_file.close()
            sp.call("cpptraj -i convert.in > convert.log", shell=True)
            os.chdir("../")
    os.chdir("../")


if stage == "equil":
    comp = "q"
    win = 0
    # Create equilibrium systems for all poses listed in the input file
    for i in range(0, len(poses_def)):
        pose = poses_def[i]
        poser = poses_def[0]
        mol = mols[i]
        molr = mols[0]
        rng = len(release_eq) - 1
        if not os.path.exists("./all-poses/" + pose + ".pdb"):
            continue
        print("Setting up " + str(poses_def[i]))
        # Get number of simulations
        num_sim = len(release_eq)
        # Create aligned initial complex
        anch = build.build_equil_heme(
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
            first_cyp_equil=first_cyp_1dum,
            second_cyp_equil=second_cyp_1dum,
            first_cyp_next_equil=first_cyp_next_1dum,
            second_cyp_next_equil=second_cyp_next_1dum,
            first_cyp_previous_equil=first_cyp_previous_1dum,
            second_cyp_previous_equil=second_cyp_previous_1dum,
            heme_1=heme_1_1dum,
            heme_2=heme_2_1dum,
        )
        if anch == "anch1":
            aa1_poses.append(pose)
            os.chdir("../")
            continue
        if anch == "anch2":
            aa2_poses.append(pose)
            os.chdir("../")
            continue
        # Solvate system with ions
        print("Creating box...")
        build.create_box_cyp_equil(
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
            first_cyp_equil=first_cyp_1dum,
            second_cyp_equil=second_cyp_1dum,
            first_cyp_next_equil=first_cyp_next_1dum,
            second_cyp_next_equil=second_cyp_next_1dum,
            first_cyp_previous_equil=first_cyp_previous_1dum,
            second_cyp_previous_equil=second_cyp_previous_1dum,
            heme_1=heme_1_1dum,
            heme_2=heme_2_1dum,
        )
        # Apply restraints and prepare simulation files
        print("Equil release weights:")
        for i in range(0, len(release_eq)):
            weight = release_eq[i]
            print("%s" % str(weight))
            setup.restraints(
                pose,
                rest,
                bb_start,
                bb_end,
                weight,
                stage,
                mol,
                molr,
                comp,
                bb_equil,
                sdr_dist,
                dec_method,
                other_mol,
            )
            shutil.copy(
                "./" + pose + "/disang.rest", "./" + pose + "/disang%02d.rest" % int(i)
            )
        shutil.copy(
            "./" + pose + "/disang%02d.rest" % int(0), "./" + pose + "/disang.rest"
        )
        setup.sim_files(
            hmr,
            temperature,
            mol,
            num_sim,
            pose,
            comp,
            win,
            stage,
            eq_steps1,
            eq_steps2,
            rng,
        )
        os.chdir("../")
    if len(aa1_poses) != 0:
        print("\n")
        print("WARNING: Could not find the ligand first anchor L1 for", aa1_poses)
        print(
            "The ligand is most likely not in the defined binding site in these systems."
        )
    if len(aa2_poses) != 0:
        print("\n")
        print("WARNING: Could not find the ligand L2 or L3 anchors for", aa2_poses)
        print("Try reducing the min_adis parameter in the input file.")

elif stage == "fe":
    # Create systems for all poses after preparation
    num_sim = apr_sim
    # Create and move to free energy directory
    if not os.path.exists("fe"):
        os.makedirs("fe")
    os.chdir("fe")
    for i in range(0, len(poses_def)):
        pose = poses_def[i]
        poser = poses_def[0]
        mol = mols[i]
        molr = mols[0]
        fwin = len(release_eq) - 1
        if not os.path.exists("../equil/" + pose):
            continue
        print("Setting up " + str(poses_def[i]))
        # Create and move to pose directory
        if not os.path.exists(pose):
            os.makedirs(pose)
        os.chdir(pose)
        # Generate folder and restraints for all components and windows
        for j in range(0, len(components)):
            comp = components[j]
            print(f"[DEBUG] Starting component: {comp}", flush=True)
            # Ligand conformational release in a small box
            if comp == "c":
                if not os.path.exists("rest"):
                    os.makedirs("rest")
                os.chdir("rest")
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if int(win) == 0:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_1dum,
                            second_cyp_dec=second_cyp_1dum,
                            first_cyp_next_dec=first_cyp_next_1dum,
                            second_cyp_next_dec=second_cyp_next_1dum,
                            first_cyp_previous_dec=first_cyp_previous_1dum,
                            second_cyp_previous_dec=second_cyp_previous_1dum,
                            heme_1=heme_1_1dum,
                            heme_2=heme_2_1dum,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        print("Creating box for ligand only...")
                        build.ligand_box(
                            mol, lig_buffer, water_model, neut, ion_def, comp, ligand_ff
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            c_steps1,
                            c_steps2,
                            rng,
                        )
                    else:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            c_steps1,
                            c_steps2,
                            rng,
                        )
                if anch != "all":
                    break
                os.chdir("../")
            # Receptor conformational release in a separate box
            elif comp == "n":
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists("rest"):
                    os.makedirs("rest")
                os.chdir("rest")
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if int(win) == 0:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        print("Creating box for protein/simultaneous release...")
                        build.create_box_cyp_fe(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                    else:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                print(
                    f"[DEBUG] comp={comp} win={win} anch={anch} cwd={os.getcwd()}",
                    flush=True,
                )

                if anch != "all":
                    print(
                        f"[DEBUG] BREAK triggered for comp={comp} because anch={anch}",
                        flush=True,
                    )
                    break

                print(
                    f"[DEBUG] continuing, leaving directory {os.getcwd()}", flush=True
                )
                os.chdir("../")
                print(f"[DEBUG] now in {os.getcwd()}", flush=True)
            # Component r for releasing protein Restraints
            elif comp == "r":
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists("rest"):
                    os.makedirs("rest")
                os.chdir("rest")
                # receptor-only: no UNL ligand residue -> other_mol (including HEM) shifts by -1
                heme_shift = 1 if comp == "r" else 0
                heme_1_for_rcomp = heme_1_1dum - heme_shift
                heme_2_for_rcomp = (
                    (heme_2_1dum - heme_shift) if heme_2_1dum is not None else None
                )
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if int(win) == 0:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_1dum,
                            second_cyp_dec=second_cyp_1dum,
                            first_cyp_next_dec=first_cyp_next_1dum,
                            second_cyp_next_dec=second_cyp_next_1dum,
                            first_cyp_previous_dec=first_cyp_previous_1dum,
                            second_cyp_previous_dec=second_cyp_previous_1dum,
                            heme_1=heme_1_for_rcomp,
                            heme_2=heme_2_for_rcomp,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        print("Creating box for protein/simultaneous release...")
                        build.create_box_cyp_equil(
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
                            first_cyp_equil=first_cyp_1dum,
                            second_cyp_equil=second_cyp_1dum,
                            first_cyp_next_equil=first_cyp_next_1dum,
                            second_cyp_next_equil=second_cyp_next_1dum,
                            first_cyp_previous_equil=first_cyp_previous_1dum,
                            second_cyp_previous_equil=second_cyp_previous_1dum,
                            heme_1=heme_1_for_rcomp,
                            heme_2=heme_2_for_rcomp,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                    else:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_1dum,
                            second_cyp_dec=second_cyp_1dum,
                            first_cyp_next_dec=first_cyp_next_1dum,
                            second_cyp_next_dec=second_cyp_next_1dum,
                            first_cyp_previous_dec=first_cyp_previous_1dum,
                            second_cyp_previous_dec=second_cyp_previous_1dum,
                            heme_1=heme_1_for_rcomp,
                            heme_2=heme_2_for_rcomp,
                            sdr_axis=sdr_axis,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                print(
                    f"[DEBUG] comp={comp} win={win} anch={anch} cwd={os.getcwd()}",
                    flush=True,
                )

                if anch != "all":
                    print(
                        f"[DEBUG] BREAK triggered for comp={comp} because anch={anch}",
                        flush=True,
                    )
                    break

                print(
                    f"[DEBUG] continuing, leaving directory {os.getcwd()}", flush=True
                )
                os.chdir("../")
                print(f"[DEBUG] now in {os.getcwd()}", flush=True)
            # Component m (needs special handling in heme system)
            elif comp == "m" or comp == "a":
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists("rest"):
                    os.makedirs("rest")
                os.chdir("rest")
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if int(win) == 0:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_1dum,
                            second_cyp_dec=second_cyp_1dum,
                            first_cyp_next_dec=first_cyp_next_1dum,
                            second_cyp_next_dec=second_cyp_next_1dum,
                            first_cyp_previous_dec=first_cyp_previous_1dum,
                            second_cyp_previous_dec=second_cyp_previous_1dum,
                            heme_1=heme_1_1dum,
                            heme_2=heme_2_1dum,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        print("Creating box for protein/simultaneous release...")
                        build.create_box_cyp_equil(
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
                            first_cyp_equil=first_cyp_1dum,
                            second_cyp_equil=second_cyp_1dum,
                            first_cyp_next_equil=first_cyp_next_1dum,
                            second_cyp_next_equil=second_cyp_next_1dum,
                            first_cyp_previous_equil=first_cyp_previous_1dum,
                            second_cyp_previous_equil=second_cyp_previous_1dum,
                            heme_1=heme_1_1dum,
                            heme_2=heme_2_1dum,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                    else:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                if anch != "all":
                    break
                os.chdir("../")
            # Simultaneous//exchange
            elif comp in ("v", "e") and dec_method == "sdr":
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if dec_method == "dd":
                    if not os.path.exists(dec_method):
                        os.makedirs(dec_method)
                    os.chdir(dec_method)
                elif dec_method == "sdr" or dec_method == "exchange":
                    if not os.path.exists("sdr"):
                        os.makedirs("sdr")
                    os.chdir("sdr")
                for k in range(0, len(lambdas)):
                    weight = lambdas[k]
                    win = k
                    print("window: %s%02d lambda: %s" % (comp, int(win), str(weight)))
                    if int(win) == 0:
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        build.create_box_cyp_fe(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                    else:
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                if anch != "all":
                    break
                os.chdir("../")

            # double decoupling/exchange
            elif comp in ("v", "e") and dec_method == "dd":
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if dec_method == "dd":
                    if not os.path.exists(dec_method):
                        os.makedirs(dec_method)
                    os.chdir(dec_method)
                elif dec_method == "sdr" or dec_method == "exchange":
                    if not os.path.exists("sdr"):
                        os.makedirs("sdr")
                    os.chdir("sdr")
                for k in range(0, len(lambdas)):
                    weight = lambdas[k]
                    win = k
                    print("window: %s%02d lambda: %s" % (comp, int(win), str(weight)))
                    if int(win) == 0:
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_1dum,
                            second_cyp_dec=second_cyp_1dum,
                            first_cyp_next_dec=first_cyp_next_1dum,
                            second_cyp_next_dec=second_cyp_next_1dum,
                            first_cyp_previous_dec=first_cyp_previous_1dum,
                            second_cyp_previous_dec=second_cyp_previous_1dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        build.create_box_cyp_fe(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_1dum,
                            second_cyp_next_dec=second_cyp_next_1dum,
                            first_cyp_previous_dec=first_cyp_previous_1dum,
                            second_cyp_previous_dec=second_cyp_previous_1dum,
                            heme_1=heme_1_1dum,
                            heme_2=heme_2_1dum,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                    else:
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_1dum,
                            second_cyp_next_dec=second_cyp_next_1dum,
                            first_cyp_previous_dec=first_cyp_previous_1dum,
                            second_cyp_previous_dec=second_cyp_previous_1dum,
                            heme_1=heme_1_1dum,
                            heme_2=heme_2_1dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                if anch != "all":
                    break
                os.chdir("../")
            # Bulk systems for dd
            elif comp == "f" or comp == "w":
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists("dd"):
                    os.makedirs("dd")
                os.chdir("dd")
                for k in range(0, len(lambdas)):
                    weight = lambdas[k]
                    win = k
                    if int(win) == 0:
                        print(
                            "window: %s%02d lambda: %s" % (comp, int(win), str(weight))
                        )
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        print("Creating box for ligand decoupling in bulk...")
                        build.ligand_box(
                            mol, lig_buffer, water_model, neut, ion_def, comp, ligand_ff
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                    else:
                        print(
                            "window: %s%02d lambda: %s" % (comp, int(win), str(weight))
                        )
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                if anch != "all":
                    break
                os.chdir("../")
            elif comp == "x":
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists("sdr"):
                    os.makedirs("sdr")
                os.chdir("sdr")
                for k in range(0, len(lambdas)):
                    weight = lambdas[k]
                    win = k
                    print("window: %s%02d lambda: %s" % (comp, int(win), str(weight)))
                    if int(win) == 0:
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        print(
                            f"[DEBUG] comp={comp} win={win} build_dec returned anch={anch}",
                            flush=True,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        build.create_box_cyp_fe(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                    else:
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.dec_files(
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            weight,
                            lambdas,
                            dec_method,
                            ntwx,
                        )
                if anch != "all":
                    break

                os.chdir("../")
            # Attachments in the bound system
            else:
                steps1 = dic_steps1[comp]
                steps2 = dic_steps2[comp]
                if not os.path.exists("rest"):
                    os.makedirs("rest")
                os.chdir("rest")
                for k in range(0, len(attach_rest)):
                    weight = attach_rest[k]
                    win = k
                    if win == 0:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        anch = build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        if anch == "anch1":
                            aa1_poses.append(pose)
                            break
                        if anch == "anch2":
                            aa2_poses.append(pose)
                            break
                        if anch != "altm":
                            print("Creating box for attaching restraints...")
                            build.create_box_cyp_fe(
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
                                first_cyp_dec=first_cyp_2dum,
                                second_cyp_dec=second_cyp_2dum,
                                first_cyp_next_dec=first_cyp_next_2dum,
                                second_cyp_next_dec=second_cyp_next_2dum,
                                first_cyp_previous_dec=first_cyp_previous_2dum,
                                second_cyp_previous_dec=second_cyp_previous_2dum,
                                heme_1=heme_1,
                                heme_2=heme_2,
                            )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                    else:
                        print(
                            "window: %s%02d weight: %s" % (comp, int(win), str(weight))
                        )
                        build.build_dec(
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
                            first_cyp_dec=first_cyp_2dum,
                            second_cyp_dec=second_cyp_2dum,
                            first_cyp_next_dec=first_cyp_next_2dum,
                            second_cyp_next_dec=second_cyp_next_2dum,
                            first_cyp_previous_dec=first_cyp_previous_2dum,
                            second_cyp_previous_dec=second_cyp_previous_2dum,
                            heme_1=heme_1_2dum,
                            heme_2=heme_2_2dum,
                            sdr_axis=sdr_axis,
                        )
                        setup.restraints(
                            pose,
                            rest,
                            bb_start,
                            bb_end,
                            weight,
                            stage,
                            mol,
                            molr,
                            comp,
                            bb_equil,
                            sdr_dist,
                            dec_method,
                            other_mol,
                        )
                        setup.sim_files(
                            hmr,
                            temperature,
                            mol,
                            num_sim,
                            pose,
                            comp,
                            win,
                            stage,
                            steps1,
                            steps2,
                            rng,
                        )
                if anch == "anch1" or anch == "anch2":
                    break
                os.chdir("../")
        os.chdir("../")
    if len(aa1_poses) != 0:
        print("\n")
        print("WARNING: Could not find the ligand first anchor L1 for", aa1_poses)
        print(
            "The ligand most likely left the binding site during equilibration of these systems."
        )
        for i in aa1_poses:
            shutil.rmtree("./" + i + "")
    if len(aa2_poses) != 0:
        print("\n")
        print("WARNING: Could not find the ligand L2 or L3 anchors for", aa2_poses)
        print("Try reducing the min_adis parameter in the input file.")
        for i in aa2_poses:
            shutil.rmtree("./" + i + "")
elif stage == "analysis":
    # Free energy analysis for OpenMM
    if software == "openmm":
        for i in range(0, len(poses_def)):
            pose = poses_def[i]
            analysis.fe_openmm(
                components,
                temperature,
                pose,
                dec_method,
                rest,
                attach_rest,
                lambdas,
                dic_itera1,
                dic_itera2,
                itera_steps,
                dt,
                dlambda,
                dec_int,
                weights,
                blocks,
                ti_points,
            )
            os.chdir("../../")
    else:
        # Free energy analysis for AMBER20
        for i in range(0, len(poses_def)):
            pose = poses_def[i]
            analysis.fe_values(
                blocks,
                components,
                temperature,
                pose,
                attach_rest,
                lambdas,
                weights,
                dec_int,
                dec_method,
                rest,
                dic_steps1,
                dic_steps2,
                dt,
            )
            os.chdir("../../")


# Convert equilibration folders to openmm

if software == "openmm" and stage == "equil":

    # Adjust a few variables
    cut = str(float(cut) / 10)
    dt = str(float(dt) * 1000)

    os.chdir("equil")
    for i in range(0, len(poses_def)):
        mol = mols[i]
        pose = poses_def[i]
        rng = len(release_eq) - 1
        if os.path.exists(pose):
            print(pose)
            os.rename(pose, pose + "-amber")
            os.mkdir(pose)
            os.chdir(pose)
            shutil.copy("../" + pose + "-amber/equil-%s.pdb" % mol.lower(), "./")
            shutil.copy("../" + pose + "-amber/cv.in", "./")
            shutil.copy("../" + pose + "-amber/assign.dat", "./")
            for file in glob.glob("../" + pose + "-amber/vac*"):
                shutil.copy(file, "./")
            for file in glob.glob("../" + pose + "-amber/full*"):
                shutil.copy(file, "./")
            for file in glob.glob("../" + pose + "-amber/disang*"):
                shutil.copy(file, "./")
            for file in glob.glob("../" + pose + "-amber/build*"):
                shutil.copy(file, "./")
            for file in glob.glob("../" + pose + "-amber/tleap_solvate*"):
                shutil.copy(file, "./")
            fin = open("../../run_files/local-equil-op.bash", "rt")
            data = fin.read()
            data = data.replace("RANGE", "%02d" % rng)
            fin.close()
            fin = open("run-local.bash", "wt")
            fin.write(data)
            fin.close()
            fin = open("../../run_files/PBS-Op", "rt")
            data = fin.read()
            data = data.replace("STAGE", stage).replace("POSE", pose)
            fin.close()
            fin = open("PBS-run", "wt")
            fin.write(data)
            fin.close()
            fin = open("../../run_files/SLURMM-Op", "rt")
            data = fin.read()
            data = data.replace("STAGE", stage).replace("POSE", pose)
            fin.close()
            fin = open("SLURMM-run", "wt")
            fin.write(data)
            fin.close()
            for j in range(0, len(release_eq)):
                fin = open("../../lib/equil_heme.py", "rt")
                data = fin.read()
                data = (
                    data.replace("LIG", mol.upper())
                    .replace("TMPRT", str(temperature))
                    .replace("TSTP", str(dt))
                    .replace("GAMMA_LN", str(gamma_ln))
                    .replace("STG", "%02d" % j)
                    .replace("CTF", cut)
                )
                if hmr == "yes":
                    data = data.replace("PRMFL", "full.hmr.prmtop")
                else:
                    data = data.replace("PRMFL", "full.prmtop")
                if j == rng:
                    data = data.replace("TOTST", str(eq_steps2))
                else:
                    data = data.replace("TOTST", str(eq_steps1))
                fin.close()
                fin = open("equil-%02d.py" % j, "wt")
                fin.write(data)
                fin.close()
            os.chdir("../")
            shutil.rmtree("./" + pose + "-amber")
    print(os.getcwd())

if software == "openmm" and stage == "fe":

    # Redefine input arrays

    components = list(components_inp)
    attach_rest = list(attach_rest_inp)
    dec_method = dec_method_inp
    lambdas = list(lambdas_inp)
    lambdas_rest = []
    for i in attach_rest:
        lbd_rst = float(i) / float(100)
        lambdas_rest.append(lbd_rst)
    Input = lambdas_rest
    lambdas_rest = ["{:.5f}".format(elem) for elem in Input]

    # Start script

    print("")
    print("#############################")
    print("## OpenMM patch for BAT.py ##")
    print("#############################")
    print("")
    print("Components: ", components)
    print("")
    print("Decoupling lambdas: ", lambdas)
    print("")
    print("Restraint lambdas: ", lambdas_rest)
    print("")
    print("Integration Method: ", dec_int.upper())
    print("")

    # Generate folder and restraints for all components and windows
    for i in range(0, len(poses_def)):
        mol = mols[i]
        molr = mols[0]
        if not os.path.exists(poses_def[i]):
            continue
        os.chdir(poses_def[i])
        for j in range(0, len(components)):
            comp = components[j]
            if (
                comp == "a"
                or comp == "l"
                or comp == "t"
                or comp == "r"
                or comp == "c"
                or comp == "m"
                or comp == "n"
            ):
                if not os.path.exists("rest"):
                    os.makedirs("rest")
                os.chdir("rest")
                if not os.path.exists(comp + "-comp"):
                    os.makedirs(comp + "-comp")
                os.chdir(comp + "-comp")
                itera1 = dic_itera1[comp]
                itera2 = dic_itera2[comp]
                shutil.copy(
                    "../../../../run_files/local-rest-op.bash", "./run-local.bash"
                )
                fin = open("../../../../run_files/PBS-Op", "rt")
                data = fin.read()
                data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                fin.close()
                fin = open("PBS-run", "wt")
                fin.write(data)
                fin.close()
                fin = open("../../../../run_files/SLURMM-Op", "rt")
                data = fin.read()
                data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                fin.close()
                fin = open("SLURMM-run", "wt")
                fin.write(data)
                fin.close()
                fin = open("../../../../lib/rest_heme.py", "rt")
                data = fin.read()
                data = (
                    data.replace("LAMBDAS", "[%s]" % " , ".join(map(str, lambdas_rest)))
                    .replace("LIG", mol.upper())
                    .replace("TMPRT", str(temperature))
                    .replace("TSTP", str(dt))
                    .replace("SPITR", str(itera_steps))
                    .replace("PRIT", str(itera2))
                    .replace("EQIT", str(itera1))
                    .replace("ITCH", str(itcheck))
                    .replace("GAMMA_LN", str(gamma_ln))
                    .replace("CMPN", str(comp))
                    .replace("CTF", cut)
                    .replace("BLCKS", str(blocks))
                )
                if hmr == "yes":
                    data = data.replace("PRMFL", "full.hmr.prmtop")
                else:
                    data = data.replace("PRMFL", "full.prmtop")
                fin.close()
                fin = open("rest_heme.py", "wt")
                fin.write(data)
                fin.close()
                if comp == "c":
                    shutil.copy(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/rest/c00/disang.rest",
                        "./",
                    )
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/c00/full*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/c00/vac*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/c00/build*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/rest/c00/tleap_solvate*"
                    ):
                        shutil.copy(file, "./")
                elif comp == "n":
                    shutil.copy(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/rest/n00/disang.rest",
                        "./",
                    )
                    shutil.copy(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/n00/cv.in",
                        "./",
                    )
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/n00/full*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/n00/vac*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/n00/build*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/rest/n00/tleap_solvate*"
                    ):
                        shutil.copy(file, "./")
                else:
                    shutil.copy(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/rest/t00/disang.rest",
                        "./",
                    )
                    shutil.copy(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/t00/cv.in",
                        "./",
                    )
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/t00/full*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/t00/vac*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/rest/t00/build*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/rest/t00/tleap_solvate*"
                    ):
                        shutil.copy(file, "./")
                os.chdir("../../")
            elif comp == "e" or comp == "v" or comp == "w" or comp == "f":
                if dec_method == "sdr" or dec_method == "exchange":
                    if not os.path.exists("sdr"):
                        os.makedirs("sdr")
                    os.chdir("sdr")
                    if dec_int == "mbar":
                        if not os.path.exists(comp + "-comp"):
                            os.makedirs(comp + "-comp")
                        os.chdir(comp + "-comp")
                        itera1 = dic_itera1[comp]
                        itera2 = dic_itera2[comp]
                        shutil.copy(
                            "../../../../run_files/local-sdr-op.bash",
                            "./run-local.bash",
                        )
                        fin = open("../../../../run_files/PBS-Op", "rt")
                        data = fin.read()
                        data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                        fin.close()
                        fin = open("PBS-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open("../../../../run_files/SLURMM-Op", "rt")
                        data = fin.read()
                        data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                        fin.close()
                        fin = open("SLURMM-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open("../../../../lib/sdr_heme.py", "rt")
                        data = fin.read()
                        data = (
                            data.replace(
                                "LAMBDAS", "[%s]" % " , ".join(map(str, lambdas))
                            )
                            .replace("LIG", mol.upper())
                            .replace("LREF", molr.upper())
                            .replace("TMPRT", str(temperature))
                            .replace("TSTP", str(dt))
                            .replace("SPITR", str(itera_steps))
                            .replace("PRIT", str(itera2))
                            .replace("EQIT", str(itera1))
                            .replace("ITCH", str(itcheck))
                            .replace("GAMMA_LN", str(gamma_ln))
                            .replace("CMPN", str(comp))
                            .replace("CTF", cut)
                            .replace("BLCKS", str(blocks))
                        )
                        if hmr == "yes":
                            data = data.replace("PRMFL", "full.hmr.prmtop")
                        else:
                            data = data.replace("PRMFL", "full.prmtop")
                        fin.close()
                        fin = open("sdr_heme.py", "wt")
                        fin.write(data)
                        fin.close()
                        shutil.copy(
                            "../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/v00/disang.rest",
                            "./",
                        )
                        shutil.copy(
                            "../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/v00/cv.in",
                            "./",
                        )
                        for file in glob.glob(
                            "../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/v00/full*"
                        ):
                            shutil.copy(file, "./")
                        for file in glob.glob(
                            "../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/v00/vac*"
                        ):
                            shutil.copy(file, "./")
                        for file in glob.glob(
                            "../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/v00/tleap_solvate*"
                        ):
                            shutil.copy(file, "./")
                        for file in glob.glob(
                            "../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/v00/build*"
                        ):
                            shutil.copy(file, "./")
                        os.chdir("../")
                    elif dec_int == "ti":
                        if not os.path.exists(comp + "-comp"):
                            os.makedirs(comp + "-comp")
                        os.chdir(comp + "-comp")
                        itera1 = int(dic_itera1[comp] * itera_steps)
                        itera2 = int(dic_itera2[comp] / 2)
                        for k in range(0, len(lambdas)):
                            if not os.path.exists("%s%02d" % (comp, int(k))):
                                os.makedirs("%s%02d" % (comp, int(k)))
                            os.chdir("%s%02d" % (comp, int(k)))
                            shutil.copy(
                                "../../../../../run_files/local-sdr-op-ti.bash",
                                "./run-local.bash",
                            )
                            fin = open("../../../../../run_files/SLURMM-Op", "rt")
                            data = fin.read()
                            data = data.replace("STAGE", poses_def[i]).replace(
                                "POSE", "%s%02d" % (comp, int(k))
                            )
                            fin.close()
                            fin = open("SLURMM-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open("../../../../../run_files/PBS-Op", "rt")
                            data = fin.read()
                            data = data.replace("STAGE", poses_def[i]).replace(
                                "POSE", "%s%02d" % (comp, int(k))
                            )
                            fin.close()
                            fin = open("PBS-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open("../../../../../lib/equil-sdr_heme.py", "rt")
                            data = fin.read()
                            data = (
                                data.replace("LBD0", "%8.6f" % lambdas[k])
                                .replace("LIG", mol.upper())
                                .replace("LREF", molr.upper())
                                .replace("TMPRT", str(temperature))
                                .replace("TSTP", str(dt))
                                .replace("SPITR", str(itera_steps))
                                .replace("PRIT", str(itera2))
                                .replace("EQIT", str(itera1))
                                .replace("ITCH", str(itcheck))
                                .replace("GAMMA_LN", str(gamma_ln))
                                .replace("CMPN", str(comp))
                                .replace("CTF", cut)
                            )
                            if hmr == "yes":
                                data = data.replace("PRMFL", "full.hmr.prmtop")
                            else:
                                data = data.replace("PRMFL", "full.prmtop")
                            fin.close()
                            fin = open("equil-sdr_heme.py", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open("../../../../../lib/sdr-ti_heme.py", "rt")
                            data = fin.read()
                            # "Split" initial lambda into two close windows
                            lambda1 = float(lambdas[k] - dlambda / 2)
                            lambda2 = float(lambdas[k] + dlambda / 2)
                            data = (
                                data.replace("LBD1", "%8.6f" % lambda1)
                                .replace("LBD2", "%8.6f" % lambda2)
                                .replace("LIG", mol.upper())
                                .replace("LREF", molr.upper())
                                .replace("TMPRT", str(temperature))
                                .replace("TSTP", str(dt))
                                .replace("SPITR", str(itera_steps))
                                .replace("PRIT", str(itera2))
                                .replace("EQIT", str(itera1))
                                .replace("ITCH", str(itcheck))
                                .replace("GAMMA_LN", str(gamma_ln))
                                .replace("CMPN", str(comp))
                                .replace("CTF", cut)
                                .replace("BLCKS", str(blocks))
                            )
                            if hmr == "yes":
                                data = data.replace("PRMFL", "full.hmr.prmtop")
                            else:
                                data = data.replace("PRMFL", "full.prmtop")
                            fin.close()
                            fin = open("sdr-ti_heme.py", "wt")
                            fin.write(data)
                            fin.close()
                            shutil.copy(
                                "../../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/sdr/v00/disang.rest",
                                "./",
                            )
                            shutil.copy(
                                "../../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/sdr/v00/cv.in",
                                "./",
                            )
                            for file in glob.glob(
                                "../../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/sdr/v00/full*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/sdr/v00/vac*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/sdr/v00/tleap_solvate*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/sdr/v00/build*"
                            ):
                                shutil.copy(file, "./")
                            os.chdir("../")
                        os.chdir("../")
                elif dec_method == "dd":
                    if not os.path.exists("dd"):
                        os.makedirs("dd")
                    os.chdir("dd")
                    if dec_int == "mbar":
                        if not os.path.exists(comp + "-comp"):
                            os.makedirs(comp + "-comp")
                        os.chdir(comp + "-comp")
                        itera1 = dic_itera1[comp]
                        itera2 = dic_itera2[comp]
                        if not os.path.exists("../run_files"):
                            shutil.copytree("../../../../run_files", "../run_files")
                        shutil.copy(
                            "../../../../run_files/local-dd-op.bash", "./run-local.bash"
                        )
                        fin = open("../../../../run_files/PBS-Op", "rt")
                        data = fin.read()
                        data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                        fin.close()
                        fin = open("PBS-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open("../../../../run_files/SLURMM-Op", "rt")
                        data = fin.read()
                        data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                        fin.close()
                        fin = open("SLURMM-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open("../../../../lib/dd.py", "rt")
                        data = fin.read()
                        data = (
                            data.replace(
                                "LAMBDAS", "[%s]" % " , ".join(map(str, lambdas))
                            )
                            .replace("LIG", mol.upper())
                            .replace("TMPRT", str(temperature))
                            .replace("TSTP", str(dt))
                            .replace("SPITR", str(itera_steps))
                            .replace("PRIT", str(itera2))
                            .replace("EQIT", str(itera1))
                            .replace("ITCH", str(itcheck))
                            .replace("GAMMA_LN", str(gamma_ln))
                            .replace("CMPN", str(comp))
                            .replace("CTF", cut)
                            .replace("BLCKS", str(blocks))
                        )
                        if hmr == "yes":
                            data = data.replace("PRMFL", "full.hmr.prmtop")
                        else:
                            data = data.replace("PRMFL", "full.prmtop")
                        fin.close()
                        fin = open("dd.py", "wt")
                        fin.write(data)
                        fin.close()
                        if comp == "f" or comp == "w":
                            shutil.copy(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/c00/disang.rest",
                                "./",
                            )
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/c00/full*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/c00/vac*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/c00/tleap_solvate*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/c00/build*"
                            ):
                                shutil.copy(file, "./")
                        else:
                            shutil.copy(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/t00/disang.rest",
                                "./",
                            )
                            shutil.copy(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/t00/cv.in",
                                "./",
                            )
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/t00/full*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/t00/vac*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/t00/tleap_solvate*"
                            ):
                                shutil.copy(file, "./")
                            for file in glob.glob(
                                "../../../../"
                                + stage
                                + "/"
                                + poses_def[i]
                                + "/rest/t00/build*"
                            ):
                                shutil.copy(file, "./")
                        os.chdir("../")
                    elif dec_int == "ti":
                        if not os.path.exists(comp + "-comp"):
                            os.makedirs(comp + "-comp")
                        os.chdir(comp + "-comp")
                        itera1 = int(dic_itera1[comp] * itera_steps)
                        itera2 = int(dic_itera2[comp] / 2)
                        for k in range(0, len(lambdas)):
                            if not os.path.exists("%s%02d" % (comp, int(k))):
                                os.makedirs("%s%02d" % (comp, int(k)))
                            os.chdir("%s%02d" % (comp, int(k)))
                            shutil.copy(
                                "../../../../../run_files/local-dd-op-ti.bash",
                                "./run-local.bash",
                            )
                            fin = open("../../../../../run_files/SLURMM-Op", "rt")
                            data = fin.read()
                            data = data.replace("STAGE", poses_def[i]).replace(
                                "POSE", "%s%02d" % (comp, int(k))
                            )
                            fin.close()
                            fin = open("SLURMM-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open("../../../../../run_files/PBS-Op", "rt")
                            data = fin.read()
                            data = data.replace("STAGE", poses_def[i]).replace(
                                "POSE", "%s%02d" % (comp, int(k))
                            )
                            fin.close()
                            fin = open("PBS-run", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open("../../../../../lib/equil-dd.py", "rt")
                            data = fin.read()
                            data = (
                                data.replace("LBD0", "%8.6f" % lambdas[k])
                                .replace("LIG", mol.upper())
                                .replace("TMPRT", str(temperature))
                                .replace("TSTP", str(dt))
                                .replace("SPITR", str(itera_steps))
                                .replace("PRIT", str(itera2))
                                .replace("EQIT", str(itera1))
                                .replace("ITCH", str(itcheck))
                                .replace("GAMMA_LN", str(gamma_ln))
                                .replace("CMPN", str(comp))
                                .replace("CTF", cut)
                            )
                            if hmr == "yes":
                                data = data.replace("PRMFL", "full.hmr.prmtop")
                            else:
                                data = data.replace("PRMFL", "full.prmtop")
                            fin.close()
                            fin = open("equil-dd.py", "wt")
                            fin.write(data)
                            fin.close()
                            fin = open("../../../../../lib/dd-ti.py", "rt")
                            data = fin.read()
                            # "Split" initial lambda into two close windows
                            lambda1 = float(lambdas[k] - dlambda / 2)
                            lambda2 = float(lambdas[k] + dlambda / 2)
                            data = (
                                data.replace("LBD1", "%8.6f" % lambda1)
                                .replace("LBD2", "%8.6f" % lambda2)
                                .replace("LIG", mol.upper())
                                .replace("TMPRT", str(temperature))
                                .replace("TSTP", str(dt))
                                .replace("SPITR", str(itera_steps))
                                .replace("PRIT", str(itera2))
                                .replace("EQIT", str(itera1))
                                .replace("ITCH", str(itcheck))
                                .replace("GAMMA_LN", str(gamma_ln))
                                .replace("CMPN", str(comp))
                                .replace("CTF", cut)
                                .replace("BLCKS", str(blocks))
                            )
                            if hmr == "yes":
                                data = data.replace("PRMFL", "full.hmr.prmtop")
                            else:
                                data = data.replace("PRMFL", "full.prmtop")
                            fin.close()
                            fin = open("dd-ti.py", "wt")
                            fin.write(data)
                            fin.close()
                            if comp == "f" or comp == "w":
                                shutil.copy(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/c00/disang.rest",
                                    "./",
                                )
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/c00/full*"
                                ):
                                    shutil.copy(file, "./")
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/c00/vac*"
                                ):
                                    shutil.copy(file, "./")
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/c00/tleap_solvate*"
                                ):
                                    shutil.copy(file, "./")
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/c00/build*"
                                ):
                                    shutil.copy(file, "./")
                            else:
                                shutil.copy(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/t00/disang.rest",
                                    "./",
                                )
                                shutil.copy(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/t00/cv.in",
                                    "./",
                                )
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/t00/full*"
                                ):
                                    shutil.copy(file, "./")
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/t00/vac*"
                                ):
                                    shutil.copy(file, "./")
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/t00/tleap_solvate*"
                                ):
                                    shutil.copy(file, "./")
                                for file in glob.glob(
                                    "../../../../../"
                                    + stage
                                    + "/"
                                    + poses_def[i]
                                    + "/rest/t00/build*"
                                ):
                                    shutil.copy(file, "./")
                            os.chdir("../")
                        os.chdir("../")
                os.chdir("../")
            elif comp == "x":
                if not os.path.exists("sdr"):
                    os.makedirs("sdr")
                os.chdir("sdr")
                if dec_int == "mbar":
                    if not os.path.exists(comp + "-comp"):
                        os.makedirs(comp + "-comp")
                    os.chdir(comp + "-comp")
                    itera1 = dic_itera1[comp]
                    itera2 = dic_itera2[comp]
                    shutil.copy(
                        "../../../../run_files/local-sdr-op.bash", "./run-local.bash"
                    )
                    fin = open("../../../../run_files/PBS-Op", "rt")
                    data = fin.read()
                    data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                    fin.close()
                    fin = open("PBS-run", "wt")
                    fin.write(data)
                    fin.close()
                    fin = open("../../../../run_files/SLURMM-Op", "rt")
                    data = fin.read()
                    data = data.replace("POSE", comp).replace("STAGE", poses_def[i])
                    fin.close()
                    fin = open("SLURMM-run", "wt")
                    fin.write(data)
                    fin.close()
                    fin = open("../../../../lib/sdr_heme.py", "rt")
                    data = fin.read()
                    data = (
                        data.replace("LAMBDAS", "[%s]" % " , ".join(map(str, lambdas)))
                        .replace("LIG", mol.upper())
                        .replace("LREF", molr.upper())
                        .replace("TMPRT", str(temperature))
                        .replace("TSTP", str(dt))
                        .replace("SPITR", str(itera_steps))
                        .replace("PRIT", str(itera2))
                        .replace("EQIT", str(itera1))
                        .replace("ITCH", str(itcheck))
                        .replace("GAMMA_LN", str(gamma_ln))
                        .replace("CMPN", str(comp))
                        .replace("CTF", cut)
                        .replace("BLCKS", str(blocks))
                    )
                    if hmr == "yes":
                        data = data.replace("PRMFL", "full.hmr.prmtop")
                    else:
                        data = data.replace("PRMFL", "full.prmtop")
                    fin.close()
                    fin = open("sdr_heme.py", "wt")
                    fin.write(data)
                    fin.close()
                    shutil.copy(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/sdr/x00/disang.rest",
                        "./",
                    )
                    shutil.copy(
                        "../../../../" + stage + "/" + poses_def[i] + "/sdr/x00/cv.in",
                        "./",
                    )
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/sdr/x00/full*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/sdr/x00/vac*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../"
                        + stage
                        + "/"
                        + poses_def[i]
                        + "/sdr/x00/tleap_solvate*"
                    ):
                        shutil.copy(file, "./")
                    for file in glob.glob(
                        "../../../../" + stage + "/" + poses_def[i] + "/sdr/x00/build*"
                    ):
                        shutil.copy(file, "./")
                    os.chdir("../")
                elif dec_int == "ti":
                    if not os.path.exists(comp + "-comp"):
                        os.makedirs(comp + "-comp")
                    os.chdir(comp + "-comp")
                    itera1 = int(dic_itera1[comp] * itera_steps)
                    itera2 = int(dic_itera2[comp] / 2)
                    for k in range(0, len(lambdas)):
                        if not os.path.exists("%s%02d" % (comp, int(k))):
                            os.makedirs("%s%02d" % (comp, int(k)))
                        os.chdir("%s%02d" % (comp, int(k)))
                        shutil.copy(
                            "../../../../../run_files/local-sdr-op-ti.bash",
                            "./run-local.bash",
                        )
                        fin = open("../../../../../run_files/SLURMM-Op", "rt")
                        data = fin.read()
                        data = data.replace("STAGE", poses_def[i]).replace(
                            "POSE", "%s%02d" % (comp, int(k))
                        )
                        fin.close()
                        fin = open("SLURMM-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open("../../../../../run_files/PBS-Op", "rt")
                        data = fin.read()
                        data = data.replace("STAGE", poses_def[i]).replace(
                            "POSE", "%s%02d" % (comp, int(k))
                        )
                        fin.close()
                        fin = open("PBS-run", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open("../../../../../lib/equil-sdr_heme.py", "rt")
                        data = fin.read()
                        data = (
                            data.replace("LBD0", "%8.6f" % lambdas[k])
                            .replace("LIG", mol.upper())
                            .replace("LREF", molr.upper())
                            .replace("TMPRT", str(temperature))
                            .replace("TSTP", str(dt))
                            .replace("SPITR", str(itera_steps))
                            .replace("PRIT", str(itera2))
                            .replace("EQIT", str(itera1))
                            .replace("ITCH", str(itcheck))
                            .replace("GAMMA_LN", str(gamma_ln))
                            .replace("CMPN", str(comp))
                            .replace("CTF", cut)
                        )
                        if hmr == "yes":
                            data = data.replace("PRMFL", "full.hmr.prmtop")
                        else:
                            data = data.replace("PRMFL", "full.prmtop")
                        fin.close()
                        fin = open("equil-sdr_heme.py", "wt")
                        fin.write(data)
                        fin.close()
                        fin = open("../../../../../lib/sdr-ti_heme.py", "rt")
                        data = fin.read()
                        # "Split" initial lambda into two close windows
                        lambda1 = float(lambdas[k] - dlambda / 2)
                        lambda2 = float(lambdas[k] + dlambda / 2)
                        data = (
                            data.replace("LBD1", "%8.6f" % lambda1)
                            .replace("LBD2", "%8.6f" % lambda2)
                            .replace("LIG", mol.upper())
                            .replace("LREF", molr.upper())
                            .replace("TMPRT", str(temperature))
                            .replace("TSTP", str(dt))
                            .replace("SPITR", str(itera_steps))
                            .replace("PRIT", str(itera2))
                            .replace("EQIT", str(itera1))
                            .replace("ITCH", str(itcheck))
                            .replace("GAMMA_LN", str(gamma_ln))
                            .replace("CMPN", str(comp))
                            .replace("CTF", cut)
                            .replace("BLCKS", str(blocks))
                        )
                        if hmr == "yes":
                            data = data.replace("PRMFL", "full.hmr.prmtop")
                        else:
                            data = data.replace("PRMFL", "full.prmtop")
                        fin.close()
                        fin = open("sdr-ti.py", "wt")
                        fin.write(data)
                        fin.close()
                        shutil.copy(
                            "../../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/x00/disang.rest",
                            "./",
                        )
                        shutil.copy(
                            "../../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/x00/cv.in",
                            "./",
                        )
                        for file in glob.glob(
                            "../../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/x00/full*"
                        ):
                            shutil.copy(file, "./")
                        for file in glob.glob(
                            "../../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/x00/vac*"
                        ):
                            shutil.copy(file, "./")
                        for file in glob.glob(
                            "../../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/x00/tleap_solvate*"
                        ):
                            shutil.copy(file, "./")
                        for file in glob.glob(
                            "../../../../../"
                            + stage
                            + "/"
                            + poses_def[i]
                            + "/sdr/x00/build*"
                        ):
                            shutil.copy(file, "./")
                        os.chdir("../")
                    os.chdir("../")
                os.chdir("../")
        print(os.getcwd())

        # Clean up amber windows
        dirpath = os.path.join("rest", "t00")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join("rest", "amber_files")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join("rest", "c00")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join("rest", "n00")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join("sdr", "v00")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join("sdr", "amber_files")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        #    dirpath = os.path.join('sdr', 'x00')
        #    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        #      shutil.rmtree(dirpath)
        os.chdir("../")
