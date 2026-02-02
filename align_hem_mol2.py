#!/usr/bin/env python3
"""
align_mol2_by_refs.py — rotate+translate a MOL2 so named atoms land at user-specified target coordinates.

You can pass anchors either with dedicated flags (--fe/--na/--nc) or as many --ref entries as you like.
Each --ref can be written as:
  NAME:x y z         or  NAME:x,y,z         or  NAME x y z

Examples
--------
# Old-style (3 anchors)
python align_mol2_by_refs.py HEM.mol2 \
  --fe "57.760 20.828 86.683" \
  --na "58.861 21.205 84.959" \
  --nc "55.991 20.480 87.682" \
  -o HEM_aligned.mol2

# Add more anchors (recommended): C2A and C3D
python align_mol2_by_refs.py HEM.mol2 \
  --ref FE:57.760,20.828,86.683 \
  --ref NA:58.861,21.205,84.959 \
  --ref NC:55.991,20.480,87.682 \
  --ref C2A:60.259,20.988,83.123 \
  --ref C3D:58.388,16.837,85.272 \
  -o HEM_aligned.mol2

Notes
-----
- Atom-name matching is case-insensitive.
- Needs numpy.
"""

import argparse, re, sys
import numpy as np

TRIPOS_ATOM = "@<TRIPOS>ATOM"
SECTION_RE = re.compile(r"^@<TRIPOS>")


def parse_vec(s: str) -> np.ndarray:
    s = s.replace(",", " ").split()
    if len(s) != 3:
        raise ValueError("need 3 numbers")
    return np.array([float(s[0]), float(s[1]), float(s[2])], dtype=float)


def parse_ref_entry(s: str):
    """
    Accept 'NAME:x y z', 'NAME:x,y,z', or 'NAME x y z' (first token is NAME).
    Returns (NAME_UPPER, np.array([x,y,z])).
    """
    if ":" in s:
        name, coords = s.split(":", 1)
    else:
        parts = s.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Cannot parse --ref entry: {s}")
        name, coords = parts[0], parts[1]
    return name.strip().upper(), parse_vec(coords.strip())


def find_atom_section(lines):
    start = None
    for i, line in enumerate(lines):
        if line.strip() == TRIPOS_ATOM:
            start = i + 1
            break
    if start is None:
        raise SystemExit("No @<TRIPOS>ATOM section found.")
    end = len(lines)
    for j in range(start, len(lines)):
        if SECTION_RE.match(lines[j]) and j > start:
            end = j
            break
    return start, end


def parse_atom_line(line):
    # atom_id atom_name x y z atom_type subst_id subst_name charge [status_bits]
    parts = line.rstrip("\n").split()
    if len(parts) < 8:
        return None
    try:
        atom_id = int(parts[0])
        name = parts[1]
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        return atom_id, name, np.array([x, y, z], dtype=float), parts
    except Exception:
        return None


def kabsch(P, Q):
    """
    Find R,t minimizing sum ||R*P_i + t - Q_i||^2
    P,Q: (N,3). Returns R (3x3), t (3,)
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - R @ Pc
    return R, t


def main():
    ap = argparse.ArgumentParser(
        description="Rigidly align a MOL2 so selected atoms match target coordinates."
    )
    ap.add_argument("mol2", help="Input MOL2")
    ap.add_argument("-o", "--out", default="HEM_aligned.mol2", help="Output MOL2")

    # Convenience flags for the common trio
    ap.add_argument("--fe", help="Target FE xyz: 'x y z' or 'x,y,z'")
    ap.add_argument("--na", help="Target NA xyz")
    ap.add_argument("--nc", help="Target NC xyz")

    # General anchors (can repeat)
    ap.add_argument(
        "--ref",
        action="append",
        default=[],
        help="Anchor entry like 'NAME:x y z' or 'NAME:x,y,z' or 'NAME x y z'. May be given multiple times.",
    )

    args = ap.parse_args()

    # Build target dict from all sources
    targets = {}
    if args.fe:
        targets["FE"] = parse_vec(args.fe)
    if args.na:
        targets["NA"] = parse_vec(args.na)
    if args.nc:
        targets["NC"] = parse_vec(args.nc)
    for entry in args.ref:
        name, vec = parse_ref_entry(entry)
        targets[name] = vec

    if len(targets) < 3:
        raise SystemExit(
            "Provide at least 3 anchors (e.g., --fe --na --nc, or multiple --ref NAME:coords)."
        )

    with open(args.mol2, "r", errors="ignore") as f:
        lines = f.readlines()

    a_start, a_end = find_atom_section(lines)

    # Collect atom records and locate anchors
    atom_records = []  # (line_idx, parts, xyz)
    found = {}
    for i in range(a_start, a_end):
        rec = parse_atom_line(lines[i])
        if not rec:
            continue
        _, name, xyz, parts = rec
        atom_records.append((i, parts, xyz))
        key = name.strip().upper()
        if key in targets and key not in found:
            found[key] = xyz

    missing = [k for k in targets if k not in found]
    if missing:
        raise SystemExit(f"Anchor atom(s) not found in MOL2: {', '.join(missing)}")

    # Build P (from MOL2) and Q (targets) with consistent ordering
    names_order = list(targets.keys())
    P = np.vstack([found[n] for n in names_order])
    Q = np.vstack([targets[n] for n in names_order])

    # Basic degeneracy check: require at least rank-2 geometry
    if (
        np.linalg.matrix_rank(P - P.mean(axis=0)) < 2
        or np.linalg.matrix_rank(Q - Q.mean(axis=0)) < 2
    ):
        raise SystemExit(
            "Anchor sets are nearly collinear. Choose three non-collinear atoms (add e.g. C2A, C3D)."
        )

    R, t = kabsch(P, Q)

    # Apply to all atoms
    def fmt(v):
        return f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}"

    for i, parts, xyz in atom_records:
        new_xyz = R @ xyz + t
        parts[2], parts[3], parts[4] = fmt(new_xyz)
        lines[i] = " ".join(parts) + "\n"

    with open(args.out, "w") as f:
        f.writelines(lines)

    # Report fit quality
    P_new = (R @ P.T).T + t
    per_anchor = {n: np.linalg.norm(P_new[k] - Q[k]) for k, n in enumerate(names_order)}
    rmsd = np.sqrt(((P_new - Q) ** 2).sum(axis=1).mean())
    print(f"✅ Wrote {args.out}")
    print(
        f"   rotation det = {np.linalg.det(R):.6f}, translation = ({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})"
    )
    print(f"   anchor RMSD = {rmsd:.5f} Å")
    for n in names_order:
        print(f"   {n}: {per_anchor[n]:.5f} Å")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
