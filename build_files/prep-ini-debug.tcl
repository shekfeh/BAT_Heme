# ===== DEBUG-ENHANCED PREP (keeps original behavior) =========================
# Original variables/placeholders (filled by your driver):
#  MMM  P1A  NN  FIRST  LAST  XDIS  YDIS  ZDIS  RANG  DMAX  DMIN  SDRD
#
# Files you get in addition to the originals:
#   - prep_debug.log
#   - candidates_L1.txt  (ligand atom -> distance to displaced protein anchor)
#   - candidates_L2.txt  (ligand atom -> angle(L1_anchor, protein_anchor, atom), length)
#   - candidates_L3.txt  (ligand atom -> angle(L2_anchor, L1_anchor, atom), length)
#   - l1_sphere.pdb      (all atoms within RANG of displaced protein anchor point)
#   - l1_candidates.pdb  (ligand heavy atoms within RANG of that point)
#
# ---------------------------------------------------------------------------

# --- open debug log ---
set _dbgfp [open "prep_debug.log" "w"]
proc wlog {fp msg} { puts $fp $msg; puts $msg; flush $fp }

# --- load & recenter exactly as original ------------------------------------
mol load pdb aligned_amber.pdb
set filini STAGE-mmm-ini.pdb
set filpdb STAGE-mmm.pdb
set n 0
set rad 100
set xd XDIS 
set yd YDIS 
set zd ZDIS
set ran RANG
set dmax DMAX
set dmin DMIN
set mat {}
set sdr_dist SDRD

# Helpful header
wlog $_dbgfp "PARAMS: XDIS=$xd YDIS=$yd ZDIS=$zd  RANG=$ran  DMIN=$dmin  DMAX=$dmax  SDRD=$sdr_dist"
wlog $_dbgfp "Ligand = {resname MMM}; Protein anchor = {(not resname MMM) and (resid P1A and name NN)}"
wlog $_dbgfp "Resid window for recenter: FIRST..LAST"

set pr  [atomselect 0 "(not resname MMM) and (resid FIRST to LAST and name CA C N O)"]
set all [atomselect 0 "(resid FIRST to LAST and not water and not resname MMM and noh) or (resname MMM) or (resname OTHRS WAT)"]
$all moveby [vecinvert [measure center $pr weight mass]]
$all writepdb $filini
mol delete all
mol load pdb $filini
set all [atomselect 1 all]
$all writepdb $filpdb

# --- quick inventory & ligand dump ------------------------------------------
set sel_all  [atomselect 1 all]
set sel_prot [atomselect 1 "protein"]
set sel_wat  [atomselect 1 "water or resname WAT HOH"]
set sel_ion  [atomselect 1 "resname NA K CL CA MG ZN"]
set resnames [lsort -unique [$sel_all get resname]]

wlog $_dbgfp "COUNTS: total=[$sel_all num]  protein=[$sel_prot num]  water=[$sel_wat num]  ions=[$sel_ion num]"
wlog $_dbgfp "RESNAMES: $resnames"

set lig  [atomselect 1 "resname MMM"]
set ligh [atomselect 1 "resname MMM and noh"]
wlog $_dbgfp "Ligand atoms: [$lig num], (noh)=[\$ligh num]"
$lig set chain S
$lig set resid 1
$lig writepdb mmm.pdb
$ligh writepdb mmm-noh.pdb

# --- sanity: protein anchor selection exists? --------------------------------
set _prot_anchor_sel "(not resname MMM) and (resid P1A and name NN)"
set _pcheck [atomselect 1 $_prot_anchor_sel]
if {[$_pcheck num] == 0} {
    wlog $_dbgfp "ERROR: protein anchor selection '$_prot_anchor_sel' matched 0 atoms."
    # preserve original failure mode: empty anchors.txt
    set _fh [open "anchors.txt" "w"]; close $_fh
    wlog $_dbgfp "WROTE empty anchors.txt due to missing protein anchor atom."
    close $_dbgfp
    exit
}
$_pcheck delete

# --- L1 search: per-atom distances & candidates file -------------------------
set a [atomselect 1 "resname MMM and noh"]
set tot [$a get name]

# displaced protein anchor point (used repeatedly)
# (we recompute inside loops as in your original, but we’ll also dump a sphere)
# dump sphere and ligand-in-range visuals
# (use a representative displaced point from the anchor atom)
set _p [atomselect 1 $_prot_anchor_sel]
set _pctr [measure center $_p weight mass]
foreach {x2 y2 z2} $_pctr {break}
set xl [expr {$x2+$xd}]
set yl [expr {$y2+$yd}]
set zl [expr {$z2+$zd}]
set sphere [atomselect 1 "sqrt((x-$xl)^2+(y-$yl)^2+(z-$zl)^2) < $ran"]
$sphere writepdb l1_sphere.pdb
set lig_in [atomselect 1 "resname MMM and noh and within $ran of point {$xl $yl $zl}"]
$lig_in writepdb l1_candidates.pdb
$sphere delete
$lig_in delete
$_p delete

# write detailed L1 candidates
set c1 [open "candidates_L1.txt" "w"]
puts $c1 "#name x y z  displaced_anchor_x displaced_anchor_y displaced_anchor_z  dist  in_range(<RANG)"
set mat {}
foreach i $tot {
    set t [atomselect 1 "resname MMM and name $i"]
    set p [atomselect 1 $_prot_anchor_sel]
    set d1 [measure center $t weight mass]
    set d2 [measure center $p weight mass]
    foreach {x2 y2 z2} $d2 {break}
    set xl [expr {$x2+$xd}]
    set yl [expr {$y2+$yd}]
    set zl [expr {$z2+$zd}]
    foreach {x1 y1 z1} $d1 {break}
    set xc [expr {abs($x1-$xl)}]
    set yc [expr {abs($y1-$yl)}]
    set zc [expr {abs($z1-$zl)}]
    set dist [expr {sqrt($xc*$xc + $yc*$yc + $zc*$zc)}]

    puts $i
    puts $dist

    set inr [expr {$dist < $ran}]
    puts $c1 "$i $x1 $y1 $z1  $xl $yl $zl  $dist  $inr"
    if {$inr} { lappend mat $i }

    $t delete
    $p delete
}
close $c1
wlog $_dbgfp "L1 candidates within RANG=$ran Å: [llength $mat]  (see candidates_L1.txt)"

# pick nearest (original logic)
foreach i $mat {
    set t [atomselect 1 "resname MMM and name $i"]
    set p [atomselect 1 $_prot_anchor_sel]
    set d1 [measure center $t weight mass]
    set d2 [measure center $p weight mass]
    foreach {x2 y2 z2} $d2 {break}
    set xl [expr {$x2+$xd}]
    set yl [expr {$y2+$yd}]
    set zl [expr {$z2+$zd}]
    foreach {x1 y1 z1} $d1 {break}
    set xc [expr {abs($x1-$xl)}]
    set yc [expr {abs($y1-$yl)}]
    set zc [expr {abs($z1-$zl)}]
    set diff [expr {sqrt($xc*$xc + $yc*$yc + $zc*$zc)}]
    if {$diff < $rad} { set rad $diff; set aa1 $i }
    $t delete
    $p delete
}

set exist [info exists aa1]
if {[expr {$exist == 0}]} {
    # preserve original failure mode
    set filename "anchors.txt"
    set fileId [open $filename "w"]; close $fileId
    wlog $_dbgfp "FAIL: Ligand first anchor not found (mat empty). Wrote empty anchors.txt"
    close $_dbgfp
    puts "Ligand first anchor not found"
    exit
}

puts "anchor 1 is" 
puts $aa1
puts $rad
puts ""
wlog $_dbgfp "Chosen L1: $aa1 (nearest distance=$rad Å)"

# --- L2 search: angle ~90 with protein anchor; length in [DMIN,DMAX] --------
set amat {}
set c2 [open "candidates_L2.txt" "w"]
puts $c2 "#name angle_deg length(aa1->i)  passed_length_window(DMIN..DMAX)"
foreach i $tot {
    set alis {}
    set angle1 {}; set angle2 {}; set angle3 {}; set angle {}
    set t [atomselect 1 "resname MMM and name $i"]
    set p [atomselect 1 "resname MMM and name $aa1"]
    set d [atomselect 1 $_prot_anchor_sel]
    if {$i ne $aa1} {
        set a1 [$d get index]
        set d1 [measure center $t weight mass]
        set d2 [measure center $p weight mass]
        set leng [veclength [vecsub $d1 $d2]]
        lappend angle1 $a1; lappend angle1 "1"; lappend angle $angle1
        lappend angle2 [$p get index]; lappend angle2 "1"; lappend angle $angle2
        lappend angle3 [$t get index]; lappend angle3 "1"; lappend angle $angle3
        set ang [measure angle $angle]
        puts $i
        puts $ang
        puts $leng
        set passLen [expr {($leng > $dmin) && ($leng < $dmax)}]
        puts $c2 "$i $ang $leng $passLen"
        if {$passLen} { lappend amat $i }
    }
    $t delete; $p delete; $d delete
}
close $c2
wlog $_dbgfp "L2 length-window candidates: [llength $amat] (see candidates_L2.txt)"

set amx 90
foreach i $amat {
    set angle1 {}; set angle2 {}; set angle3 {}; set angle {}
    set t [atomselect 1 "resname MMM and name $i"]
    set p [atomselect 1 "resname MMM and name $aa1"]
    set d [atomselect 1 $_prot_anchor_sel]
    set d1 [measure center $t weight mass]
    set d2 [measure center $p weight mass]
    lappend angle1 [$d get index]; lappend angle1 "1"; lappend angle $angle1
    lappend angle2 [$p get index]; lappend angle2 "1"; lappend angle $angle2
    lappend angle3 [$t get index]; lappend angle3 "1"; lappend angle $angle3
    set ang [measure angle $angle]
    if {[expr {abs($ang - 90.0)}] < $amx} {
        set amx [expr {abs($ang - 90.0)}]
        set angl $ang
        set aa2 $i
        set leng [veclength [vecsub $d1 $d2]]
    }
    $t delete; $p delete; $d delete
}

set exist [info exists aa2]
if {[expr {$exist == 0}]} {
    set data "$aa1\n"
    set filename "anchors.txt"
    set fileId [open $filename "w"]; puts -nonewline $fileId $data; close $fileId
    wlog $_dbgfp "FAIL: Ligand second anchor not found. Wrote anchors.txt with L1 only."
    close $_dbgfp
    puts "Ligand second anchor not found"
    exit
}

puts "anchor 2 is" 
puts $aa2
puts $angl
puts $leng
puts ""
wlog $_dbgfp "Chosen L2: $aa2 (angle=$angl deg, len=$leng Å)"

# --- L3 search: angle ~90 with (aa1, aa2) and length window -----------------
set amat {}
set c3 [open "candidates_L3.txt" "w"]
puts $c3 "#name angle_deg length(aa2->i)  passed_length_window(DMIN..DMAX)"
foreach i $tot {
    set alis {}; set angle1 {}; set angle2 {}; set angle3 {}; set angle {}
    set t [atomselect 1 "resname MMM and name $i"]
    set p [atomselect 1 "resname MMM and name $aa2"]
    set d [atomselect 1 "resname MMM and name $aa1"]
    if {$i ne $aa1 && $i ne $aa2} {
        set a1 [$d get index]
        set d1 [measure center $t weight mass]
        set d2 [measure center $p weight mass]
        set leng [veclength [vecsub $d1 $d2]]
        lappend angle1 $a1; lappend angle1 "1"; lappend angle $angle1
        lappend angle2 [$p get index]; lappend angle2 "1"; lappend angle $angle2
        lappend angle3 [$t get index]; lappend angle3 "1"; lappend angle $angle3
        set ang [measure angle $angle]
        puts $i
        puts $ang
        puts $leng
        set passLen [expr {($leng > $dmin) && ($leng < $dmax)}]
        puts $c3 "$i $ang $leng $passLen"
        if {$passLen} { lappend amat $i }
    }
    $t delete; $p delete; $d delete
}
close $c3
wlog $_dbgfp "L3 length-window candidates: [llength $amat] (see candidates_L3.txt)"

set adf 90
foreach i $amat {
    set angle1 {}; set angle2 {}; set angle3 {}; set angle {}
    set t [atomselect 1 "resname MMM and name $i"]
    set p [atomselect 1 "resname MMM and name $aa2"]
    set d [atomselect 1 "resname MMM and name $aa1"]
    set d1 [measure center $t weight mass]
    set d2 [measure center $p weight mass]
    lappend angle1 [$d get index]; lappend angle1 "1"; lappend angle $angle1
    lappend angle2 [$p get index]; lappend angle2 "1"; lappend angle $angle2
    lappend angle3 [$t get index]; lappend angle3 "1"; lappend angle $angle3
    set ang [measure angle $angle]
    if {[expr {abs($ang - 90.0)}] < $adf} {
        set adf [expr {abs($ang - 90.0)}]
        set angf $ang
        set aa3 $i
        set leng [veclength [vecsub $d1 $d2]]
    }
    $t delete; $p delete; $d delete
}

set exist [info exists aa3]
if {[expr {$exist == 0}]} {
    set data "$aa1 $aa2\n"
    set filename "anchors.txt"
    set fileId [open $filename "w"]; puts -nonewline $fileId $data; close $fileId
    wlog $_dbgfp "FAIL: Ligand third anchor not found. Wrote anchors.txt with L1 L2."
    close $_dbgfp
    puts "Ligand third anchor not found"
    exit
}

puts "anchor 3 is"
puts $aa3
puts $angf
puts $leng
wlog $_dbgfp "Chosen L3: $aa3 (angle=$angf deg, len=$leng Å)"

puts "The three anchors are"

set data "$aa1 $aa2 $aa3\n"
set filename "anchors.txt"
set fileId [open $filename "w"]; puts -nonewline $fileId $data; close $fileId
wlog $_dbgfp "WROTE anchors.txt: $aa1 $aa2 $aa3"

# --- keep original tail (dummy alignment) ------------------------------------
mol load pdb dum.pdb

set a [atomselect 1 "(not resname MMM) and (resid FIRST to LAST and name CA C N O)"]
set b [atomselect 1 "resname MMM and noh"]
set c [atomselect 2 all]
$c moveby [vecsub [measure center $a weight mass] [measure center $c weight mass]]
$c writepdb dum1.pdb
if {[expr {$sdr_dist != 0}]} {
    set dlis [list 0 0 [expr {$sdr_dist}]]
    $b moveby $dlis
    $c moveby [vecsub [measure center $b weight mass] [measure center $c weight mass]]
    $c set resid 2
    $c writepdb dum2.pdb
    set dlis2 [list 0 0 [expr {-1*$sdr_dist}]]
    $b moveby $dlis2
}

close $_dbgfp
exit
# ============================================================================ 
