import os
import pathlib
import re
import time
import multiprocessing as mp
import random

import numpy as np

import matplotlib.pyplot as plt


def sym_op(symm_op_line, symm_atom_info):

    import numpy as np
    import re
    import pandas as pd

    # Here I use re.split to split the string first(seperated by several signs)
    # Then filter out empty string
    # Then, reversely assign each operations
    symm_op_x = list(filter(None, (re.split(r"[\s,']", symm_op_line))))[-3].replace("X", "x").replace("Y", "y").replace("Z", "z")
    symm_op_y = list(filter(None, (re.split(r"[\s,']", symm_op_line))))[-2].replace("X", "x").replace("Y", "y").replace("Z", "z")
    symm_op_z = list(filter(None, (re.split(r"[\s,']", symm_op_line))))[-1].replace("X", "x").replace("Y", "y").replace("Z", "z")

    # Atom positions after apply symmetry operation
    # Here we reduce the shape of matrix from n*6 to n*5, dumped the idx column
    atom_info_symm = np.zeros((symm_atom_info.shape[0], symm_atom_info.shape[1]))
    # We know x, y, z fraction should be copied from previous matrix's column 3, 4, 5
    x = symm_atom_info[:, 2]
    y = symm_atom_info[:, 3]
    z = symm_atom_info[:, 4]
    # Here we use pd.eval to change string to executable expression
    # Inwhich 3 expressions, varaibles are x, y, z defined above
    x_new = pd.eval(symm_op_x)
    y_new = pd.eval(symm_op_y)
    z_new = pd.eval(symm_op_z)

    # Build the matrix that stores atom information after applying symmetry op
    # This column is atom number, remain unchanged
    atom_info_symm[:, 0] = symm_atom_info[:, 0]
    atom_info_symm[:, 1] = symm_atom_info[:, 1]
    # These 3 columns are changed, they are coordinates
    atom_info_symm[:, 2] = x_new
    atom_info_symm[:, 3] = y_new
    atom_info_symm[:, 4] = z_new
    # This column is atom occupancy, remain unchanged
    atom_info_symm[:, 5] = symm_atom_info[:, 5]

    # Return the newly built matrix, ready to be appended
    return atom_info_symm


def rmv_brkt(string):
    if string == '.':
        return 0
    return float(string.replace("(", "").replace(")", "").replace("-.", "-0.").replace('?', '0').replace("..", "."))


def gaus(x, h):
    const_g = 4 * np.log(2)
    value = ((const_g**(1/2)) / (np.pi**(1/2) * h)) * np.exp(-const_g * (x/h)**2)
    return value


def y_multi(x_val, step, xy_merge, H):
    y_val = 0
    for xy_idx in range(0, xy_merge.shape[0]):
        angle = xy_merge[xy_idx, 0]
        inten = xy_merge[xy_idx, 1]
        if angle > (x_val * step - 5) and angle < (x_val * step + 5):
            y_val = y_val + inten * (gaus((x_val * step - angle), H[xy_idx, 0])*1.5)
    return y_val


def process_cif(cif_dir, cif_file, out_dir, hkl_info, x_min=0, x_max=90, x_step=0.01, cal_mode='draw_xrd'):
    """
    # Aims to load CIF files and extract the necessary information.
    # For XRD we need:
    # 1. CELL: cell information
    # 2. SYMM: symmetry operation to expand the whole cell
    # 3. ATOM: atom fractional position, occupancy
    """

    # Set print options for digits.
    # Set 3 decimals.
    # np.set_printoptions(precision=3)
    # Suppress scientific notations.
    # np.set_printoptions(suppress=True)

    # FILE & DICT.
    # File management & resource dictionary management.

    # Locate .cif files. This works for both absolute dir and relative dir.
    cif_file_dir = "{}/{}".format(cif_dir, cif_file)

    # This is an error log for files that are too large.
    # If the file is too large, this implies:
    # 1) There are too many atoms in this unit cell, which reduce its obvious of symmetry.
    # 2) The format of .cif has include terms that are of no use.
    # File size of 50kb allows to ignore heavy chaos. And reduce time cost.
    fileSize = round(os.path.getsize(cif_file_dir) / 1000, 1)
    if False and fileSize > 50:
        return "Failed: File size too large({}).".format(fileSize)

    # Read file to program memory and close file. Here we need encoding with UTF-8(due to the European language).
    with open(cif_file_dir, "r", encoding="UTF-8") as cif_content:
        cif_content_lines = cif_content.readlines()

    # # Check existing XRD output files in output folder to avoid repeated calculations.
    if True and cal_mode == "draw_xrd":
        for path, dirs, files in os.walk(out_dir):
            for file in files:
                if file == "{}.txt".format(re.split(r"[.]", cif_file)[0]):
                    return "Cancelled: Repeated XRD calculation of .cif file({}).".format(cif_file)

    # Open tables for ion scattering data.
    with open(pathlib.Path(__file__).parent.resolve() / "ION_SCATTERING_TABLE.txt", "r") as scat_table:
        scat_table_lines = scat_table.readlines()
    # And create a dictionary using this file to log ions with its idx.
    chem_dict = {}
    for line in scat_table_lines:
        chem_dict[line.split()[0]] = line.split()[1]

    # Create a dictionary mapping space group to crystal structure.
    # This will be used for the headers of XRD files.
    space_group_map_dict = {}
    for i in range(1, 3):
        space_group_map_dict[i] = 1
    for i in range(3, 16):
        space_group_map_dict[i] = 2
    for i in range(16, 75):
        space_group_map_dict[i] = 3
    for i in range(75, 143):
        space_group_map_dict[i] = 4
    for i in range(143, 168):
        space_group_map_dict[i] = 5
    for i in range(168, 195):
        space_group_map_dict[i] = 6
    for i in range(195, 231):
        space_group_map_dict[i] = 7

    # DEFINE VAR
    # Define variables needed during extraction

    # 1. CELL & INFO.
    # There are 6 parameters to describe a cell. Its 3 lengths and 3 angles.
    cell_a = 0.0
    cell_b = 0.0
    cell_c = 0.0
    cell_alpha = 0.0
    cell_beta = 0.0
    cell_gamma = 0.0
    # There are various chemical formula for this crystal.
    chem_form = ""
    # This is the "_space_group_IT_number" in CIF file.
    space_group = 0
    # Once we have locate "_space_group_IT_number", this become Ture.
    space_group_line_judge = False

    # 2. ATOM: atomic data
    # This is the line index of the start of atoms lists of a CIF file.
    atom_start_line = 0
    # This is the line index of the end of atoms lists of a CIF file.
    atom_end_line = 0
    # This is the line index of the "loop_", with ATOM info followed.
    atom_loop_line = 0
    # Once we locate "_atom_label_line", this becomes False.
    atom_loop_line_judge = True
    # This is the line index of the "_atom_label_line".
    atom_label_line = 0
    # This is the line index of the "_atom_site_type_symbol".
    ion_label_line = 0
    # Once we locate "_atom_site_type_symbol", this becomes True.
    # False indicate the ion label is missing and can be followed by neutral atom scattering factor.
    # True indicate that ion label is identified and can be followed by ion scattering factor.
    ion_label_line_judge = False
    # This is the line index of the term "_atom_site_fract_x".
    atom_x_line = 0
    # This is the line index of the term "_atom_site_fract_y".
    atom_y_line = 0
    # This is the line index of the term "_atom_site_fract_z".
    atom_z_line = 0
    # This is the line index of the term "_atom_site_occupancy".
    atom_ocp_line = 0
    # Once we locate "_atom_site_occupancy", this becomes True.
    atom_ocp_line_judge = False
    # Once we locate "_atom_label_line", this becomes True.
    atom_start_line_judge = False
    # Once we locate "atom_start_line", this becomes True.
    atom_end_line_judge = False
    # Once we locate "atom_start_line", this becomes True. If we find normal end, this becomes False.
    atom_end_blank_judge = False

    # 3. SYMM: symmetry operations
    # This is the line index of the start of symm. op. list of a CIF file.
    symm_start_line = 0
    # This is the line index of the start of symm. op. of a CIF file.
    symm_end_line = 0
    # Once we locate "_space_group_symop_operation_xyz" or "_symmetry_equiv_pos_as_xyz", this becomes True.
    symm_end_line_judge = False

    # For some cod file, there are comments inside ; ;
    # Which means, if line == ';', this means the start of comments, comment_judge = True
    # if again line == ';', this means the end of comments, comment_judge = False
    # Initially, comment judge = False
    comment_judge = False

    # ASSIGN VAR
    # Read CIF for desired variables.

    line_count = 0
    for line in cif_content_lines:
        line_count += 1

        # 1. CELL

        if ';' in line.strip():
            if ";" == line.strip()[0]:
                if not comment_judge:
                    comment_judge = True
                else:
                    comment_judge = False
        elif "_database_code_ICSD" in line and not comment_judge:
            icsd_coll_code = line.split()[1]
        elif '_space_group_IT_number' in line and not comment_judge:
            if len(line.split()) == 2:
                space_group = rmv_brkt(line.split()[1])
                space_group_line_judge = True
            else:
                return 'Failed:_space_group_IT_number wrong format'
        elif '_cell_length_a' in line and not comment_judge:
            if len(line.split()) == 2:
                cell_a = rmv_brkt(line.split()[1])
            else:
                return 'Failed: _cell_length_1 wrong format'
        elif '_cell_length_b' in line and not comment_judge:
            if len(line.split()) == 2:
                cell_b = rmv_brkt(line.split()[1])
            else:
                return 'Failed: _cell_length_b wrong format'
        elif '_cell_length_c' in line and not comment_judge:
            if len(line.split()) == 2:
                cell_c = rmv_brkt(line.split()[1])
            else:
                return 'Failed: _cell_length_c wrong format'
        elif '_cell_angle_alpha' in line and not comment_judge:
            if len(line.split()) == 2:
                cell_alpha = rmv_brkt(line.split()[1])
                if cell_alpha > 180:
                    return 'Failed: _cell_length_alpha wrong value {}'.format(cell_alpha)
                cell_alpha = rmv_brkt(line.split()[1]) * np.pi / 180.
            else:
                return 'Failed: _cell_length_alpha wrong format'
        elif '_cell_angle_beta' in line and not comment_judge:
            if len(line.split()) == 2:
                cell_beta = rmv_brkt(line.split()[1])
                if cell_beta > 180:
                    return 'Failed: _cell_length_beta wrong value {}'.format(cell_beta)
                cell_beta = rmv_brkt(line.split()[1]) * np.pi / 180.
            else:
                return 'Failed: _cell_length_beta wrong format'
        elif '_cell_angle_gamma' in line and not comment_judge:
            if len(line.split()) == 2:
                cell_gamma = rmv_brkt(line.split()[1])
                if cell_gamma > 180:
                    return 'Failed: _cell_length_gamma wrong value {}'.format(cell_gamma)
                cell_gamma = rmv_brkt(line.split()[1]) * np.pi / 180.
            else:
                return 'Failed: _cell_length_gamma wrong format'
        elif '_chemical_formula_sum' in line and not comment_judge:
            chem_form = re.split(r"['\sB]", line, 1)[1].replace("'", "").strip()

        # 2. ATOM

        elif 'loop_' in line and atom_loop_line_judge and not comment_judge:
            atom_loop_line = line_count
            # Find the label that all CIF has: "_atom_site_label"
        elif '_atom_site_label' == line.strip() and not comment_judge:
            atom_label_line = line_count
            atom_start_line_judge = True
            atom_loop_line_judge = False
        # Find the label that indicate ionization: '_atom_site_type_symbol'
        elif '_atom_site_type_symbol' == line.strip() and not comment_judge:
            ion_label_line = line_count
            ion_label_line_judge = True
        # Find xyz fractions: "_atom_site_fract_x" & "...y" & "...z"
        elif '_atom_site_fract_x' == line.strip() and not comment_judge:
            atom_x_line = line_count
        elif '_atom_site_fract_y' == line.strip() and not comment_judge:
            atom_y_line = line_count
        elif '_atom_site_fract_z' == line.strip() and not comment_judge:
            atom_z_line = line_count
        # Find "_atom_site_occupancy"
        elif '_atom_site_occupancy' == line.strip() and not comment_judge:
            atom_ocp_line = line_count
            atom_ocp_line_judge = True
        # Find the line doesn't start with "_atom" and this line is the start of atoms' list
        elif ('_atom' not in line) and atom_start_line_judge and not comment_judge:
            atom_start_line = line_count
            atom_start_line_judge = False
            atom_end_line_judge = True
            atom_end_blank_judge = True
        # Find the line start with either "loop_" or "#End" or blank. The previous line is the end of atoms
        elif atom_end_line_judge and (('loop_' in line) or ('#End' in line)) and not comment_judge:
            atom_end_line = line_count - 1
            atom_start_line_judge = False
            atom_end_line_judge = False
            atom_end_blank_judge = False
        # Sometimes the CIF file end with nothing
        if atom_end_blank_judge and not comment_judge:
            atom_end_line = line_count

        # print(atom_start_line, atom_end_line)

        # 3. SYMM

        # Find "_space_group_symop_operation_xyz" or "_symmetry_equiv_pos_as_xyz"
        if '_space_group_symop_operation_xyz' in line or '_symmetry_equiv_pos_as_xyz' in line and not comment_judge:
            symm_start_line = line_count + 1
            symm_end_line_judge = True
        # Find "loop_" after symm. op.
        if 'loop_' in line.strip() and symm_end_line_judge and not comment_judge:
            symm_end_line = line_count - 1
            symm_end_line_judge = False

        # Print varaibles out
    # print("Done: CIF {}".format(cif_file))
    startTime = time.time()

    # CAL VAR
    # Calculate variables.

    # 1. ATOM
    # If no space group information, we return Failed
    if not space_group_line_judge:
        return "Failed: Space Group(no space group id)"
    if space_group < 1 or space_group > 230:
        return "Failed: Space group(not between 1-230)"
    # Mapping space group and crystal structure
    crystal_sys = space_group_map_dict.get(space_group)
    # If either "atom_start_line" or "atom_end_line" is 0, we return False
    if atom_start_line == 0:
        return "Failed: Atom list(no start find)"
    if atom_end_line == 0:
        return "Failed: Atom list(no end find)"
    # "num_sym_atom": number of non-symmetry atoms
    num_symm_atom = 0
    num_symm_atom = atom_end_line - atom_start_line + 1
    # this is the total variable column that the atom list should have
    atom_site_count = atom_start_line - atom_loop_line - 1
    # The column number 6 is defined as "type, idx, x, y, z, ocp". May subject to change
    symm_atom_info = np.zeros((num_symm_atom, 6))
    # "count_symm_atom" is the idx for future matrix "symm_atom_info"
    for count_symm_atom, line in enumerate(cif_content_lines[atom_start_line - 1: atom_end_line]):
        line = line.replace("'", "").replace("..", ".").replace(")(", "").replace("))", ")").replace(")", " ")
        if len(line.split()) < atom_site_count:
            return 'Failed: Atom list column doesnot match loop labels'
        # If CIF format is "Al1" which is element + label count
        if re.match('([A-z]+)([0-9]+)', line.split()[atom_label_line - atom_loop_line - 1]) != None:
            # "symm_atom_type": CIF format label of chemical name
            symm_atom_type = re.match('([A-z]+)([0-9]+)', line.split()[atom_label_line - atom_loop_line - 1]).group(1)
            # "sym_atom_idx": CIF format indices for non-symmetry atoms, not the final sequence of atoms
            symm_atom_idx = re.match('([A-z]+)([0-9]+)', line.split()[atom_label_line - atom_loop_line - 1]).group(2)
        # If CIF format is "Al" without index
        elif re.match('([A-z]+)([0-9]+[+-]*)', line.split()[atom_label_line - atom_loop_line - 1]) == None:
            symm_atom_type = line.split()[atom_label_line - atom_loop_line - 1]
            symm_atom_idx = "1"
        # Sometimes, CIF missing ATOM info, like this "? ? ? ?"
        if len(line.split()) < (atom_z_line - atom_loop_line):
            return "Failed: Atom list(list not complete)"
        # For this line, extract x, y, z information
        symm_atom_x = line.split()[atom_x_line - atom_loop_line - 1]
        symm_atom_y = line.split()[atom_y_line - atom_loop_line - 1]
        symm_atom_z = line.split()[atom_z_line - atom_loop_line - 1]
        # For this line, extract occupancy information
        if atom_ocp_line_judge:
            symm_atom_ocp = line.split()[atom_ocp_line - atom_loop_line - 1]
        # If no occupancy in CIF, default to 1
        elif not atom_ocp_line_judge:
            symm_atom_ocp = "1"
        # Here generate matrix "symm_atom_info"
        # Here we search for tables for its corresbonding Z
        if chem_dict.get(symm_atom_type) != None:
            symm_atom_info[count_symm_atom, 0] = int(chem_dict.get(symm_atom_type))
            symm_atom_info[count_symm_atom, 1] = rmv_brkt(symm_atom_idx)
            symm_atom_info[count_symm_atom, 2] = rmv_brkt(symm_atom_x)
            symm_atom_info[count_symm_atom, 3] = rmv_brkt(symm_atom_y)
            symm_atom_info[count_symm_atom, 4] = rmv_brkt(symm_atom_z)
            symm_atom_info[count_symm_atom, 5] = rmv_brkt(symm_atom_ocp)
        # This error case happens when atom label is like "ALM"
        elif chem_dict.get(symm_atom_type) == None:
            return "Failed: Can not match neutrual atom type"
        # Then we update [:, 0] if ion label exists and REPLACE NEUTRAL atom index with ION index
        if True and ion_label_line_judge and re.fullmatch('([A-z]+)([1-9]+[+-]+)', line.split()[ion_label_line - atom_loop_line - 1]) != None and chem_dict.get(line.split()[ion_label_line - atom_loop_line - 1].strip()) != None:
            symm_atom_info[count_symm_atom, 0] = int(chem_dict.get(line.split()[ion_label_line - atom_loop_line - 1].strip()))
        # Here we define a estimated cell total number of atom. To filter out cases with too many to save time.

    # print("symm_atom_info")
    # print(symm_atom_info)

    # print("Done: ATOM {} in {:.2f}s".format(symm_atom_info.shape[0], time.time()-startTime))
    startTime = time.time()

    # 2. SYMM
    if cal_mode == "draw_xrd" or cal_mode == "draw_cell":

        # Count how many symmetry operations
        num_symm_op = 0
        num_symm_op = symm_end_line - symm_start_line + 1

        num_symm_atom_count = 400
        if False and num_symm_atom*num_symm_op > num_symm_atom_count:
            return "Failed: Estimated atom number exceed limit {} (Estimated total: {})".format(num_symm_atom_count, num_symm_atom*num_symm_op)


        # Summarize information
        lattice_atom_info = np.zeros((0, 6))
        # First consider no symmetry information
        if symm_start_line == 0 or symm_end_line == 0:
            lattice_atom_info = np.vstack([lattice_atom_info, symm_atom_info])
        # Apply symm. op. on every line of symmetry operations
        for line in cif_content_lines[symm_start_line - 1: symm_end_line]:
            if len(list(filter(None, (re.split(r"[\s,']", line))))) < 3:
                return "Failed: Symmetry operations <3 (x, y, z)"
            if re.match('([0-9]+)([\s]+)', line) != None and len(list(filter(None, (re.split(r"[\s,']", line))))) >= 5:
                return "Failed: Symmetry operations >= 5 with idx (x, y, z)"
            if re.match('([0-9]+)([\s]+)', line) == None and len(list(filter(None, (re.split(r"[\s,']", line))))) >= 4:
                return "Failed: Symmetry operations >= 4 without idx (x, y, z)"
            lattice_atom_info = np.vstack([lattice_atom_info, sym_op(line, symm_atom_info)])

        # After complete "lattice_atom_info", we need to filter out identical expanded atoms
        # This process is a matrix process that compares all rows of the matrix and erase the rows identical to previous ones
        lattice_atom_info_redu = np.unique(lattice_atom_info, axis=0)
        # print("lattice_atom_info_redu")
        # print(lattice_atom_info_redu)

    elif cal_mode == "cif_check":
        print("2/5 Skip: SYMM")

    # 3. ATOM + SYMM
    if cal_mode == "draw_xrd" or cal_mode == "draw_cell":

        # 1. We translate atoms outside unit cell back to unit cell by +1/-1.
        # 2. We reduce identicial atoms, if translated atoms are at the same position as its pairs.
        # 3. Special cases are <=0.01 and >=0.99. Those atoms should be considered as 0 and 1, which means they are on the cell surface.
        # Atoms of special cases should be duplicated.
        # 4. Reduce identical atoms.

        # 1.
        translatedCell = np.zeros((0, 6))
        translatedAtom = np.zeros((1, 6))
        for i in range (0, lattice_atom_info_redu.shape[0]):
            for j in range (2, 5):
                if lattice_atom_info_redu[i, j] < 0:
                    translatedAtom[0, j] = lattice_atom_info_redu[i, j] + 1
                elif lattice_atom_info_redu[i, j] > 1:
                    translatedAtom[0, j] = lattice_atom_info_redu[i, j] - 1
                elif lattice_atom_info_redu[i, j] == 0 or lattice_atom_info_redu[i, j] == 1:
                    translatedAtom[0, j] = lattice_atom_info_redu[i, j]
                else:
                    translatedAtom[0, j] = lattice_atom_info_redu[i, j]
            translatedAtom[0, [0, 1, 5]] = lattice_atom_info_redu[i, [0, 1, 5]]
            translatedCell = np.vstack([translatedCell, translatedAtom])

        # print("translatedCell")
        # print(translatedCell.shape[0])
        # print(translatedCell)

        # 2.
        reducedCell = np.unique(translatedCell, axis=0)

        # print("reducedCell")
        # print(reducedCell.shape[0])
        # print(reducedCell)

        # 3.
        roundedCell = np.zeros((0, 6))
        for i in range (0, reducedCell.shape[0]):
            zeroCount = 0
            for j in range (2, 5):
                if reducedCell[i, j] <= 0.02 or reducedCell[i, j] >= 0.98:
                    zeroCount += 1
            if zeroCount == 0:
                roundedCell = np.vstack([roundedCell, reducedCell[i, :]])
            elif zeroCount == 1:
                for j in range (2, 5):
                    if reducedCell[i, j] <= 0.02 or reducedCell[i, j] >= 0.98:
                        roundedAtom = np.array([reducedCell[i, :]]*2)
                        roundedAtom[:, j] = np.array([0, 1]).T
                        roundedCell = np.vstack([roundedCell, roundedAtom])
            elif zeroCount == 2:
                j1 = 0
                j2 = 0
                for j in range (2, 5):
                    if reducedCell[i, j] <= 0.05 or reducedCell[i, j] >= 0.95:
                        if j1 == 0:
                            j1 = j
                        else:
                            j2 = j
                roundedAtom = np.array([reducedCell[i, :]]*4)
                roundedAtom[:, [j1, j2]] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                roundedCell = np.vstack([roundedCell, roundedAtom])
            elif zeroCount == 3:
                roundedAtom = np.array([reducedCell[i, :]]*8)
                roundedAtom[:, [2, 3, 4]] = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
                roundedCell = np.vstack([roundedCell, roundedAtom])

        # print("roundedCell")
        # print(roundedCell.shape[0])
        # print(roundedCell)

        #4.
        reducedCell = np.unique(roundedCell, axis=0)

        # print("reducedCell")
        # print(reducedCell.shape[0])
        # print(reducedCell)

        cell_info = np.around(reducedCell, decimals=2)
        # print("cell_info")
        # print(cell_info.shape)
        # print(cell_info[:10])

        # print("Done: SYMM {} in {:.2f}s".format(cell_info.shape[0], time.time()-startTime))
        startTime = time.time()

        if cal_mode == "draw_cell":
            # print("return cell_info")
            return cell_info

        if False and cell_info.shape[0] > 200:
            return "Failed: Confirmed atom number exceed limit 200 (Total: {})".format(cell_info.shape[0])

        if False:
            i = 0
            for i in range (0, cell_info.shape[0]):
                if True and cell_info[i, 0] == 7 and cell_info[i, 1] == 1:
                    print(cell_info[i, :])

    elif cal_mode == "check_cif":
        print("3/5 Skip: ATOM + SYMM")

    # CAL PEAK
    # Calculate peak positions and peak intensities.

    if cal_mode == "draw_xrd":

        time_start = time.time()

        # Since we have hkl_info inputed, we use general equations to calculate distance between hkl planes.
        hkl_h = hkl_info[:, 0]
        hkl_k = hkl_info[:, 1]
        hkl_l = hkl_info[:, 2]
        # The general equation is described via Triclinic. Although other would be simplier, but we don't necessarily need to classify.
        a = cell_a
        b = cell_b
        c = cell_c
        alpha = cell_alpha
        beta  = cell_beta
        gamma = cell_gamma
        v = 0
        v = (a*b*c) * ( 1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 )**(1/2)
        hkl_d = np.zeros((hkl_info.shape[0], 1))
        hkl_d = (1/v) * (hkl_h**2*b**2*c**2*np.sin(alpha)**2 + hkl_k**2*a**2*c**2*np.sin(beta)**2 + hkl_l**2*a**2*b**2*np.sin(gamma)**2 + 2*hkl_h*hkl_k*a*b*c**2*(np.cos(alpha)*np.cos(beta)-np.cos(gamma)) + 2*hkl_k*hkl_l*a**2*b*c*(np.cos(beta)*np.cos(gamma)-np.cos(alpha)) + 2*hkl_h*hkl_l*a*b**2*c*(np.cos(alpha)*np.cos(gamma)-np.cos(beta)))**(1/2)
        hkl_d = 1/hkl_d

        # print(a, b, c, alpha, beta, gamma)
        # print(v, hkl_d)

        # Then calculate two_theta.
        wavelength   = 1.5418
        two_theta    = np.zeros((hkl_info.shape[0], 1))
        two_theta_pi = np.zeros((hkl_info.shape[0], 1))
        for i in range (0, hkl_info.shape[0]):
            if wavelength / 2 / hkl_d[i] < 1:
                theta_cal = np.arcsin(wavelength / 2 / hkl_d[i])
                # Here we have the options to add 2theta_errors
                theta_err = 0
                theta_obs = theta_cal + theta_err
                two_theta[i] = 2*theta_obs/np.pi*180
                two_theta_pi[i] = 2*theta_obs
        two_theta = np.around(two_theta, decimals=2)

        # Here before hkl matrix is passed on, we delete those hkls, with 0 as 2 theta.
        # hkl_2theta is a n*5 array.
        hkl_2theta = np.hstack([hkl_info, two_theta])
        hkl_2theta = np.hstack([hkl_2theta, two_theta_pi])
        # print("hkl_2theta")
        # print(hkl_2theta.shape)
        # print(hkl_2theta[-10:])

        # Then we delete its row if 2theta is 0.
        hkl_2theta = hkl_2theta[np.all(hkl_2theta[:, 4:5] != 0, axis=1)]
        # print("hkl_2theta")
        # print(hkl_2theta.shape)
        # print(hkl_2theta[-10:])

        hkl_info = np.zeros((hkl_2theta.shape[0], 4))
        two_theta = np.zeros((hkl_2theta.shape[0], 1))
        two_theta_pi = np.zeros((hkl_2theta.shape[0], 1))
        hkl_info[:, 0:4] = hkl_2theta[:, 0:4]
        two_theta[:, 0] = hkl_2theta[:, 4]
        two_theta_pi[:, 0] = hkl_2theta[:, 5]

        # After two_theta, we calculate lorentz-polarization factor.
        lp = np.zeros((hkl_2theta.shape[0], 1))
        for i in range (0, hkl_2theta.shape[0]):
            lp[i] = (1 + np.cos(two_theta_pi[i])**2) / (np.cos(two_theta_pi[i]/2)*np.sin(two_theta_pi[i]/2)**2)
        # print("lp")
        # print(lp.shape)
        # print(lp[:10])

        # STRUCTURE FACTOR

        # Next, vector product of h * x_j.
        hkl_pos = np.matmul(hkl_info[:, [0, 1, 2]], cell_info[:, [2, 3, 4]].T)
        # print("hkl_pos")
        # print(hkl_pos.shape)
        # print(hkl_pos[:10])

        # Next, population factor.
        # This depend on the position of the atoms, if one 0/1 -> 1/2, two 0/1 -> 1/4, three 0/1 -> 1/8.
        # This population should be determined by symetry, so it is WRONG for now.
        pos_pop = np.zeros((cell_info.shape[0], 1))
        i = 0
        for i in range (0, cell_info.shape[0]):
            j = 1
            count = 0
            for j in range (2, 5):
                if cell_info[i, j] == 1 or cell_info[i, j] == 0:
                    count += 1
            if count == 0:
                pos_pop[i, 0] = 1
            elif count == 1:
                pos_pop[i, 0] = 1/2
            elif count == 2:
                pos_pop[i, 0] = 1/4
            elif count == 3:
                pos_pop[i, 0] = 1/8
        # print("pos_pop")
        # print(pos_pop.shape)
        # print(pos_pop[:10])

        # Next, temperature factor, we use the simplest equation. b ranges from 0.5-1 or 1-3.
        temp_factor_b = 0
        s = np.zeros((two_theta_pi.shape[0], 1))
        s = np.sin(two_theta_pi/2) / wavelength
        # print("s")
        # print(s.shape)
        # print(s[:10])
        temp_factor = np.exp(-temp_factor_b * s**2)
        # print("temp_factor")
        # print(temp_factor.shape)
        # print(temp_factor[:10])

        ### ATOM SCATTERING FACTOR
        # Next, atmoic scattering factor.
        # For neutral atom: International Tables for Crystallography (2006). Vol. C. ch. 4.3, p. 262
        # https://it.iucr.org/Cb/ch4o3v0001/sec4o3o2/
        # For ion: International Tables for Crystallography (2006). Vol. C. ch. 6.1, pp. 554-590
        # https://it.iucr.org/Cb/ch6o1v0001/sec6o1o1/
        atom_scat = np.zeros((hkl_info.shape[0], cell_info.shape[0]))
        # i is the sequense of atoms
        i = 0
        for i in range (0, cell_info.shape[0]):
            col = np.zeros((hkl_info.shape[0], 1))
            abc_ion = np.zeros((9, 1))
            abc_ion[0:9, 0] = np.array(scat_table_lines[int(cell_info[i, 0]) - 1].split()[3:12])
            # j is the appraoch for integration
            j = 0
            for j in range (0, 7, 2):
                col[:, 0] = col[:, 0] + abc_ion[j, 0] * np.exp(- abc_ion[j+1, 0] * s[:, 0]**2)
            col[:, 0] = col[:, 0] + abc_ion[8]
            # Then multiply by occupancy
            col[:, 0] = col[:, 0] * cell_info[i, 5]
            # here replace the correct result in to matrix 'atom_scat'
            atom_scat[:, i] = col[:, 0]
        # print("atom_scat")
        # print(atom_scat.shape)
        # print(atom_scat[:10])

        # Final equation. This is the loop of n atoms, calculating f_hkl integration. f_hkl is a 2x1 matrix.
        time_start = time.time()
        expArray = np.exp(2 * np.pi * 1j * hkl_pos)
        struc = np.zeros((hkl_info.shape[0], 1))
        struc[:, 0] = np.square(np.absolute(np.sum(np.matmul(np.multiply(atom_scat, expArray), pos_pop), axis=1))).T
        # print('struc mp time', format(time.time() - time_start))
        # print(struc.shape)
        # print(struc[:10])

        # INTENSITIES
        # Intensities take structural factor, polarization factor, angular-velocity and etc together
        inte = np.zeros((hkl_info.shape[0], 1))
        inte[:, 0] = np.multiply(lp[:, 0], struc[:, 0])
        x_y = np.zeros((hkl_info.shape[0], 2))
        x_y[:, 0] = two_theta[:, 0]
        x_y[:, 1] = inte[:, 0]
        if True:
            xy_hkl = np.hstack([np.around(x_y, decimals=2), hkl_info])
            # print("xy_hkl")
            # print(xy_hkl.shape)
            # print(xy_hkl[:10])

        # After having all twotheta and its intensities, we realize that some intensities are zero, we first filter them out.
        xy_redu = x_y[np.all(x_y!=0, axis=1)]
        # print("xy_redu")
        # print(xy_redu.shape)
        # print(xy_redu[:10])

        # After having a pure xy data, we merge their intensites if they are in same peak position.
        xy_merge = np.zeros((1, 2))
        xy_merge[0, :] = xy_redu[0, :]
        i = 1
        for i in range (1, xy_redu.shape[0]):
            merge_judge = True
            j = 0
            for j in range (0, xy_merge.shape[0]):
                if xy_redu[i, 0] == xy_merge[j, 0]:
                    xy_merge[j, 1] = xy_merge[j, 1] + xy_redu[i, 1]
                    merge_judge = False
                    break
            if merge_judge:
                xy_merge = np.vstack((xy_merge, xy_redu[i, :]))
        # print("xy_merge")
        # print(xy_merge.shape)
        # print(xy_merge[:10])

        # Additional log for exact peak information.
        if False:
            new_filename = "xy_hkl_info_{}.txt".format(re.split(r"[.]", cif_file)[0])
            new_file = open("{}/{}".format(out_dir, new_filename), "w+")
            #np.savetxt(new_file, pattern, delimiter=',')
            new_file.write("\n".join(str(item).replace("[", "").replace("]", "") for item in xy_hkl.tolist()))
            new_file.close()

        # Additional plot bar to verify result.
        if False:
            plt.figure(figsize=(15,4))
            plt.xlim([x_min, x_max])
            plt.bar(xy_hkl[:,0], xy_hkl[:,1], width=0.2, bottom=None, align='center')
            plt.show()
            # This comparison shows an small overlap issue with bar plot.
            plt.figure(figsize=(15,4))
            plt.xlim([x_min, x_max])
            plt.bar(xy_merge[:,0], xy_merge[:,1], width=0.2, bottom=None, align='center')
            plt.show()
        # print("Done: xy merged {} in {:.2f}s".format(xy_merge.shape[0], time.time()-startTime))

    # Peak Shape Functions
    # Defined in "func_peak_shape.py"

    # Set up x-axis and resolutions
    if cal_mode == "draw_xrd":
        # Augment different peak shape functions, determined by U, V, W - Note: can be moved to training-time
        for peak_shape, (U, V, W) in enumerate([(0.05, -0.06, 0.07), (0.05, -0.01, 0.01),
                                                (0.0, 0.0, 0.01), (0.0, 0.0, random.uniform(0.001, 0.1))]):
            # U = 0.05
            # V = -0.01
            # W = 0.01
            H = np.zeros((xy_merge.shape[0], 1))
            H[:, 0] = (U * (np.tan(xy_merge[:, 0]*(np.pi/180)/2))**2 + V * np.tan(xy_merge[:, 0] * (np.pi/180)/2) + W)**(1/2)

            step = x_step
            total_points = int(180/step)

            #------------------------------------
            # multiprocess of peak shape function
            pool = mp.Pool(8)
            pattern = pool.starmap(y_multi, [(x_val, step, xy_merge, H) for x_val in range(0, total_points)])
            pool.close()
            pattern2 = np.zeros((total_points,2))
            pattern2[:, 1] = np.asarray(pattern)
            pattern2[:, 0] = np.arange(0,180,step)
            # print("Done: XRD pattern {} steps in {:.2f}s".format(total_points, time.time()-startTime))
            startTime = time.time()
            #--------------------------------------------------------

            # Normalization, leaving only 2 decimal
            pattern2[:, 0] = pattern2[:, 0].round(decimals=3)
            pattern2[:, 1] = (pattern2[:, 1] / np.max(pattern2[:, 1])).round(decimals=3)
            # print('Normalization True')

            for noise in range(2):
                # Random noise augmentation
                if noise and peak_shape != 2:  # Peak shape 2 represents a perfect crystal, so should not be augmented
                    pattern2[:, 1] += np.random.randint(2, 20, size=pattern2[:, 1].shape)
                    # pattern2[:, 1] /= np.amax(pattern2[:, 1])
                    pattern2[:, 1] = np.around(pattern2[:, 1] * 1000, decimals=0)

                # Print the plot for preview

                # plt.figure(figsize=(10,4))
                # plt.xlim([x_min, x_max])
                # plt.plot(pattern1[:, 0], pattern1[:,1])
                # plt.show()
                #
                # plt.figure(figsize=(15,4))
                # plt.xlim([x_min, x_max])
                # plt.plot(pattern2[:, 0], pattern2[:,1])
                # plt.xlabel("Two Theta (Degrees)", fontsize=15)
                # plt.ylabel("Normalized Intensity (Counts)", fontsize=15)
                # plt.title(f"An example of generated XRD plot: {cif_file.split('.')[0]} 'F5 K2 Y1'", fontsize=15)
                # plt.grid(alpha=0.2)
                # plt.tight_layout()
                # # figName = "archive_xrd/archive_figure/{}.jpeg".format(cif_file)
                # figName = f"archive_xrd/temp/{cif_file}.jpg"
                # plt.savefig(figName, dpi=600)

                # print ("formula = ", chem_form, "\n", "a = ", cell_a, "\n", "b = ", cell_b, "\n", "c = ", cell_c, "\n", "alpha = ", cell_alpha, "\n", "beta  = ",  cell_beta, "\n", "gamma = ", cell_gamma, "\n")

                # Write to new txt file.
                new_filename = "{}.txt".format(re.split(r"[.]", cif_file)[0])
                with open("{}/{}".format(out_dir, new_filename), "w") as new_file:
                    new_file.write(chem_form + "\n")
                    new_file.write("crystal_structure " + str(int(crystal_sys)) + "\n")
                    new_file.write("space_group " + str(int(space_group)) + "\n")
                    new_file.write("\n".join(str(item).replace("[", "").replace("]", "") for item in pattern2.tolist()))
                # print("Done: pipeline in {:.2f}s".format(time.time()-startTime))

    elif cal_mode == "check_cif":
        print("5/5 Skip: Pattern")

    return "GOOD ✓"
