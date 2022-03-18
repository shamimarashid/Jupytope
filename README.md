# Jupytope
computational extraction of viral epitopes
# This tool will extract structural properties for viral epitopes from a set of PDB chains.

This program was tested in a Linux environment. 
Running Jupyter Notebook: 
ver 2.2.8 & Python: ver 3.6.11

# Dependencies:
1. NACCESS
2. DSSP
3. MSMS
4. HSE (from Bio.PDB import HSExposure )
5. numpy
6. pandas


# Installation
The following programs should be downloaded and installed from their individual websites:

1. NACCESS (http://www.bioinf.manchester.ac.uk/naccess/)
2. DSSP (https://swift.cmbi.umcn.nl/gv/dssp/)
3. MSMS (https://ccsb.scripps.edu/mgltools/#msms)

These binaries or executables should be added to path: 'naccess', 'dssp', 'msms'.
Names are case sensitive for *nix-like systems. 
Non-default executable names may be attempted, but might cause an error in the Biopython parsers.

#  File Setup 

It is recommended to create an antigen directory from within the Jupytope folder, e.g. 'Spike'.
Then, two sub-directories (Chains and Alignments) should be created:

# 1. Spike/Chains
Should contain individual chains of the PDB files, named with the chain identifier: (e.g. 7DZW_A.pdb)
Not all chains are needed, only the viral antigen chain(s) of interest.
See code 'ChainSplitter.py' of David Cain. Alternatively, the Biopython PDBParser can be used to extract the chains.

# 2. Spike/Alignments
Should be created, and left blank. 
# 3. Epitope Sites
A list of epitope sites of interest (see example: Sites7DZW_Num.txt)
# 4. PDB List
A list of PDBIDs, Chains, and Year or Strain (see example: SARSCoV2_PDBID.txt). 
If Year or Strain data are not available, a placeholder such as 'x' or 'Absent' should be used
#5 . Reference Structure
PDBID and chain of a reference structure

# Running the Notebook

Then the program can be run by changing the appropriate input and output filenames (see first cell of notebook).

Jupytope can be used to extract properties for any list of residues of interest (not just epitopes). 
Just provide the same PDB and chain IDs for reference and query structures. 
The full list of extracted features is available in our publication.

cryo-EM files may slow the system and raise exceptions when computing normalized B-factors. 
However, if allowed to complete, the other features will be computed as normal.

Data that could not be extracted will be marked as 'None' or 'NaN'.
For example, missing residues in the query structure at a given epitope position. 


# Citation
If you use Jupytope as part of your workflow, please consider citing our paper:
"Jupytope: Computational extraction of structural properties of viral epitopes", Shamima Rashid, Ng Teng Ann and Kwoh Chee Keong. Mar.2022. Nanyang Technological University.
