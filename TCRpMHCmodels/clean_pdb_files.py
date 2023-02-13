# this file cleans pdb files
# output: 1 MHC alpha chain and one 10-mer peptide per file

import os
import Bio
from Bio import PDB
from Bio.PDB import *
from Bio.PDB.Polypeptide import three_to_one

import tempfile 
import subprocess
import numpy as np

import bcr_models as igm
import bcr_models.utils 

############

convert3_1  = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 
			   'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
			   'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
			   'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
			   'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def cleanhet(chain):
	"""Removes heteroatoms from chain"""
	residues  = chain.get_list()

	i = 0
	while i < len(residues):
		if not residues[i].has_id('CA'):
			temp = residues.pop(i)
		else:
			try:
				temp_res = convert3_1[residues[i].get_resname()]
				i += 1
			except KeyError:
				temp = residues.pop(i)
	
	return(residues)

def retrieve_file_from_PDB(pdb_id,pdb_dir):
	# This function will download structures from the PDB 
	# The downloaded structure will be saved in the pdb_dir directory 
	# Saved structures will be name pdb{pdb_id}.ent
	
	pdbl = PDBList()
	pdbl.retrieve_pdb_file(pdb_id,pdir=pdb_dir,file_format='pdb') 
	ent_file = '{}/pdb{}.ent'.format(pdb_dir,pdb_id)
	pdb_file = '{}/{}.pdb'.format(pdb_dir,pdb_id)

	# rename ent to pdb 
	run_cmd(['mv', ent_file, pdb_file])

	return pdb_file

### Finde the distance between two chains ###
def calc_residue_dist(residue_one, residue_two) :
	"""Returns the C-alpha distance between two residues"""
	try:
		diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
	except KeyError:
		print 'KeyError'
		
	return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
	"""Returns a matrix of C-alpha distances between two chains"""
	answer = np.zeros((len(chain_one), len(chain_two)), np.float)

	for row, residue_one in enumerate(chain_one) :
		for col, residue_two in enumerate(chain_two) :
			answer[row, col] = calc_residue_dist(residue_one, residue_two)
	return answer

### Commandline ###

def run_cmd(cmd, input_string=''):
	"""Run the cmd with input_string as stdin and return output."""
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
						 stderr=subprocess.PIPE, universal_newlines=True, close_fds=True)
	out, stderr = p.communicate(input=input_string)
	if p.returncode:
		raise Exception('Cmd {} failed: {}'.format(cmd[0], stderr))
	return out

### Find MHC sequences ###

def hmm_search(seq, hmm):
	# create a temporary file

	fp = tempfile.NamedTemporaryFile()
	fp.write('>seq\n'.format(seq))
	fp.write('{}'.format(seq))
	fp.flush() 

	#Find MHC sequences
	cmd = ['hmmsearch', hmm, fp.name ]

	output = run_cmd(cmd)
	aln = [ line.split() for line in output.splitlines() ]

	# Search for score to see if there is a match
	score = None
	for i, line in enumerate(aln):
		if line[0:3] == ['E-value', 'score', 'bias'] and aln[i+2]:
			try:
				E_value = float(aln[i+2][0])
				score = float(aln[i+2][1])
				break
			except ValueError:
				E_value = float(aln[i+3][0])
				score = float(aln[i+3][1])
				break
	
	# close the file. When the file is closed it will be removed.
	fp.close()	

	return score
	
def hmm_align(hmm, sequence=None, trim=True):
	"""Align a sequence to an HMM profile."""

	#Run hmmalign
	if trim:
		cmd = ['hmmalign', '--trim', hmm, '-']
	else:
		cmd = ['hmmalign', hmm, '-']

	aln = run_cmd(cmd, '>actseq\n' + sequence)

	#Parse output and find aligned sequence 
	aln_seq = []
	for line in aln.splitlines():
		line = line.split()
		if line and line[0] == 'actseq':
			aln_seq.append(line[1])
		
	aln_seq = ''.join(aln_seq)

	if not aln_seq.isupper():
		err = '{}'.format(aln_seq)
		print err

	return aln_seq

### Extracting the sequence from a given chain ###

def get_chain_seq(model, chainID):
	"""Exstracting the sequence from the given chain""" 
	residues = cleanhet(model[chainID])
	seq = ''.join([ convert3_1[x.get_resname()] for x in residues ])
	
	return seq

def make_pdb(pdb_id, pdb_path, out_dir, chain_letters, overwrite=False, struct=None):
		# Create a new PDB file containing only the specified chains. Returns the path to the created file """

		chain_letters = [chain.upper() for chain in chain_letters]

		# Input/output files
		out_name = "%s_%s.pdb" % (pdb_id, ''.join(chain_letters))
		out_path = ''.join([out_dir, '/' ,out_name])

		# Skip PDB generation if the file already exists
		if (not overwrite) and (os.path.isfile(out_path)):
			print("Chain %s of '%s' already extracted to '%s'." % (", ".join(chain_letters), pdb_id, out_name))
			return out_path

		# Get structure, write new file with only given chains
		if struct is None:
			parser = PDBParser()
			struct = parser.get_structure(pdb_id, pdb_path)

		writer = PDB.PDBIO()
		writer.set_structure(struct)
		writer.save(out_path, select=SelectChains(chain_letters))

		return out_path

class SelectChains(PDB.Select):
	""" Only accept the specified chains when saving. """
	def __init__(self, chain_letters):
		self.chain_letters = chain_letters

	def accept_chain(self, chain):
		return (chain.get_id() in self.chain_letters)

###

def make_all_chain_seq_dict(model):

	all_chain_seq_dict = dict() # format: {chian: seq, chian: seq, ...} eg. {'A', 'GSHSMRYFFT...RWE', .., ..}
	
	for chain in model:
		chainID = chain.get_id()
		seq = get_chain_seq(model,chainID)
		
		all_chain_seq_dict[chainID] = seq

	return all_chain_seq_dict


def renumber_chain(pdbchain, hmm=None):
	"""Renumber a Bio.PDB.Chain."""
	seq  = igm.utils.chain2seq(pdbchain)
	
	# Align seg to hmm 
	aln_seq = hmm_align(hmm, sequence=seq, trim=True)

	new_chain = Bio.PDB.Chain.Chain(pdbchain.id)
	pdb_residues = pdbchain.get_list()
	pdb_seq = igm.utils.chain2seq(pdb_residues)

	#print aln_seq
	#print pdb_seq

	pdb_counter = 0

	if aln_seq[0] == '-' and aln_seq[1] != pdb_seq[0] and aln_seq[0:2] != '--':
	 	pdb_counter += 1

	for i, aln_res in enumerate(aln_seq):

		pdb_res = pdb_residues[pdb_counter].copy()
		new_resid = igm.utils.resid2biopdb(i)
		pdb_res.id = new_resid

		if aln_res == '-':
			continue

		#print i, aln_res, pdb_res, pdb_res.id, three_to_one(pdb_res.get_resname())

		# Check for errors
		if aln_res != three_to_one(pdb_res.get_resname()):
			print 'The residue in the aligned pdb sequence is not identical with the residue in the pdb.'
			print 'aln_res: {} and pdb_res: {}'.format(aln_res,three_to_one(pdb_res.get_resname()))
			exit()

		# do not add disordered residues
		if pdb_res.is_disordered() != 1:
			new_chain.add(pdb_res)
		else:
			print 'Disordered residue! pdb_counter: {} and pdb_res: {}'.format(pdb_counter, pdb_res)

		pdb_counter += 1

	return new_chain

def renumber_pmhc_tcr(pdbmodel, hmms=[]):

	new_model = Bio.PDB.Model.Model('')
	mhc_chain,pep_chain,alpha_chain,alpha_chain=None,None,None,None
	
	for chain in pdbmodel:
		chainID = chain.get_id()
		
		seq = get_chain_seq(model, chainID)

		MHC_score = hmm_search(seq,MHC_hmm) 
		TCRalpha_score = hmm_search(seq,TCRalpha_hmm)
		TCRbeta_score = hmm_search(seq,TCRbeta_hmm)

		if MHC_score > 80:
			mhc_chain = Bio.PDB.Chain.Chain('M')
			pdb_residues = cleanhet(pdbmodel[chainID])
			for res in pdb_residues: 
				mhc_chain.add(res)
			
			mhc_chain = renumber_chain(mhc_chain, hmm=MHC_hmm)

		if len(seq) < 20:
			pep_chain = Bio.PDB.Chain.Chain('P')
			pdb_residues = cleanhet(pdbmodel[chainID])
			for res in pdb_residues: 
				pep_chain.add(res)

		if TCRalpha_score > 80:
			alpha_chain = Bio.PDB.Chain.Chain('A')
			pdb_residues = cleanhet(pdbmodel[chainID])
			for res in pdb_residues: 
				alpha_chain.add(res)
			alpha_chain = igm.IgChain.renumber_pdbchain(alpha_chain, hmms=[TCRalpha_hmm])

		if TCRbeta_score > 80:
			beta_chain = Bio.PDB.Chain.Chain('B')
			pdb_residues = cleanhet(pdbmodel[chainID])
			for res in pdb_residues: 
				beta_chain.add(res)

			beta_chain = igm.IgChain.renumber_pdbchain(beta_chain, hmms=[TCRbeta_hmm])

	new_model.add(alpha_chain)
	new_model.add(beta_chain)
	new_model.add(mhc_chain)
	new_model.add(pep_chain)

	return new_model  

### MAIN ###

# how to run 
# python clean_pdb_files.py

### variablers ### 
# All TCR-pMHC from IEDB from after database construction
#pdb_list = ['3PQY', '5IVX', ....]
pdb_list = ['3PQY']

h_dir = '/Users/s112819/Desktop/CleanPDBfiles/data' 
p_dir = '{}/raw_data/PDBs'.format(h_dir) 
out_dir = '{}/cleaned_data/PDBs'.format(h_dir)
b_dir = '{}/cleaned_data/fasta'.format(h_dir)

MHC_hmm = '{}/hmms/MHC_I_complete.hmm'.format(h_dir)
TCRalpha_hmm = '{}/hmms/alpha.hmm'.format(h_dir)
TCRbeta_hmm = '{}/hmms/beta.hmm'.format(h_dir)

threshold = 10 # the minimum distance between two chains 
peptide_length = 20 # the minimum distance between two chains   
hmm_threshold = 80 # the % treshold for hmm search

# METHOD:
# For each PDB: 
# 1) find the MHC molecule
# 2) find the peptide bound to this MHC molecule 
# 3) find the TCR alpha and beta chain 

alignedout = open('{}/cleaned_data/aligned_TCRpMHC.csv'.format(h_dir),'w')
alignedout.write('pdb_name,tcr_alpha_seq,tcr_beta_seq,mhc_seq,peptide_seq\n')

# loop through all the pdbs in the pdb list 
for pdb in pdb_list:
	print('Running PDB {}'.format(pdb))

	pdb = pdb.lower()
	pdbout = open('{}/{}.fsa'.format(b_dir,pdb),'w')

	finaldict = dict() # fromat: {chain names: sequences, ...}

	# downloading pdbs and setting the path downloaded fiels 
	pdb_path = retrieve_file_from_PDB(pdb,p_dir)
	
	# import the structure as a model object 
	structure = PDBParser().get_structure(pdb,pdb_path) 
	model = structure[0] 

	TCRpHMC_chains = list() # list for the chain names of the TCR-pMHC
	all_chain_seq_dict = make_all_chain_seq_dict(model) # fromat: {chain names: sequences, ...}  

	# loop through the different chains in the model 
	for chain in model:
		chainID = chain.get_id()
		seq = get_chain_seq(model, chainID)
		
		# Find the HMM score
		MHC_score = hmm_search(seq, MHC_hmm) 

		# Find the MHC molecule
		if MHC_score > hmm_threshold:
			
			# appending the MHC chain
			MHC_chain = chainID	 
			TCRpHMC_chains.append(MHC_chain) 
			
			# remove the chain form the dictinoary
			all_chain_seq_dict.pop(MHC_chain)
			
			#print 'MHC:', chainID, seq
			pdbout.write('>{}:M\n{}\n'.format(pdb,seq))
			MHCaligned = hmm_align(MHC_hmm, sequence=seq, trim=True)
			finaldict['M'] = MHCaligned

			# Find the peptide which is in contact with the current MHC
			for test_chain in all_chain_seq_dict: 
					
				# 1) Heteroatoms are removed from the chains
				# 2) Calculate the distance between all atoms the two chains 
				# 3) Finding the minimum distance

				test_seq = all_chain_seq_dict[test_chain]
				residues = cleanhet(model[MHC_chain])
				test_residues = cleanhet(model[test_chain]) 
				dist_matrix = calc_dist_matrix(test_residues, residues)
				min_dist = np.min(dist_matrix)

				# The peptide chain will always be smaller than X amino acids (X = peptide_length)
				# The minimum distance between the two chains is >= X aangstroem (X = threshold)
				if len(test_seq) < peptide_length and min_dist < threshold: 

					#print 'peptide: ', test_chain, test_seq
					pdbout.write('>{}:P\n{}\n'.format(pdb,test_seq)) 
					finaldict['P'] = test_seq

					# the current test chain is the peptide chain 
					peptide_chain = test_chain
					TCRpHMC_chains.append(peptide_chain) # appending the peptide chain 
					all_chain_seq_dict.pop(peptide_chain)

					# Find the TCR chains which are in contact with the current peptide
					for test_chain in all_chain_seq_dict:

						test_seq = all_chain_seq_dict[test_chain]
						
						# 1) Heteroatoms are removed from the chains
						# 2) Calculate the distance between all atoms the two chains 
						# 3) Finding the minimum distance

						residues = cleanhet(model[peptide_chain])
						test_residues = cleanhet(model[test_chain])
						dist_matrix = calc_dist_matrix(test_residues, residues)
						min_dist = np.min(dist_matrix)

						# Find the HMM scores
						TCRalpha_score = hmm_search(test_seq,TCRalpha_hmm)
						TCRbeta_score = hmm_search(test_seq,TCRbeta_hmm)

						# the TCR alpha chain
						if TCRalpha_score > hmm_threshold and min_dist < threshold:
							#print 'TCRalpha: ', test_chain, test_seq, TCRalpha_score
							pdbout.write('>{}:A\n{}\n'.format(pdb,test_seq)) 
							TCRpHMC_chains.append(test_chain) # appending the peptide chain 
							TCRalpha_aligned = hmm_align(TCRalpha_hmm, sequence=test_seq, trim=True)
							finaldict['A'] = TCRalpha_aligned

						# the TCR beta chain
						if TCRbeta_score > hmm_threshold and min_dist < threshold:
							#print 'TCRbeta: ', test_chain, test_seq, TCRbeta_score
							pdbout.write('>{}:B\n{}\n'.format(pdb,test_seq)) 
							TCRpHMC_chains.append(test_chain) # appending the peptide chain
							TCRbeta_aligned = hmm_align(TCRbeta_hmm, sequence=test_seq, trim=True)
							finaldict['B'] = TCRbeta_aligned 
					
					break # end the loop to find only one TCR alpha and beta
			break # end the loop to find only one MHC

	# save the MHC:peptide complex
	if len(TCRpHMC_chains) == 4:
		out_path = make_pdb(pdb, pdb_path, out_dir, TCRpHMC_chains)
		parser = PDBParser()
		struct = parser.get_structure(pdb, out_path)[0]

		renamed_model = renumber_pmhc_tcr(struct,hmms=[MHC_hmm,TCRalpha_hmm,TCRbeta_hmm])

		writer = PDB.PDBIO()
		writer.set_structure(renamed_model)
		writer.save('{}/{}_cleaned.pdb'.format(out_dir,pdb))

	# write the aligned sequences from the TCR-pMHC complex
	alignedout.write(pdb + ',' + ','.join(['{}'.format(finaldict[pdbid]) for pdbid in sorted(finaldict)]) + '\n')
	pdbout.close()
	print('DONE')

alignedout.close()





