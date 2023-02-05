from __future__ import division

import os
import csv
import copy
import tempfile

import Bio.PDB
from Bio.PDB.Polypeptide import three_to_one

import bcr_models as igm

from modeller import *
from modeller.automodel import *

def save_pdb_model(model, outfile, name='model'):
    """Save the PDB model structure."""

    new_structure = Bio.PDB.Structure.Structure(name)
    new_structure.add(model)

    io = Bio.PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(outfile)

def getPDB(pdbfile):
    """Extracting the model object from a PDB file"""

    p = Bio.PDB.PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdbfile, pdbfile)
    model = structure[0]
    
    return model

def save_templates(templates, out_file='outputfile.csv', comp_type=None):
    """Save a template dictionary"""
    
    if templates != {}:
        sorted_list = sorted(templates.items(), key=lambda x: x[1]['identity_score'], reverse=True)
        w = open(out_file,'w')
        if comp_type == 'pMHC':
            w.write('pdb_id,pep_identity,mhc_identity,total_identity\n')
            for item in sorted_list:
                pdb_id, total_ident, pep_ident, mhc_ident = item[0], item[1]['identity_score'], item[1]['pep_identity']*100, item[1]['mhc_identity']*100
                w.write('{},{},{},{}\n'.format(pdb_id, pep_ident, mhc_ident, total_ident))
        elif comp_type == 'TCR-pMHC':
            w.write('pdb_id,pep_identity,mhc_identity,tcrA_identity,tcrB_identity,total_identity\n')
            for item in sorted_list:    
                pdb_id, total_ident, pep_ident, mhc_ident, tcrA_ident, tcrB_ident = item[0], item[1]['identity_score'], item[1]['pep_identity']*100, item[1]['mhc_identity']*100, item[1]['tcr_alpha_identity']*100, item[1]['tcr_beta_identity']*100
                w.write('{},{},{},{},{},{}\n'.format(pdb_id, pep_ident, mhc_ident, tcrA_ident, tcrB_ident, total_ident))
        else:
            print('Error in comp_type. Should be "pMHC" or "TCR-pMHC" comp_type = {}'.format(comp_type))
            exit()
        w.close()

def hmmalign(hmm, sequence=None, trim=True):
    """Align a sequence to an HMM profile."""
    #Run hmmalign
    if trim:
        cmd = [igm.utils.HMMALIGN_BIN, '--trim', hmm, '-']
    else:
        cmd = [igm.utils.HMMALIGN_BIN, hmm, '-']

    aln = igm.utils.run_cmd(cmd, '>actseq\n' + sequence)

    #Parse output and find aligned sequence 
    aln_seq = []
    for line in aln.splitlines():
        line = line.split()
        if line and line[0] == 'actseq':
            aln_seq.append(line[1])
        
    aln_seq = ''.join(aln_seq)

    if not aln_seq.isupper():
        err = '{}'.format(aln_seq)
        print(err)

    return aln_seq

def hmmsearch(seq, hmms):
    """Find the best alignment from the hmms and align the sequence."""

    hmmsearch_scores = dict()
    for hmm in hmms:
        #Run hmmsearch
        aln_raw = igm.utils.run_cmd([igm.utils.HMMSEARCH_BIN, hmm, '-'],
            '>actseq\n' + seq)
        aln = [line.split() for line in aln_raw.splitlines()]
        #Parse
        score, E_value = None, None
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
        
        #Error checking
        if score is not None:
            #Register score and E_value
            hmmsearch_scores[hmm] = score, E_value

    top_hmm = max(hmmsearch_scores.items(), key=lambda x: x[1][0])
    
    best_hmm, best_score, best_E_value = top_hmm[0], top_hmm[1][0], top_hmm[1][1]  
    return best_hmm, best_score, best_E_value

def renumber_chain(pdbchain, hmms=None):
    """Renumber a Bio.PDB.Chain."""
    seq  = igm.utils.chain2seq(pdbchain)
    
    # find best hmm
    hmm, score, E_value = hmmsearch(seq, hmms)
    #print hmm, score, E_value

    # Align seg to hmm 
    aln_seq = hmmalign(hmm, sequence=seq, trim=False)

    new_chain = Bio.PDB.Chain.Chain(pdbchain.id)
    pdb_residues = pdbchain.get_list()

    pdb_counter = 0
    for i, aln_res in enumerate(aln_seq):

        if aln_res == '-':
            continue

        pdb_res = pdb_residues[pdb_counter].copy()
        new_resid = igm.utils.resid2biopdb(i+1)
        pdb_res.id = new_resid

        # Check for errors
        if aln_res != three_to_one(pdb_res.get_resname()):
            print('The residue type in the aligned pdb sequence is not identical with the residue in the pdb.\n')
            print('aln_res: {} and pdb_res: {}'.format(aln_res,three_to_one(pdb_res.get_resname())))
            exit()
        
        #print i, aln_res, pdb_res, pdb_res.id, three_to_one(pdb_res.get_resname())

        new_chain.add(pdb_res)
        pdb_counter += 1

    return new_chain
    
def renumber_pmhc(pdbmodel, length=20, hmms=None):
    """Renumber a Bio.PDB.Model."""
    
    new_model = Bio.PDB.Model.Model('')
    for chain in pdbmodel:
        
        # Add peptide chain
        seq = igm.utils.chain2seq(chain)
        if len(seq) < length:
            new_model.add(chain)
            continue

        new_chain = renumber_chain(chain, hmms)
        new_model.add(new_chain)

    return new_model


def find_templates(template_db=None, blacklist=(), comp_type=None, peptide_seq=None, aln_mhc_seq=None, tcr_alpha_seq=None, tcr_beta_seq=None, pmhc_thres=None, complex_thres=None):

    """Select templates from template database"""
    # comp_type = 'pMHC' or pMHC-TCR'

    if comp_type != 'pMHC' and comp_type != 'pMHC-TCR':
        print('comp_type should be pMHC or pMHC-TCR. comp_type = {}'.format(comp_type))
        exit()    

    templates = {} # format: {'pdb_name':{'peptide_seq': 'SALQNAASIA', 'mhc_seq': 'GSHSMRYFY...', 'blosum_score': 956, 'seq_id': 0.946}, {...}}

    # calculate the sequence idenitity between the target and all templates in the database 
    for temp_id in template_db:

        temp_peptide_seq = template_db[temp_id]['peptide_seq']
        temp_mhc_seq = template_db[temp_id]['mhc_seq']
        
        #temp_pmhc_seq = temp_mhc_seq + temp_peptide_seq   
        #target_pmhc_seq = aln_mhc_seq+peptide_seq
        #blosum_score, identity_score = igm.utils.pairwise_score(target_pmhc_seq,temp_pmhc_seq)

        # Exclude templates from the blacklist
        if temp_id in blacklist:
            continue

        # Use only templates with same length of peptide to the target
        if len(peptide_seq) != len(temp_peptide_seq):
            continue

        mhc_blosum, mhc_identity = igm.utils.pairwise_score(aln_mhc_seq,temp_mhc_seq)
        pep_blosum, pep_identity = igm.utils.pairwise_score(peptide_seq,temp_peptide_seq)

        if float(pep_identity)*100 <= 6/9*100:  # RESTRICTIONS
            if comp_type == 'pMHC':
                blosum_score = float(mhc_blosum)*(1/2)+float(pep_blosum)*(1/2)
                identity_score = float(mhc_identity)*(1/2)+float(pep_identity)*(1/2)

                # Include templates with a identity score below the threshold
                identity_score = float(identity_score)*100
                if float(pmhc_thres) >= identity_score:
                    templates[temp_id] = {'peptide_seq': temp_peptide_seq, 'mhc_seq': temp_mhc_seq, 'blosum_score': blosum_score, 'identity_score': identity_score, 'pep_identity':pep_identity, 'mhc_identity':mhc_identity}
            
            if comp_type == 'pMHC-TCR': 
                temp_tcr_alpha_seq = template_db[temp_id]['tcr_alpha_seq']
                temp_tcr_beta_seq = template_db[temp_id]['tcr_beta_seq']
        
                tcr_alpha_blosum, tcr_alpha_identity = igm.utils.pairwise_score(tcr_alpha_seq,temp_tcr_alpha_seq)
                tcr_beta_blosum, tcr_beta_identity = igm.utils.pairwise_score(tcr_beta_seq,temp_tcr_beta_seq)
                blosum_score = float(mhc_blosum)*(2/6)+float(pep_blosum)*(2/6)+float(tcr_alpha_blosum)*(1/6)+float(tcr_beta_blosum)*(1/6)           # <= RESTRICTIONS
                #blosum_score = float(mhc_blosum)*(1/4)+float(pep_blosum)*(1/4)+float(tcr_alpha_blosum)*(1/4)+float(tcr_beta_blosum)*(1/4)           #  <= RESTRICTIONS
                
                ## investigate weight on MHC and peptide #### !!!!!
                ## orig: 
                identity_score = float(mhc_identity)*(2/6)+float(pep_identity)*(2/6)+float(tcr_alpha_identity)*(1/6)+float(tcr_beta_identity)*(1/6)        #  <= RESTRICTIONS
                #identity_score = float(mhc_identity)*(1/4)+float(pep_identity)*(1/4)+float(tcr_alpha_identity)*(1/4)+float(tcr_beta_identity)*(1/4)                 #  <= RESTRICTIONS

                # Include templates with a identity score below the threshold
                identity_score = float(identity_score)*100
                if float(tcr_alpha_identity)*100 <= 95 and float(tcr_beta_identity)*100 <= 95:      #  <= RESTRICTIONS
                    if float(complex_thres) >= identity_score:
                        templates[temp_id] = {'peptide_seq': temp_peptide_seq, 'mhc_seq': temp_mhc_seq, 'tcr_alpha_seq': temp_tcr_alpha_seq, 'tcr_beta_seq': temp_tcr_beta_seq, 'blosum_score': blosum_score, 'identity_score': identity_score, 'pep_identity':pep_identity, 'mhc_identity':mhc_identity, 'tcr_alpha_identity':tcr_alpha_identity, 'tcr_beta_identity':tcr_beta_identity}
        
    n_length = len(templates)
    
    if n_length == 0: 
        print('No templates were found! pmhc_thres = {}, complex_thres = {}, blacklist = {}, peptide length = {}\n'.format(pmhc_thres,complex_thres,blacklist,len(peptide_seq)))
        exit()

    return templates

def top_n_templates(sorted_temp_list,num_temp):

    num_temp = int(num_temp)
    top_n_templates = {}
    for i,s in enumerate(sorted_temp_list):
        key, value, score = s[0], s[1], s[1]['identity_score']
        
        if i+1 <= num_temp:
            top_n_templates[key] = value

    n_length = len(top_n_templates)
    if n_length < num_temp:
        print('Only {} templates were found! num_temp = {}'.format(n_length,num_temp))
        exit()

    return top_n_templates 

def multi_templates(sorted_temp_list,comp_type=None):
    top_templates = {}

    best_temp_id = sorted_temp_list[0][0]
    best_temp_score = sorted_temp_list[0][1]['identity_score']

    for temp in sorted_temp_list: 
        pdb_id, identity_score = temp[0], temp[1]['identity_score']

        # always add the template with the highest siquence similarity to the target
        if pdb_id == best_temp_id:
            top_templates[pdb_id]=temp[1]
            continue

        # calculate the similarity score between the current temp_complex_seq to the complex sequence from the templates found in the top_templates
        flag = 0

        for top_temp in sorted(top_templates.items(), key=lambda x: x[1]['identity_score'], reverse=True): 
            top_pdb_id, top_identity_score = top_temp[0], top_temp[1]['identity_score']

            top_mhc_blosum, top_mhc_identity = igm.utils.pairwise_score(top_temp[1]['mhc_seq'],temp[1]['mhc_seq'])
            top_pep_blosum, top_pep_identity = igm.utils.pairwise_score(top_temp[1]['peptide_seq'],temp[1]['peptide_seq'])
            
            if comp_type=='pMHC':
                top_blosum_score = float(top_mhc_blosum)*(1/2)+float(top_pep_blosum)*(1/2)
                top_identity_score = float(top_mhc_identity)*(1/2)+float(top_pep_identity)*(1/2)
                top_identity_score = float(top_identity_score)*100 

            elif comp_type=='pMHC-TCR':
                top_tcr_alpha_blosum, top_tcr_alpha_identity = igm.utils.pairwise_score(top_temp[1]['tcr_alpha_seq'],temp[1]['tcr_alpha_seq'])
                top_tcr_beta_blosum, top_tcr_beta_identity = igm.utils.pairwise_score(top_temp[1]['tcr_beta_seq'],temp[1]['tcr_beta_seq'])
                top_blosum_score = float(top_mhc_blosum)*(2/6)+float(top_pep_blosum)*(2/6)+float(top_tcr_alpha_blosum)*(1/6)+float(top_tcr_beta_blosum)*(1/6)                   # <= RESTRICTION
                #top_blosum_score = float(top_mhc_blosum)*(1/4)+float(top_pep_blosum)*(1/4)+float(top_tcr_alpha_blosum)*(1/4)+float(top_tcr_beta_blosum)*(1/4)                   # <= RESTRICTION
                top_identity_score = float(top_mhc_identity)*(2/6)+float(top_pep_identity)*(2/6)+float(top_tcr_alpha_identity)*(1/6)+float(top_tcr_beta_identity)*(1/6)         # <= RESTRICTION
                #top_identity_score = float(top_mhc_identity)*(1/4)+float(top_pep_identity)*(1/4)+float(top_tcr_alpha_identity)*(1/4)+float(top_tcr_beta_identity)*(1/4)         # <= RESTRICTION
                top_identity_score = float(top_identity_score)*100

            else:
                print('comp_type should be pMHC or pMHC-TCR. comp_type = {}'.format(comp_type))
                exit()   

            if top_identity_score >= best_temp_score*0.95:
                flag = 1
                break

        #print flag, identity_score, top_identity_score, best_temp_score*0.95, best_temp_score*0.80        

        if flag == 0 and identity_score >= best_temp_score*0.80:
            #print pdb_id, top_identity_score                        
            top_templates[pdb_id]=temp[1]

    n_length = len(top_templates)
    if n_length == 0: 
        print('No templates were found! pmhc_thres = {}, complex_thres = {}, blacklist = {}'.format(pmhc_thres,complex_thres,blacklist))
        exit()
        
    return top_templates


########################################################################
#####                         pMHC class                           #####
########################################################################

class pMHC(object):
    """
    Main class to model a single pMHC structure.
    """

    def __init__(self, target_id, mhc_seq, peptide_seq, template_db, template_folder):
        
        # -- Set by init()
        self.target_id          = target_id
        self.peptide_seq        = str(peptide_seq)
        self.mhc_seq            = str(mhc_seq)
        self.template_db        = template_db # already aligned with data/MHC_I_complete.hmm
        self.template_folder    = template_folder  
        
        #Set by hmmalign()
        self.aln_mhc_seq        = None

        #Set by hmmsearch()
        self.hmmsearch_scores   = {}        

        #Set by find_pmhc_templates()
        self.pmhc_templates     = {}  
        self.top_n_templates    = {}
        self.top_templates      = {} 

        #Set by get_aligned_pdb()
        #self._pdb_templates     = {}

        #Set by build_pmhc_structure()
        self.pmhc_model_file     = None
        self.pmhc_model          = None 
        # 
        #self.mhc_chian
        #self.aligned_chain      = None

        # renumber 
        self.pdbmodel_renum = None


    def find_pmhc_templates(self, blacklist=(), pmhc_thres=None, pmhc_temp=None, save_temp=False, path=''):
        """Select templates from template database"""

        #Format of templates: {'pdb_name':{'peptide_seq': 'SALQNAASIA', 'mhc_seq': 'GSHSMRYFY...','blosum_score': 956, 'seq_id': 0.946}, {...}}
        initial_templates = find_templates(template_db=self.template_db, blacklist=blacklist, comp_type='pMHC', peptide_seq=self.peptide_seq, aln_mhc_seq=self.aln_mhc_seq, pmhc_thres=pmhc_thres)   
        self.pmhc_templates = initial_templates
        
        sorted_temp_list = sorted(self.pmhc_templates.items(), key=lambda x: x[1]['identity_score'], reverse=True) 

        #### Top N best template ####
        if pmhc_temp.isdigit() == True:
            top_templates = top_n_templates(sorted_temp_list,pmhc_temp)    
            self.top_templates = top_templates      

        #### Multitemplate method ####
        if pmhc_temp == 'multi':
            top_templates = multi_templates(sorted_temp_list,comp_type='pMHC')           
            self.top_templates = top_templates
        
        # save templates
        if save_temp:
            save_templates(self.top_templates, out_file='{}{}-pmhc-templates.csv'.format(path,self.target_id), comp_type='pMHC')

    def seq2pir(self):
        
        target_id = self.target_id
        templates = self.top_templates
        # make random filname if non indicated
        rondom_name = next(tempfile._get_candidate_names())
        pir_filename = '{}.{}.pir'.format(target_id,rondom_name) 
        outfile_pir = open(pir_filename,'w')
          
        # Add all relevant template sequences
        #for temp_id in templates:
        for temp in sorted(templates.items(), key=lambda x: x[1]['identity_score'], reverse=True):    
            temp_id, temp_dict, score = temp[0], temp[1], temp[1]['identity_score']
            temp_mhc_seq, temp_peptide_seq = temp_dict['mhc_seq'], temp_dict['peptide_seq']
            outfile_pir.write('>P1;{}\nstructureX:{}.pdb::M::::::\n'.format(temp_id,temp_id))
            outfile_pir.write('{}/{}*\n'.format(temp_mhc_seq, temp_peptide_seq))

            # Add target sequence  
            outfile_pir.write('>P1;{}_pMHC\nsequence:{}_pMHC::::::::\n'.format(target_id,target_id))
            outfile_pir.write('{}/{}*\n'.format(self.aln_mhc_seq, self.peptide_seq))     

        outfile_pir.close()
        return pir_filename

    def build_pmhc_structure(self, n_start=1, n_end=1, model_seed=-12312, verbose=False, save_pir='', path=''):
        print(os.getcwd())
        # make input pir file for MODELLER 
        pir_filename = self.seq2pir()

        if save_pir != '':
            os.system('cp {} {}'.format(pir_filename,save_pir))

        # generate model 
        if verbose is False:
            log.level(output=0, notes=0, warnings=0, errors=0, memory=0)
            log.none() 

        pdb_folder = self.template_folder
        templates = self.top_templates
        template_ids = [temp_id for temp_id in templates]
    
        class MyModel(automodel):
                def special_patches(self, aln):
                        self.rename_segments(segment_ids=['M','P'],renumber_residues=[1,1])

        env = environ(rand_seed=model_seed)
        env.io.atom_files_directory = [pdb_folder, path]
        #env.io.hetatm = True

        a=MyModel(env, 
                    alnfile=pir_filename, 
                    knowns=template_ids, 
                    sequence=self.target_id+'_pMHC',  
                    )

        #a=MyModel(env, 
        #            alnfile=pir_filename, 
        #            knowns=template_ids, 
        #            sequence=self.target_id+'_pMHC',
        #            assess_methods=(assess.DOPE) # DOPE (Discrete Optimized Protein Energy)   
        #            )

        # For fast optimization
        #a.very_fast() 

        a.starting_model = n_start
        a.ending_model = n_end
          
        a.make()  

        model_name = '{}_pMHC.B99990001.pdb'.format(self.target_id)
        pmhc_model = getPDB(model_name)

        self.pmhc_model = pmhc_model    

        # remove files generated by modeller 
        
        #os.system('rm *.B9999* *.D0000* *.V9999* *.ini *.rsr *.sch')
        os.system("rm {0}*.B9999* {0}*.D0000* {0}*.V9999* {0}*.ini {0}*.rsr {0}*.sch".format(self.target_id+'_pMHC'))
        os.system('rm {}'.format(pir_filename))


########################################################################
#####                   Build pMHC-TCR complex                     #####
########################################################################

class pMHC_TCR_complex():

    def __init__(self, target_id, pmhc_model, tcr_alpha_model, tcr_beta_model,
        peptide_seq, aln_mhc_seq, aln_tcr_alpha_seq, aln_tcr_beta_seq, template_db, template_folder, hmms):

        # Set by init()
        self.target_id          = target_id
        self.template_db        = template_db
        self.template_folder    = template_folder 

        # HMM aligned sequences  
        self.peptide_seq        = peptide_seq 
        self.aln_mhc_seq        = aln_mhc_seq
        self.aln_tcr_alpha_seq  = aln_tcr_alpha_seq
        self.aln_tcr_beta_seq   = aln_tcr_beta_seq

        # PDB file name of models 
        self.pmhc_model         = pmhc_model
        self.tcr_alpha_model    = tcr_alpha_model
        self.tcr_beta_model     = tcr_beta_model

        # HMMs
        self.hmms               = hmms 

        # set by find_pMHC_TCR_templates()
        self.pmhc_tcr_templates = {} # all pMHC-TCR templates 
        self.top_templates      = {}
        self.top_n_templates    = {}
        self.top_temp_list      = []
        self.pmhc_tcr_model     = None

        # set by hmmalign_TCR_model()
        self.aln_tcr_model_alpha_seq  = None
        self.aln_tcr_model_beta_seq   = None

        # renumbered pMHC-TCR model
        self.pdbmodel_renum    = None  

    def align_seq(self, seq, hmms=''):
        # find best hmm and align seg to hmm 
        hmm, score, E_value = hmmsearch(seq, hmms)
        aln_seq = hmmalign(hmm, sequence=seq, trim=True)

        return aln_seq

    def hmmalign_TCR_model(self):

        tcr_alpha_model = getPDB(self.tcr_alpha_model)
        #print tcr_alpha_model
        #print tcr_alpha_model['A']
        
        alpha_seq = igm.utils.chain2seq(tcr_alpha_model['A'])
        self.aln_tcr_model_alpha_seq = self.align_seq(alpha_seq, hmms=self.hmms)
 
        tcr_beta_model = getPDB(self.tcr_beta_model)
        beta_seq = igm.utils.chain2seq(tcr_beta_model['B'])
        self.aln_tcr_model_beta_seq = self.align_seq(beta_seq, hmms=self.hmms)


    def find_pMHC_TCR_templates(self, blacklist=(), complex_thres=0, complex_temp=None, save_temp=False, path=''):
        """Select templates from template database"""

        template_db = self.template_db

        initial_templates = find_templates(template_db=self.template_db, blacklist=blacklist, comp_type='pMHC-TCR', peptide_seq=self.peptide_seq, aln_mhc_seq=self.aln_mhc_seq, tcr_alpha_seq=self.aln_tcr_alpha_seq, tcr_beta_seq=self.aln_tcr_beta_seq, complex_thres=complex_thres)   
        self.pmhc_tcr_templates = initial_templates
        
        sorted_temp_list = sorted(self.pmhc_tcr_templates.items(), key=lambda x: x[1]['identity_score'], reverse=True)

        #### Top N best template ####
        if complex_temp.isdigit() == True:
            top_templates = top_n_templates(sorted_temp_list,complex_temp)    
            self.top_templates = top_templates

        #print "\ntop_n_templates"
        #print self.top_n_templates
        
        #### Multitemplate method ####
        if complex_temp == 'multi':
            top_templates = multi_templates(sorted_temp_list,comp_type='pMHC-TCR')           
            self.top_templates = top_templates
    
        if save_temp:
            save_templates(self.top_templates, out_file='{}{}-complex-templates.csv'.format(path,self.target_id), comp_type='TCR-pMHC')
            
        #print "\ntop_templates"
        #print self.top_templates


    def seq2pir(self, outfile_pir=None):
        
        target_id = self.target_id
        templates = self.top_templates 

        #print templates
        # PDB file name of taget models
        pmhc_model_id = self.pmhc_model.split('/')[-1].replace('.pdb','')
        tcr_alpha_model_id = self.tcr_alpha_model.split('/')[-1].replace('.pdb','') 
        tcr_beta_model_id  = self.tcr_beta_model.split('/')[-1].replace('.pdb','')

        # Add models to top_temp_list 
        # It is important to have the templates before the models in the top_temp_list!
        self.top_temp_list.extend([temp_id for temp_id in templates])
        self.top_temp_list.append(pmhc_model_id)
        self.top_temp_list.append(tcr_alpha_model_id)
        self.top_temp_list.append(tcr_beta_model_id)
        
        # Realign the TCR chains 
        self.hmmalign_TCR_model()   
        #print self.aln_tcr_model_alpha_seq

        # HMM aligned sequences from target  
        peptide_seq         = self.peptide_seq       
        aln_mhc_seq         = self.aln_mhc_seq      
        aln_tcr_alpha_seq   = self.aln_tcr_model_alpha_seq    
        aln_tcr_beta_seq    = self.aln_tcr_model_beta_seq         

        pep_length  = len(peptide_seq) 
        mhc_length  = len(aln_mhc_seq)
        tcra_length = len(aln_tcr_alpha_seq)
        tcrb_length = len(aln_tcr_beta_seq)         

        # make random filname if non indicated
        if not outfile_pir:
            rondom_name = next(tempfile._get_candidate_names()) 
            pir_filename = '{}.{}.pir'.format(target_id,rondom_name)  # make random filname if non indicated
            outfile_pir = open(pir_filename,'w')

        # Add the pMHC model as template 
        outfile_pir.write('>P1;{}\n'.format(pmhc_model_id))
        outfile_pir.write('structureX:{}::M::P::::\n'.format(pmhc_model_id,mhc_length))
        outfile_pir.write('{}/{}/{}/{}*\n'.format(aln_mhc_seq, peptide_seq, '-'*tcra_length, '-'*tcrb_length))        

        # Add the TCR alpha model as template
        outfile_pir.write('>P1;{}\n'.format(tcr_alpha_model_id))
        outfile_pir.write('structureX:{}::A::A::::\n'.format(tcr_alpha_model_id))
        outfile_pir.write('{}/{}/{}/{}*\n'.format('-'*mhc_length, '-'*pep_length, aln_tcr_alpha_seq, '-'*tcrb_length))        

        # Add the TCR beta model as template
        outfile_pir.write('>P1;{}\n'.format(tcr_beta_model_id))
        outfile_pir.write('structureX:{}::B::B::::\n'.format(tcr_beta_model_id,tcrb_length))
        outfile_pir.write('{}/{}/{}/{}*\n'.format('-'*mhc_length, '-'*pep_length, '-'*tcra_length, aln_tcr_beta_seq))  

        # Add all relevant template sequences
        #for temp_id in templates:

        for temp in sorted(templates.items(), key=lambda x: x[1]['identity_score'], reverse=True):
             
            temp_id, identity_score = temp[0], temp[1]['identity_score']
            temp_mhc_seq, temp_peptide_seq = temp[1]['mhc_seq'], temp[1]['peptide_seq']
            temp_tcr_alpha, temp_tcr_beta  = temp[1]['tcr_alpha_seq'], temp[1]['tcr_beta_seq'] 
     
            outfile_pir.write('>P1;{}\n'.format(temp_id))
            outfile_pir.write('structureX:{}.pdb::M::B::::\n'.format(temp_id,temp_id,mhc_length))
            outfile_pir.write('{}/{}/{}/{}*\n'.format(temp_mhc_seq, temp_peptide_seq, temp_tcr_alpha, temp_tcr_beta))

        # Add target sequence  
        outfile_pir.write('>P1;{}_pMHC_TCR\n'.format(target_id))
        outfile_pir.write('sequence:{}_pMHC_TCR::::::::\n'.format(target_id))
        outfile_pir.write('{}/{}/{}/{}*\n'.format(aln_mhc_seq, peptide_seq, aln_tcr_alpha_seq, aln_tcr_beta_seq))

        outfile_pir.close()

        return pir_filename

    def build_pmhc_tcr_structure(self, n_start=1, n_end=1, model_seed=-12312, verbose=False, save_pir='',path=''):
        # make input pir file for MODELLER 
        print(os.getcwd())
        pir_filename = self.seq2pir()

        if save_pir != '':
            os.system('cp {} {}'.format(pir_filename,save_pir))

        # generate model 
        if verbose is False:
            log.level(output=0, notes=0, warnings=0, errors=0, memory=0)
            log.none() 

        #print self.top_temp_list
        template_ids = [temp_id for temp_id in self.top_temp_list]
        #print template_ids

        class MyModel(automodel):
                def special_patches(self, aln):
                        self.rename_segments(segment_ids=['M','P','A','B'],renumber_residues=[1, 1, 1, 1])    


        env = environ(rand_seed=model_seed)
        env.io.atom_files_directory = [self.template_folder, path, './']        
        
        # Include hetero atom 
        #env.io.hetatm = True

        a=MyModel(env, 
                    alnfile=pir_filename, 
                    knowns=template_ids, 
                    sequence=self.target_id+'_pMHC_TCR',
                    )
        
        #a = automodel(env, alnfile=pir_filename,
        #      knowns=template_ids, sequence=self.target_id, #allow_alternates=True,
        #      assess_methods=(assess.DOPE))

        # For fast optimization
        #a.very_fast() 
        
        # no refinement
        #a.md_level = None

        a.starting_model = n_start
        a.ending_model = n_end
        
        a.make()
                                
        model_name = '{}_pMHC_TCR.B99990001.pdb'.format(self.target_id)
        pmhc_tcr_model = getPDB(model_name)    
        
        self.pmhc_tcr_model = pmhc_tcr_model
 
        # remove files generated by modeller 
        os.system("rm {0}*.B9999* {0}*.D0000* {0}*.V9999* {0}*.ini {0}*.rsr {0}*.sch".format(self.target_id+'_pMHC_TCR'))
        os.system("rm {}".format(pir_filename))

    def renumber_pmhc_tcr(self, pdbmodel, hmms=[]):
        
        new_model = Bio.PDB.Model.Model('')
        for chain in pdbmodel:

            if chain.id == 'M':
                mhc_chain = renumber_chain(chain, hmms)
                new_model.add(mhc_chain)

            if chain.id == 'P':
                new_model.add(chain)
            
            if chain.id == 'A' or chain.id == 'B':
                ig_chain = igm.IgChain.renumber_pdbchain(chain, hmms=hmms)
                new_model.add(ig_chain)

        self.pdbmodel_renum = new_model       

########################################################################
#####                   Build database classes                     #####
########################################################################

class BuildPmhcDatabase():
    """
    #Read an database from a csv file.
    :param filehandle csv_file: Open filehandle for database csv file.
    """

    def __init__(self, csv_file):
        self.pmhc_db = {}
        self._parse_csv(csv_file)

    def _parse_csv(self, csv_file):
        """Parse the csv file."""
        
        d = {} # format: {'pdb_name':{'peptide_seq': 'SALQNAASIA', 'mhc_seq': 'GSHSMRYFY...'}, ...}

        f = open(csv_file)
        reader = csv.DictReader(f)
        #print reader.fieldnames
        
        for row in reader:
            d[row['pdb_name']] = {  'peptide_seq': row['peptide_seq'], 
                                    'mhc_seq': row['mhc_seq'] }

        self.pmhc_db = d


class BuildComplexDatabase():
    """
    Read an database from a csv file.
    :param filehandle csv_file: Open filehandle for database csv file.
    """

    def __init__(self, csv_file):
        self.complex_db = {}
        self._parse_csv(csv_file)

    def _parse_csv(self, csv_file):
        """Parse the csv file."""
        
        d = {} # format: {'pdb_name':{'peptide_seq': 'SALQNAASIA', 'mhc_seq': 'GSHSMRYFY...','tcr_alpha_seq': '','tcr_beta_seq': ''}, ...}

        f = open(csv_file)
        reader = csv.DictReader(f)
        #print reader.fieldnames
        
        for row in reader:
            d[row['pdb_name']] = {  'peptide_seq': row['peptide_seq'], 
                                    'mhc_seq': row['mhc_seq'],
                                    'tcr_alpha_seq': row['tcr_alpha_seq'], 
                                    'tcr_beta_seq': row['tcr_beta_seq']  }

        self.complex_db = d







