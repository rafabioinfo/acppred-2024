from Bio.SeqUtils import ProtParam
import pandas as pd

def compute_aa_composition(peptide:str) -> dict: 
    """
    Computes the aminoacid composition for a peptide sequence and returns a dictionary containing the percentage of each amino aicid.

    Args:

    - peptide (str): sequence of the peptide (one letter code).

    Returns:

    - aa_composition (dict): dictionary of aminoacid composition.


    """
    protein_analysis = ProtParam.ProteinAnalysis(peptide)
    return protein_analysis.get_amino_acids_percent()

def preprocess_datasets(positive_peptide_file, negative_peptide_file, output_file:str='data/processed/train.csv') -> None:
    """
    
    Preprocess the raw peptide files and produces a CSV file containing the aminoacid compositions.
    
    Args:

    - positive_peptide_file (str): file containing peptides that have anticancer activity
    - negative_peptide_file (str): file containing peptides that have no anticancer activity

    Kwargs: 

    - output_file (str): output CSV file

    Returns:

    -None

    """
    rows = []
    for line in open(positive_peptide_file):
        peptide = line.strip('\n')
        aa_composition = compute_aa_composition(peptide)
        aa_composition['activity'] = 1
        rows.append(aa_composition)

    for line in open(negative_peptide_file):
        peptide = line.strip('\n')
        aa_composition = compute_aa_composition(peptide)
        aa_composition['activity'] = 0
        rows.append(aa_composition)    


    df_processed = pd.DataFrame(rows)     
    df_processed.to_csv(output_file, index=False)

if __name__=='__main__':   
    preprocess_datasets(
    'data/raw/train_positive.txt',
    'data/raw/train_negative.txt'
)

    preprocess_datasets(
    'data/raw/test_positive.txt',
    'data/raw/test_negative.txt',
    output_file='data/processed/test.csv'
)         
