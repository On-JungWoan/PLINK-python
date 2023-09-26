from scipy.stats import chi2
def qc_process(df):
    print('[Quality control] in progress...')
    df = df.drop(df.drop(['fid','iid'],axis=1).apply(qc_maf, axis=0).dropna().to_list(), axis=1)
    df = df.drop(df.drop(['fid','iid'],axis=1).apply(qc_geno, axis=0).dropna().to_list(), axis=1)
    df = df.drop(df.drop(['fid','iid'],axis=1).apply(qc_hwe, axis=0).dropna().to_list(), axis=1)
    print(df)
    print('[Quality control] Success!')
    return df

def qc_maf(col):
    """
    Perform Minor Allele Frequency (MAF) test.
    """
    p = (col==2).sum() / len(col)
    if p > 0.99:
        return col.name

def qc_geno(col):
    """
    perform genotype missingness test.
    """
    p = col.isnull().sum()/len(col)
    if p > 0.05:
        return col.name
    
def qc_hwe(col):
    """
    Perform Hardy-Weinberg Equilibrium (HWE) test using observed genotype counts.
    
    :param O11: Observed count of individuals with two copies of the alternate allele.
    :param O12: Observed count of individuals with one copy of the alternate allele and one copy of the reference allele.
    :param O22: Observed count of individuals with two copies of the reference allele.
    
    :return: Tuple containing the HWE test statistic and its p-value.
    """

    O11, O12, O22 = (col==2).sum(), (col==1).sum(), (col==0).sum()
    N = O11 + O12 + O22
    
    E11 = ((O11*2 + O12) / (2*N))**2 * N
    E22 = ((O12 + O22*2) / (2*N))**2 * N
    E12 = 2*(O11*2 + O12) / (2*N) * (O12 + O22*2) / (2*N) * N
    HWE = ((O11-E11)**2 / E11) + ((O12-E12)**2 / E12) + ((O22-O22)**2 / E22)
    df = 1
    
    p_value = chi2.sf(HWE, df)
    if p_value < 0.000001:
        return col.name