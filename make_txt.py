import pickle

with open('not_matched_snp.pkl', 'rb') as f:
    snp = pickle.load(f)

for s in snp:
    with open('l_res.txt', 'a') as f:
        f.write(s + '\n')