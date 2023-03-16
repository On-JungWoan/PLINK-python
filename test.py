import re

head = []
for u in bim['snp'].unique():
    txt = re.findall('[a-zA-Z]', u)
    txt = ''.join(txt)
    if txt not in head:
        head.append(txt)