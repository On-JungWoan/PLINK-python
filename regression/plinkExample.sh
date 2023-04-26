#!/bin/bash
cd ../dataset
plink --bfile data --maf 0.01 --geno 0.05 --hwe 0.000001 --make-bed --out data_snpqc
plink --bfile data_snpqc --update-sex sex_info.txt --make-bed --out data_snpqc_sex
plink --bfile data_snpqc_sex --logistic --pheno logistic_pheno.txt --mpheno 1 --covar covar.txt --covar-number 1,2 --out logistic_result
plink --bfile data_snpqc_sex --linear --pheno linear_pheno.txt --covar covar.txt --covar-number 1,2 --out linear_result
<<<<<<< HEAD
cat linear_result.assoc.linear | grep ADD | awk '{if($1 < 23)print}' | sed '/NA/d' | sed 's/ \{1,\}/\t/g' | awk '{print $1"\t"$3"\t"$2"\t"$4"\t"$5"\t"$6"\t"$7"\t"$9}' > linear_result_manhattan2.txt
cat logistic_result.assoc.logistic | grep ADD | awk '{if($1 < 23)print}' | sed '/NA/d' | sed 's/ \{1,\}/\t/g' | awk '{print $1"\t"$3"\t"$2"\t"$4"\t"$5"\t"$6"\t"$7"\t"$9}' > logistic_result_manhattan.txt
=======
cat logistic_result.assoc.logistic | grep ADD | awk '{if($1 < 23)print}' | sed '/NA/d' | sed 's/ \{1,\}/\t/g' | awk '{print $1"\t"$3"\t"$2"\t"$4"\t"$7"\t"$9}' > logistic_result_manhattan.txt
>>>>>>> 71a0bdf66d6af80b00288c00770e3d4f4da65825
