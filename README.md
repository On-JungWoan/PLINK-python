# SNP Regression
## 1. Installation

- **conda**

```
git clone https://github.com/On-JungWoan/PLINK-python.git

cd PLINK-python

conda create -n plink-env python==3.9

conda activate plink-env

pip install -r requirements.txt
```

<br>

## 2. Usage
### 2-1. Linear Regression

```
python main.py --mode linear
```

### 2-1. Logistic Regression

```
python main.py --mode Logistic
```

<br>

## 3. Arguments

!! Need to modify it !!

args | description
:--: | :--:
--path | Manually specify the project path.
--mode | Specify the regression mode(linear or logistic).
--file | Specify the file format to be used.(bim or fam).
--nosave | If this argument is used, the result will not be saved.

<details>

<summary>Examples</summary>

```
python main.py \
    --path your/prj/path \
    --mode linear (or logistic) \
    --file bim (or fam) \
    --nosave
```

</details>

.
