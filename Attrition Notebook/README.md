The first step is to import all librairies required for data vizualisation, data processing, modeling, etc. Then I have imported the dataset and displayed some key informations about it, like type of columns or values count. This conducted me to drop some columns which contains only one value.
The second step is to clean the dataset by handling duplicated rows, missing values and outliers. I haven’t found any duplicated rows or missing values, but i have detected some outliers that I removed.
Thirdly, i have selected the best features by applying chi square and mannwhitneyu tests. It remains 14 features.
After that I have encoded some categorical features with onehotencoding and spliting dataset into train set and test set.
