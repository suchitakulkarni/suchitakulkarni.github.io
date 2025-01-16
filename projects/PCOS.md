```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import *
%matplotlib inline
```

# EDA

Let us make some elementary charts and plots to understand what the data is like. 


```python
df_PCOS_all = pd.read_excel('PCOS_data_without_infertility.xlsx', sheet_name = 'Full_new')
```


```python
df_PCOS_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sl. No</th>
      <th>Patient File No.</th>
      <th>PCOS (Y/N)</th>
      <th>Age (yrs)</th>
      <th>Weight (Kg)</th>
      <th>Height(Cm)</th>
      <th>BMI</th>
      <th>Blood Group</th>
      <th>Pulse rate(bpm)</th>
      <th>RR (breaths/min)</th>
      <th>...</th>
      <th>Fast food (Y/N)</th>
      <th>Reg.Exercise(Y/N)</th>
      <th>BP _Systolic (mmHg)</th>
      <th>BP _Diastolic (mmHg)</th>
      <th>Follicle No. (L)</th>
      <th>Follicle No. (R)</th>
      <th>Avg. F size (L) (mm)</th>
      <th>Avg. F size (R) (mm)</th>
      <th>Endometrium (mm)</th>
      <th>Unnamed: 44</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>28</td>
      <td>44.6</td>
      <td>152.0</td>
      <td>19.300000</td>
      <td>15</td>
      <td>78</td>
      <td>22</td>
      <td>...</td>
      <td>1.0</td>
      <td>0</td>
      <td>110</td>
      <td>80</td>
      <td>3</td>
      <td>3</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>8.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>36</td>
      <td>65.0</td>
      <td>161.5</td>
      <td>24.921163</td>
      <td>15</td>
      <td>74</td>
      <td>20</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>120</td>
      <td>70</td>
      <td>3</td>
      <td>5</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>3.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>33</td>
      <td>68.8</td>
      <td>165.0</td>
      <td>25.270891</td>
      <td>11</td>
      <td>72</td>
      <td>18</td>
      <td>...</td>
      <td>1.0</td>
      <td>0</td>
      <td>120</td>
      <td>80</td>
      <td>13</td>
      <td>15</td>
      <td>18.0</td>
      <td>20.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>37</td>
      <td>65.0</td>
      <td>148.0</td>
      <td>29.674945</td>
      <td>13</td>
      <td>72</td>
      <td>20</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>120</td>
      <td>70</td>
      <td>2</td>
      <td>2</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>7.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>25</td>
      <td>52.0</td>
      <td>161.0</td>
      <td>20.060954</td>
      <td>11</td>
      <td>72</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>120</td>
      <td>80</td>
      <td>3</td>
      <td>4</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
df_PCOS_all.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sl. No</th>
      <th>Patient File No.</th>
      <th>PCOS (Y/N)</th>
      <th>Age (yrs)</th>
      <th>Weight (Kg)</th>
      <th>Height(Cm)</th>
      <th>BMI</th>
      <th>Blood Group</th>
      <th>Pulse rate(bpm)</th>
      <th>RR (breaths/min)</th>
      <th>...</th>
      <th>Pimples(Y/N)</th>
      <th>Fast food (Y/N)</th>
      <th>Reg.Exercise(Y/N)</th>
      <th>BP _Systolic (mmHg)</th>
      <th>BP _Diastolic (mmHg)</th>
      <th>Follicle No. (L)</th>
      <th>Follicle No. (R)</th>
      <th>Avg. F size (L) (mm)</th>
      <th>Avg. F size (R) (mm)</th>
      <th>Endometrium (mm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>...</td>
      <td>541.000000</td>
      <td>540.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>541.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>271.000000</td>
      <td>271.000000</td>
      <td>0.327172</td>
      <td>31.430684</td>
      <td>59.637153</td>
      <td>156.484835</td>
      <td>24.311285</td>
      <td>13.802218</td>
      <td>73.247689</td>
      <td>19.243993</td>
      <td>...</td>
      <td>0.489834</td>
      <td>0.514815</td>
      <td>0.247689</td>
      <td>114.661738</td>
      <td>76.927911</td>
      <td>6.129390</td>
      <td>6.641405</td>
      <td>15.018115</td>
      <td>15.451701</td>
      <td>8.475915</td>
    </tr>
    <tr>
      <th>std</th>
      <td>156.317519</td>
      <td>156.317519</td>
      <td>0.469615</td>
      <td>5.411006</td>
      <td>11.028287</td>
      <td>6.033545</td>
      <td>4.056399</td>
      <td>1.840812</td>
      <td>4.430285</td>
      <td>1.688629</td>
      <td>...</td>
      <td>0.500359</td>
      <td>0.500244</td>
      <td>0.432070</td>
      <td>7.384556</td>
      <td>5.574112</td>
      <td>4.229294</td>
      <td>4.436889</td>
      <td>3.566839</td>
      <td>3.318848</td>
      <td>2.165381</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>31.000000</td>
      <td>137.000000</td>
      <td>12.417882</td>
      <td>11.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>136.000000</td>
      <td>136.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>52.000000</td>
      <td>152.000000</td>
      <td>21.641274</td>
      <td>13.000000</td>
      <td>72.000000</td>
      <td>18.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>110.000000</td>
      <td>70.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>271.000000</td>
      <td>271.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>59.000000</td>
      <td>156.000000</td>
      <td>24.238227</td>
      <td>14.000000</td>
      <td>72.000000</td>
      <td>18.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>110.000000</td>
      <td>80.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>15.000000</td>
      <td>16.000000</td>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>406.000000</td>
      <td>406.000000</td>
      <td>1.000000</td>
      <td>35.000000</td>
      <td>65.000000</td>
      <td>160.000000</td>
      <td>26.634958</td>
      <td>15.000000</td>
      <td>74.000000</td>
      <td>20.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>80.000000</td>
      <td>9.000000</td>
      <td>10.000000</td>
      <td>18.000000</td>
      <td>18.000000</td>
      <td>9.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>541.000000</td>
      <td>541.000000</td>
      <td>1.000000</td>
      <td>48.000000</td>
      <td>108.000000</td>
      <td>180.000000</td>
      <td>38.900000</td>
      <td>18.000000</td>
      <td>82.000000</td>
      <td>28.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>140.000000</td>
      <td>100.000000</td>
      <td>22.000000</td>
      <td>20.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>18.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 42 columns</p>
</div>




```python
df_PCOS_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541 entries, 0 to 540
    Data columns (total 45 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Sl. No                  541 non-null    int64  
     1   Patient File No.        541 non-null    int64  
     2   PCOS (Y/N)              541 non-null    int64  
     3    Age (yrs)              541 non-null    int64  
     4   Weight (Kg)             541 non-null    float64
     5   Height(Cm)              541 non-null    float64
     6   BMI                     541 non-null    float64
     7   Blood Group             541 non-null    int64  
     8   Pulse rate(bpm)         541 non-null    int64  
     9   RR (breaths/min)        541 non-null    int64  
     10  Hb(g/dl)                541 non-null    float64
     11  Cycle(R/I)              541 non-null    int64  
     12  Cycle length(days)      541 non-null    int64  
     13  Marraige Status (Yrs)   540 non-null    float64
     14  Pregnant(Y/N)           541 non-null    int64  
     15  No. of aborptions       541 non-null    int64  
     16    I   beta-HCG(mIU/mL)  541 non-null    float64
     17  II    beta-HCG(mIU/mL)  541 non-null    object 
     18  FSH(mIU/mL)             541 non-null    float64
     19  LH(mIU/mL)              541 non-null    float64
     20  FSH/LH                  541 non-null    float64
     21  Hip(inch)               541 non-null    int64  
     22  Waist(inch)             541 non-null    int64  
     23  Waist:Hip Ratio         541 non-null    float64
     24  TSH (mIU/L)             541 non-null    float64
     25  AMH(ng/mL)              541 non-null    object 
     26  PRL(ng/mL)              541 non-null    float64
     27  Vit D3 (ng/mL)          541 non-null    float64
     28  PRG(ng/mL)              541 non-null    float64
     29  RBS(mg/dl)              541 non-null    float64
     30  Weight gain(Y/N)        541 non-null    int64  
     31  hair growth(Y/N)        541 non-null    int64  
     32  Skin darkening (Y/N)    541 non-null    int64  
     33  Hair loss(Y/N)          541 non-null    int64  
     34  Pimples(Y/N)            541 non-null    int64  
     35  Fast food (Y/N)         540 non-null    float64
     36  Reg.Exercise(Y/N)       541 non-null    int64  
     37  BP _Systolic (mmHg)     541 non-null    int64  
     38  BP _Diastolic (mmHg)    541 non-null    int64  
     39  Follicle No. (L)        541 non-null    int64  
     40  Follicle No. (R)        541 non-null    int64  
     41  Avg. F size (L) (mm)    541 non-null    float64
     42  Avg. F size (R) (mm)    541 non-null    float64
     43  Endometrium (mm)        541 non-null    float64
     44  Unnamed: 44             2 non-null      object 
    dtypes: float64(19), int64(23), object(3)
    memory usage: 190.3+ KB



```python
print(df_PCOS_all["Marraige Status (Yrs)"].isnull().sum())
index = df_PCOS_all.index[df_PCOS_all['AMH(ng/mL)'] == "a"]
df_PCOS_all = df_PCOS_all.drop(index)
```

Let's find all the missing values

Let us convert the Y/N columns to boolean type. 


```python
df_PCOS_all["Cycle(R/I)"] = np.where(df_PCOS_all["Cycle(R/I)"] > 2, True, False)
columns_to_convert = df_PCOS_all.filter(like = 'Y/N').columns
df_PCOS_all[columns_to_convert] = df_PCOS_all[columns_to_convert].astype('bool')
```

Let us check if it has worked. 


```python
df_PCOS_all.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 540 entries, 0 to 540
    Data columns (total 45 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Sl. No                  540 non-null    int64  
     1   Patient File No.        540 non-null    int64  
     2   PCOS (Y/N)              540 non-null    bool   
     3    Age (yrs)              540 non-null    int64  
     4   Weight (Kg)             540 non-null    float64
     5   Height(Cm)              540 non-null    float64
     6   BMI                     540 non-null    float64
     7   Blood Group             540 non-null    int64  
     8   Pulse rate(bpm)         540 non-null    int64  
     9   RR (breaths/min)        540 non-null    int64  
     10  Hb(g/dl)                540 non-null    float64
     11  Cycle(R/I)              540 non-null    bool   
     12  Cycle length(days)      540 non-null    int64  
     13  Marraige Status (Yrs)   539 non-null    float64
     14  Pregnant(Y/N)           540 non-null    bool   
     15  No. of aborptions       540 non-null    int64  
     16    I   beta-HCG(mIU/mL)  540 non-null    float64
     17  II    beta-HCG(mIU/mL)  540 non-null    object 
     18  FSH(mIU/mL)             540 non-null    float64
     19  LH(mIU/mL)              540 non-null    float64
     20  FSH/LH                  540 non-null    float64
     21  Hip(inch)               540 non-null    int64  
     22  Waist(inch)             540 non-null    int64  
     23  Waist:Hip Ratio         540 non-null    float64
     24  TSH (mIU/L)             540 non-null    float64
     25  AMH(ng/mL)              540 non-null    object 
     26  PRL(ng/mL)              540 non-null    float64
     27  Vit D3 (ng/mL)          540 non-null    float64
     28  PRG(ng/mL)              540 non-null    float64
     29  RBS(mg/dl)              540 non-null    float64
     30  Weight gain(Y/N)        540 non-null    bool   
     31  hair growth(Y/N)        540 non-null    bool   
     32  Skin darkening (Y/N)    540 non-null    bool   
     33  Hair loss(Y/N)          540 non-null    bool   
     34  Pimples(Y/N)            540 non-null    bool   
     35  Fast food (Y/N)         540 non-null    bool   
     36  Reg.Exercise(Y/N)       540 non-null    bool   
     37  BP _Systolic (mmHg)     540 non-null    int64  
     38  BP _Diastolic (mmHg)    540 non-null    int64  
     39  Follicle No. (L)        540 non-null    int64  
     40  Follicle No. (R)        540 non-null    int64  
     41  Avg. F size (L) (mm)    540 non-null    float64
     42  Avg. F size (R) (mm)    540 non-null    float64
     43  Endometrium (mm)        540 non-null    float64
     44  Unnamed: 44             2 non-null      object 
    dtypes: bool(10), float64(18), int64(14), object(3)
    memory usage: 173.3+ KB


## Data Cleaning

Apparently there is one numeric value mis-typed as 1.99. rather than 1.99 which makes it difficult for the column to be converted to float, let's correct that. 


```python
print(df_PCOS_all['II    beta-HCG(mIU/mL)'] == "1.99.")
print(df_PCOS_all.loc[123, 'II    beta-HCG(mIU/mL)'])
df_PCOS_all.loc[123, 'II    beta-HCG(mIU/mL)'] = '1.99'
#df_PCOS_all = df_PCOS_all.drop(302)
print(df_PCOS_all.loc[123, 'II    beta-HCG(mIU/mL)'])
```

    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    536    False
    537    False
    538    False
    539    False
    540    False
    Name: II    beta-HCG(mIU/mL), Length: 540, dtype: bool
    1.99.
    1.99


## First visualisations

Make some plots to understand how the data looks like


```python
columns_to_drop = ['Unnamed: 44', 'Sl. No', 'Patient File No.', 'Blood Group']
#columns_to_drop = ['Sl. No', 'Patient File No.', 'Blood Group']
df_temp  = df_PCOS_all.select_dtypes(exclude = 'bool')
df_temp = df_temp.drop(columns_to_drop, axis = 1)
#df_temp = df_temp.drop[df_temp.index[df_temp['AMH(ng/mL)'] == "a"]]

df_temp[df_temp.columns] = df_temp[df_temp.columns].astype('float64')
variables = Dropdown(options = list(df_temp.columns), description = 'Variables')

output = Output()

def make_plot(*args):
    output.clear_output(wait=True)
    with output: 
        sns.histplot(x = variables.value, data= df_temp, kde = True)
        plt.show()

variables.observe(make_plot, names = 'value')

display(variables, output)
```


    Dropdown(description='Variables', options=(' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI', 'Pulse rate(bpm)…



    Output()


## Finding correlations

Let's understand the correlations 


```python
sns.clustermap(df_temp.corr(), cmap="rocket_r")
```




    <seaborn.matrix.ClusterGrid at 0x13bf76a50>




    
![png](PCOS_files/PCOS_17_1.png)
    



```python
df_temp.corr().style.background_gradient(cmap='coolwarm')
```




<style type="text/css">
#T_a7b5f_row0_col0, #T_a7b5f_row1_col1, #T_a7b5f_row2_col2, #T_a7b5f_row3_col3, #T_a7b5f_row4_col4, #T_a7b5f_row5_col5, #T_a7b5f_row6_col6, #T_a7b5f_row7_col7, #T_a7b5f_row8_col8, #T_a7b5f_row9_col9, #T_a7b5f_row10_col10, #T_a7b5f_row11_col11, #T_a7b5f_row12_col12, #T_a7b5f_row13_col13, #T_a7b5f_row14_col14, #T_a7b5f_row15_col15, #T_a7b5f_row16_col16, #T_a7b5f_row17_col17, #T_a7b5f_row18_col18, #T_a7b5f_row19_col19, #T_a7b5f_row20_col20, #T_a7b5f_row21_col21, #T_a7b5f_row22_col22, #T_a7b5f_row23_col23, #T_a7b5f_row24_col24, #T_a7b5f_row25_col25, #T_a7b5f_row26_col26, #T_a7b5f_row27_col27, #T_a7b5f_row28_col28, #T_a7b5f_row29_col29, #T_a7b5f_row30_col30, #T_a7b5f_row31_col31 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_a7b5f_row0_col1, #T_a7b5f_row1_col30, #T_a7b5f_row2_col14, #T_a7b5f_row3_col15, #T_a7b5f_row4_col2, #T_a7b5f_row4_col29, #T_a7b5f_row6_col5, #T_a7b5f_row6_col17, #T_a7b5f_row7_col23, #T_a7b5f_row11_col19, #T_a7b5f_row12_col5, #T_a7b5f_row12_col24, #T_a7b5f_row14_col1, #T_a7b5f_row17_col21, #T_a7b5f_row23_col24, #T_a7b5f_row24_col26, #T_a7b5f_row24_col29, #T_a7b5f_row30_col9 {
  background-color: #465ecf;
  color: #f1f1f1;
}
#T_a7b5f_row0_col2, #T_a7b5f_row0_col20, #T_a7b5f_row0_col27, #T_a7b5f_row0_col28, #T_a7b5f_row0_col31, #T_a7b5f_row2_col4, #T_a7b5f_row5_col11, #T_a7b5f_row5_col23, #T_a7b5f_row6_col12, #T_a7b5f_row6_col13, #T_a7b5f_row6_col14, #T_a7b5f_row7_col8, #T_a7b5f_row8_col7, #T_a7b5f_row8_col22, #T_a7b5f_row8_col26, #T_a7b5f_row8_col28, #T_a7b5f_row9_col31, #T_a7b5f_row10_col25, #T_a7b5f_row10_col30, #T_a7b5f_row11_col5, #T_a7b5f_row11_col25, #T_a7b5f_row12_col6, #T_a7b5f_row16_col18, #T_a7b5f_row16_col21, #T_a7b5f_row18_col16, #T_a7b5f_row19_col29, #T_a7b5f_row20_col0, #T_a7b5f_row20_col9, #T_a7b5f_row20_col17, #T_a7b5f_row25_col11, #T_a7b5f_row29_col19, #T_a7b5f_row29_col24, #T_a7b5f_row30_col1, #T_a7b5f_row30_col3, #T_a7b5f_row30_col10, #T_a7b5f_row30_col19, #T_a7b5f_row31_col13, #T_a7b5f_row31_col15 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_a7b5f_row0_col3, #T_a7b5f_row1_col5, #T_a7b5f_row2_col12, #T_a7b5f_row2_col23, #T_a7b5f_row3_col23, #T_a7b5f_row5_col26, #T_a7b5f_row7_col6, #T_a7b5f_row7_col30, #T_a7b5f_row10_col7, #T_a7b5f_row14_col21, #T_a7b5f_row16_col23, #T_a7b5f_row16_col31, #T_a7b5f_row21_col31, #T_a7b5f_row22_col8, #T_a7b5f_row22_col10, #T_a7b5f_row26_col29, #T_a7b5f_row28_col12, #T_a7b5f_row30_col12 {
  background-color: #5f7fe8;
  color: #f1f1f1;
}
#T_a7b5f_row0_col4, #T_a7b5f_row4_col20, #T_a7b5f_row6_col29, #T_a7b5f_row7_col12, #T_a7b5f_row10_col4, #T_a7b5f_row10_col17, #T_a7b5f_row10_col19, #T_a7b5f_row12_col31, #T_a7b5f_row15_col12, #T_a7b5f_row17_col29, #T_a7b5f_row18_col26, #T_a7b5f_row21_col0, #T_a7b5f_row27_col25, #T_a7b5f_row30_col14, #T_a7b5f_row30_col16, #T_a7b5f_row30_col25 {
  background-color: #5d7ce6;
  color: #f1f1f1;
}
#T_a7b5f_row0_col5, #T_a7b5f_row2_col30, #T_a7b5f_row6_col10, #T_a7b5f_row11_col0, #T_a7b5f_row11_col10, #T_a7b5f_row12_col27, #T_a7b5f_row14_col9, #T_a7b5f_row18_col9, #T_a7b5f_row18_col25, #T_a7b5f_row22_col20, #T_a7b5f_row24_col20, #T_a7b5f_row25_col8, #T_a7b5f_row30_col2 {
  background-color: #6c8ff1;
  color: #f1f1f1;
}
#T_a7b5f_row0_col6, #T_a7b5f_row0_col15, #T_a7b5f_row1_col22, #T_a7b5f_row1_col29, #T_a7b5f_row6_col11, #T_a7b5f_row8_col1, #T_a7b5f_row8_col25, #T_a7b5f_row9_col28, #T_a7b5f_row10_col14, #T_a7b5f_row11_col21, #T_a7b5f_row12_col13, #T_a7b5f_row13_col3, #T_a7b5f_row13_col24, #T_a7b5f_row14_col2, #T_a7b5f_row15_col1, #T_a7b5f_row16_col6, #T_a7b5f_row16_col15, #T_a7b5f_row16_col22, #T_a7b5f_row16_col25, #T_a7b5f_row20_col2, #T_a7b5f_row21_col11, #T_a7b5f_row21_col13, #T_a7b5f_row21_col15, #T_a7b5f_row22_col23, #T_a7b5f_row23_col19, #T_a7b5f_row24_col13, #T_a7b5f_row27_col21, #T_a7b5f_row29_col9, #T_a7b5f_row29_col15, #T_a7b5f_row30_col13, #T_a7b5f_row31_col0, #T_a7b5f_row31_col30 {
  background-color: #4e68d8;
  color: #f1f1f1;
}
#T_a7b5f_row0_col7, #T_a7b5f_row1_col6, #T_a7b5f_row2_col3, #T_a7b5f_row4_col21, #T_a7b5f_row7_col5, #T_a7b5f_row8_col3, #T_a7b5f_row8_col11, #T_a7b5f_row9_col17, #T_a7b5f_row9_col22, #T_a7b5f_row13_col27, #T_a7b5f_row13_col29, #T_a7b5f_row15_col26, #T_a7b5f_row15_col29, #T_a7b5f_row16_col20, #T_a7b5f_row20_col1, #T_a7b5f_row20_col26, #T_a7b5f_row22_col30, #T_a7b5f_row25_col1, #T_a7b5f_row25_col5, #T_a7b5f_row25_col22, #T_a7b5f_row25_col24, #T_a7b5f_row27_col8, #T_a7b5f_row28_col11, #T_a7b5f_row31_col19 {
  background-color: #5875e1;
  color: #f1f1f1;
}
#T_a7b5f_row0_col8, #T_a7b5f_row3_col18, #T_a7b5f_row24_col18, #T_a7b5f_row25_col0, #T_a7b5f_row25_col7, #T_a7b5f_row26_col0 {
  background-color: #80a3fa;
  color: #f1f1f1;
}
#T_a7b5f_row0_col9, #T_a7b5f_row1_col16 {
  background-color: #f7aa8c;
  color: #000000;
}
#T_a7b5f_row0_col10, #T_a7b5f_row20_col28, #T_a7b5f_row30_col28 {
  background-color: #9fbfff;
  color: #000000;
}
#T_a7b5f_row0_col11, #T_a7b5f_row1_col4, #T_a7b5f_row3_col6, #T_a7b5f_row4_col1, #T_a7b5f_row7_col2, #T_a7b5f_row12_col3, #T_a7b5f_row13_col10, #T_a7b5f_row14_col3, #T_a7b5f_row14_col10, #T_a7b5f_row15_col3, #T_a7b5f_row18_col12, #T_a7b5f_row20_col12, #T_a7b5f_row21_col2, #T_a7b5f_row21_col4, #T_a7b5f_row21_col27, #T_a7b5f_row22_col29, #T_a7b5f_row26_col6, #T_a7b5f_row27_col23, #T_a7b5f_row27_col24, #T_a7b5f_row29_col12 {
  background-color: #5572df;
  color: #f1f1f1;
}
#T_a7b5f_row0_col12, #T_a7b5f_row1_col0, #T_a7b5f_row6_col2, #T_a7b5f_row6_col25, #T_a7b5f_row6_col30, #T_a7b5f_row7_col25, #T_a7b5f_row15_col2, #T_a7b5f_row15_col9, #T_a7b5f_row15_col30, #T_a7b5f_row16_col4, #T_a7b5f_row19_col18, #T_a7b5f_row21_col16, #T_a7b5f_row24_col5, #T_a7b5f_row24_col25, #T_a7b5f_row25_col19, #T_a7b5f_row26_col5, #T_a7b5f_row26_col30, #T_a7b5f_row27_col22, #T_a7b5f_row29_col16 {
  background-color: #6282ea;
  color: #f1f1f1;
}
#T_a7b5f_row0_col13, #T_a7b5f_row1_col21, #T_a7b5f_row9_col20, #T_a7b5f_row9_col27, #T_a7b5f_row10_col13, #T_a7b5f_row14_col24, #T_a7b5f_row21_col6, #T_a7b5f_row25_col10, #T_a7b5f_row31_col4, #T_a7b5f_row31_col11 {
  background-color: #4358cb;
  color: #f1f1f1;
}
#T_a7b5f_row0_col14, #T_a7b5f_row1_col12, #T_a7b5f_row1_col31, #T_a7b5f_row4_col24, #T_a7b5f_row5_col21, #T_a7b5f_row5_col30, #T_a7b5f_row8_col5, #T_a7b5f_row10_col2, #T_a7b5f_row11_col1, #T_a7b5f_row12_col29, #T_a7b5f_row15_col11, #T_a7b5f_row17_col6, #T_a7b5f_row17_col11, #T_a7b5f_row18_col1, #T_a7b5f_row21_col5, #T_a7b5f_row21_col12, #T_a7b5f_row23_col10, #T_a7b5f_row28_col14, #T_a7b5f_row29_col2, #T_a7b5f_row29_col31, #T_a7b5f_row30_col0 {
  background-color: #5470de;
  color: #f1f1f1;
}
#T_a7b5f_row0_col16, #T_a7b5f_row1_col26, #T_a7b5f_row2_col27, #T_a7b5f_row7_col18, #T_a7b5f_row10_col18, #T_a7b5f_row13_col8, #T_a7b5f_row13_col16, #T_a7b5f_row14_col18, #T_a7b5f_row24_col9, #T_a7b5f_row24_col16, #T_a7b5f_row28_col3, #T_a7b5f_row28_col18 {
  background-color: #799cf8;
  color: #f1f1f1;
}
#T_a7b5f_row0_col17, #T_a7b5f_row2_col13, #T_a7b5f_row4_col17, #T_a7b5f_row4_col26, #T_a7b5f_row4_col30, #T_a7b5f_row5_col17, #T_a7b5f_row6_col1, #T_a7b5f_row8_col4, #T_a7b5f_row8_col13, #T_a7b5f_row8_col15, #T_a7b5f_row8_col23, #T_a7b5f_row9_col26, #T_a7b5f_row11_col14, #T_a7b5f_row12_col14, #T_a7b5f_row12_col15, #T_a7b5f_row12_col21, #T_a7b5f_row12_col23, #T_a7b5f_row14_col19, #T_a7b5f_row15_col14, #T_a7b5f_row17_col15, #T_a7b5f_row17_col19, #T_a7b5f_row22_col1, #T_a7b5f_row22_col2, #T_a7b5f_row22_col19, #T_a7b5f_row23_col11, #T_a7b5f_row23_col14, #T_a7b5f_row25_col17, #T_a7b5f_row26_col4, #T_a7b5f_row26_col15 {
  background-color: #516ddb;
  color: #f1f1f1;
}
#T_a7b5f_row0_col18, #T_a7b5f_row22_col7 {
  background-color: #8caffe;
  color: #000000;
}
#T_a7b5f_row0_col19, #T_a7b5f_row1_col11, #T_a7b5f_row2_col24, #T_a7b5f_row5_col24, #T_a7b5f_row5_col25, #T_a7b5f_row8_col18, #T_a7b5f_row8_col20, #T_a7b5f_row8_col21, #T_a7b5f_row8_col30, #T_a7b5f_row9_col6, #T_a7b5f_row11_col3, #T_a7b5f_row13_col21, #T_a7b5f_row13_col26, #T_a7b5f_row14_col26, #T_a7b5f_row15_col21, #T_a7b5f_row16_col11, #T_a7b5f_row17_col30, #T_a7b5f_row19_col9, #T_a7b5f_row20_col11, #T_a7b5f_row21_col20, #T_a7b5f_row23_col12, #T_a7b5f_row24_col10, #T_a7b5f_row24_col22, #T_a7b5f_row26_col12, #T_a7b5f_row27_col26, #T_a7b5f_row28_col5, #T_a7b5f_row29_col3, #T_a7b5f_row30_col26, #T_a7b5f_row31_col14 {
  background-color: #5673e0;
  color: #f1f1f1;
}
#T_a7b5f_row0_col21, #T_a7b5f_row0_col30, #T_a7b5f_row3_col21, #T_a7b5f_row4_col13, #T_a7b5f_row10_col24, #T_a7b5f_row10_col29, #T_a7b5f_row10_col31, #T_a7b5f_row18_col23, #T_a7b5f_row19_col17, #T_a7b5f_row20_col13, #T_a7b5f_row25_col12, #T_a7b5f_row25_col15, #T_a7b5f_row26_col22, #T_a7b5f_row28_col8, #T_a7b5f_row28_col10, #T_a7b5f_row29_col23, #T_a7b5f_row30_col23 {
  background-color: #445acc;
  color: #f1f1f1;
}
#T_a7b5f_row0_col22, #T_a7b5f_row2_col21, #T_a7b5f_row2_col29, #T_a7b5f_row4_col11, #T_a7b5f_row4_col31, #T_a7b5f_row6_col19, #T_a7b5f_row7_col14, #T_a7b5f_row13_col19, #T_a7b5f_row15_col25, #T_a7b5f_row16_col14, #T_a7b5f_row17_col13, #T_a7b5f_row17_col14, #T_a7b5f_row18_col8, #T_a7b5f_row19_col6, #T_a7b5f_row20_col5, #T_a7b5f_row20_col22, #T_a7b5f_row20_col23, #T_a7b5f_row21_col23, #T_a7b5f_row22_col17, #T_a7b5f_row27_col0, #T_a7b5f_row27_col9, #T_a7b5f_row29_col22, #T_a7b5f_row31_col25 {
  background-color: #4c66d6;
  color: #f1f1f1;
}
#T_a7b5f_row0_col23, #T_a7b5f_row1_col14, #T_a7b5f_row2_col0, #T_a7b5f_row2_col5, #T_a7b5f_row4_col14, #T_a7b5f_row4_col22, #T_a7b5f_row4_col25, #T_a7b5f_row5_col14, #T_a7b5f_row5_col29, #T_a7b5f_row7_col10, #T_a7b5f_row9_col21, #T_a7b5f_row12_col4, #T_a7b5f_row13_col17, #T_a7b5f_row13_col22, #T_a7b5f_row13_col25, #T_a7b5f_row13_col31, #T_a7b5f_row14_col22, #T_a7b5f_row14_col25, #T_a7b5f_row15_col22, #T_a7b5f_row19_col12, #T_a7b5f_row19_col23, #T_a7b5f_row20_col24, #T_a7b5f_row23_col4, #T_a7b5f_row23_col31, #T_a7b5f_row24_col17, #T_a7b5f_row25_col23, #T_a7b5f_row26_col21, #T_a7b5f_row27_col10, #T_a7b5f_row28_col9, #T_a7b5f_row29_col10, #T_a7b5f_row30_col31, #T_a7b5f_row31_col17 {
  background-color: #4a63d3;
  color: #f1f1f1;
}
#T_a7b5f_row0_col24, #T_a7b5f_row5_col20, #T_a7b5f_row9_col7, #T_a7b5f_row12_col2, #T_a7b5f_row12_col30, #T_a7b5f_row13_col0, #T_a7b5f_row13_col20, #T_a7b5f_row18_col6, #T_a7b5f_row18_col28, #T_a7b5f_row20_col6, #T_a7b5f_row23_col7, #T_a7b5f_row24_col27, #T_a7b5f_row25_col30, #T_a7b5f_row29_col0 {
  background-color: #6687ed;
  color: #f1f1f1;
}
#T_a7b5f_row0_col25, #T_a7b5f_row3_col25, #T_a7b5f_row10_col6, #T_a7b5f_row11_col27, #T_a7b5f_row14_col28, #T_a7b5f_row19_col20, #T_a7b5f_row20_col16, #T_a7b5f_row23_col20, #T_a7b5f_row27_col5, #T_a7b5f_row27_col6, #T_a7b5f_row28_col22, #T_a7b5f_row30_col21 {
  background-color: #6788ee;
  color: #f1f1f1;
}
#T_a7b5f_row0_col26, #T_a7b5f_row1_col23, #T_a7b5f_row2_col11, #T_a7b5f_row3_col5, #T_a7b5f_row3_col24, #T_a7b5f_row4_col27, #T_a7b5f_row5_col18, #T_a7b5f_row6_col0, #T_a7b5f_row6_col9, #T_a7b5f_row7_col11, #T_a7b5f_row7_col31, #T_a7b5f_row10_col26, #T_a7b5f_row13_col2, #T_a7b5f_row14_col30, #T_a7b5f_row19_col1, #T_a7b5f_row19_col10, #T_a7b5f_row23_col0, #T_a7b5f_row23_col1, #T_a7b5f_row25_col6, #T_a7b5f_row25_col27, #T_a7b5f_row26_col19 {
  background-color: #6485ec;
  color: #f1f1f1;
}
#T_a7b5f_row0_col29, #T_a7b5f_row2_col9, #T_a7b5f_row3_col11, #T_a7b5f_row3_col14, #T_a7b5f_row6_col24, #T_a7b5f_row7_col19, #T_a7b5f_row10_col22, #T_a7b5f_row11_col23, #T_a7b5f_row12_col1, #T_a7b5f_row13_col23, #T_a7b5f_row14_col23, #T_a7b5f_row15_col17, #T_a7b5f_row15_col23, #T_a7b5f_row18_col24, #T_a7b5f_row19_col5, #T_a7b5f_row21_col25, #T_a7b5f_row22_col4, #T_a7b5f_row22_col5, #T_a7b5f_row22_col11, #T_a7b5f_row22_col21, #T_a7b5f_row22_col31, #T_a7b5f_row23_col17, #T_a7b5f_row23_col21, #T_a7b5f_row23_col30, #T_a7b5f_row24_col14, #T_a7b5f_row24_col23, #T_a7b5f_row25_col21, #T_a7b5f_row26_col13, #T_a7b5f_row28_col19, #T_a7b5f_row28_col21 {
  background-color: #4f69d9;
  color: #f1f1f1;
}
#T_a7b5f_row1_col2 {
  background-color: #d8dce2;
  color: #000000;
}
#T_a7b5f_row1_col3 {
  background-color: #d24b40;
  color: #f1f1f1;
}
#T_a7b5f_row1_col7, #T_a7b5f_row7_col16 {
  background-color: #aac7fd;
  color: #000000;
}
#T_a7b5f_row1_col8, #T_a7b5f_row1_col9, #T_a7b5f_row11_col20, #T_a7b5f_row14_col8, #T_a7b5f_row19_col3, #T_a7b5f_row19_col8, #T_a7b5f_row22_col27, #T_a7b5f_row23_col28, #T_a7b5f_row24_col8, #T_a7b5f_row25_col26, #T_a7b5f_row27_col17, #T_a7b5f_row27_col31, #T_a7b5f_row28_col31 {
  background-color: #6f92f3;
  color: #f1f1f1;
}
#T_a7b5f_row1_col10, #T_a7b5f_row2_col18, #T_a7b5f_row4_col0, #T_a7b5f_row9_col12, #T_a7b5f_row12_col0, #T_a7b5f_row12_col7, #T_a7b5f_row15_col8, #T_a7b5f_row20_col30, #T_a7b5f_row20_col31, #T_a7b5f_row26_col1 {
  background-color: #779af7;
  color: #f1f1f1;
}
#T_a7b5f_row1_col13, #T_a7b5f_row2_col22, #T_a7b5f_row6_col4, #T_a7b5f_row7_col13, #T_a7b5f_row8_col27, #T_a7b5f_row9_col23, #T_a7b5f_row11_col24, #T_a7b5f_row12_col25, #T_a7b5f_row14_col17, #T_a7b5f_row18_col4, #T_a7b5f_row19_col4, #T_a7b5f_row19_col13, #T_a7b5f_row20_col4, #T_a7b5f_row21_col1, #T_a7b5f_row25_col13, #T_a7b5f_row28_col13, #T_a7b5f_row31_col5 {
  background-color: #4055c8;
  color: #f1f1f1;
}
#T_a7b5f_row1_col15, #T_a7b5f_row2_col31, #T_a7b5f_row5_col22, #T_a7b5f_row8_col24, #T_a7b5f_row9_col2, #T_a7b5f_row9_col19, #T_a7b5f_row10_col27, #T_a7b5f_row11_col17, #T_a7b5f_row11_col22, #T_a7b5f_row13_col1, #T_a7b5f_row13_col6, #T_a7b5f_row14_col13, #T_a7b5f_row14_col15, #T_a7b5f_row16_col13, #T_a7b5f_row16_col24, #T_a7b5f_row19_col22, #T_a7b5f_row20_col15, #T_a7b5f_row21_col26, #T_a7b5f_row22_col13, #T_a7b5f_row23_col13, #T_a7b5f_row23_col15, #T_a7b5f_row23_col29, #T_a7b5f_row25_col2, #T_a7b5f_row25_col4, #T_a7b5f_row27_col13, #T_a7b5f_row28_col15, #T_a7b5f_row30_col22, #T_a7b5f_row31_col10 {
  background-color: #485fd1;
  color: #f1f1f1;
}
#T_a7b5f_row1_col17 {
  background-color: #f7bca1;
  color: #000000;
}
#T_a7b5f_row1_col18, #T_a7b5f_row2_col28, #T_a7b5f_row3_col10, #T_a7b5f_row7_col17, #T_a7b5f_row11_col16, #T_a7b5f_row15_col16, #T_a7b5f_row16_col27, #T_a7b5f_row20_col29 {
  background-color: #7da0f9;
  color: #f1f1f1;
}
#T_a7b5f_row1_col19, #T_a7b5f_row2_col7, #T_a7b5f_row3_col19, #T_a7b5f_row4_col18, #T_a7b5f_row5_col28, #T_a7b5f_row9_col5, #T_a7b5f_row14_col7, #T_a7b5f_row16_col0, #T_a7b5f_row20_col3, #T_a7b5f_row23_col2, #T_a7b5f_row24_col2, #T_a7b5f_row25_col20 {
  background-color: #6a8bef;
  color: #f1f1f1;
}
#T_a7b5f_row1_col20, #T_a7b5f_row9_col3, #T_a7b5f_row11_col18, #T_a7b5f_row14_col16, #T_a7b5f_row17_col9 {
  background-color: #7396f5;
  color: #f1f1f1;
}
#T_a7b5f_row1_col24, #T_a7b5f_row5_col3, #T_a7b5f_row6_col27, #T_a7b5f_row10_col1, #T_a7b5f_row11_col28, #T_a7b5f_row12_col20, #T_a7b5f_row14_col0, #T_a7b5f_row15_col7, #T_a7b5f_row19_col7, #T_a7b5f_row22_col0, #T_a7b5f_row25_col9, #T_a7b5f_row28_col6, #T_a7b5f_row29_col21, #T_a7b5f_row30_col8, #T_a7b5f_row31_col8 {
  background-color: #6b8df0;
  color: #f1f1f1;
}
#T_a7b5f_row1_col25, #T_a7b5f_row3_col31, #T_a7b5f_row6_col3, #T_a7b5f_row8_col12, #T_a7b5f_row9_col4, #T_a7b5f_row9_col25, #T_a7b5f_row13_col12, #T_a7b5f_row15_col27, #T_a7b5f_row17_col4, #T_a7b5f_row18_col10, #T_a7b5f_row21_col9, #T_a7b5f_row26_col8, #T_a7b5f_row26_col14, #T_a7b5f_row29_col26, #T_a7b5f_row31_col12 {
  background-color: #5a78e4;
  color: #f1f1f1;
}
#T_a7b5f_row1_col27, #T_a7b5f_row5_col16, #T_a7b5f_row7_col1 {
  background-color: #8fb1fe;
  color: #000000;
}
#T_a7b5f_row1_col28, #T_a7b5f_row4_col16 {
  background-color: #8badfd;
  color: #000000;
}
#T_a7b5f_row2_col1 {
  background-color: #d2dbe8;
  color: #000000;
}
#T_a7b5f_row2_col6, #T_a7b5f_row5_col1, #T_a7b5f_row8_col10, #T_a7b5f_row9_col1, #T_a7b5f_row10_col20, #T_a7b5f_row13_col9, #T_a7b5f_row14_col31, #T_a7b5f_row17_col25, #T_a7b5f_row17_col31, #T_a7b5f_row19_col31, #T_a7b5f_row20_col8, #T_a7b5f_row21_col19, #T_a7b5f_row24_col4, #T_a7b5f_row24_col6, #T_a7b5f_row27_col4, #T_a7b5f_row28_col26, #T_a7b5f_row30_col6, #T_a7b5f_row31_col3 {
  background-color: #5b7ae5;
  color: #f1f1f1;
}
#T_a7b5f_row2_col8, #T_a7b5f_row4_col8, #T_a7b5f_row6_col16, #T_a7b5f_row16_col10, #T_a7b5f_row23_col8, #T_a7b5f_row24_col1, #T_a7b5f_row26_col20, #T_a7b5f_row28_col2, #T_a7b5f_row31_col18 {
  background-color: #7295f4;
  color: #f1f1f1;
}
#T_a7b5f_row2_col10, #T_a7b5f_row2_col26, #T_a7b5f_row3_col22, #T_a7b5f_row5_col2, #T_a7b5f_row7_col0, #T_a7b5f_row7_col21, #T_a7b5f_row8_col14, #T_a7b5f_row8_col19, #T_a7b5f_row8_col31, #T_a7b5f_row13_col11, #T_a7b5f_row13_col14, #T_a7b5f_row14_col12, #T_a7b5f_row15_col10, #T_a7b5f_row18_col14, #T_a7b5f_row18_col15, #T_a7b5f_row22_col12, #T_a7b5f_row22_col14, #T_a7b5f_row22_col24, #T_a7b5f_row23_col9, #T_a7b5f_row24_col12, #T_a7b5f_row24_col31, #T_a7b5f_row25_col31, #T_a7b5f_row26_col11, #T_a7b5f_row26_col31, #T_a7b5f_row27_col14 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_a7b5f_row2_col15, #T_a7b5f_row3_col12, #T_a7b5f_row3_col29, #T_a7b5f_row4_col12, #T_a7b5f_row5_col19, #T_a7b5f_row6_col26, #T_a7b5f_row10_col5, #T_a7b5f_row10_col28, #T_a7b5f_row11_col6, #T_a7b5f_row11_col26, #T_a7b5f_row12_col26, #T_a7b5f_row14_col11, #T_a7b5f_row17_col22, #T_a7b5f_row18_col13, #T_a7b5f_row18_col22, #T_a7b5f_row18_col31, #T_a7b5f_row19_col14, #T_a7b5f_row19_col27, #T_a7b5f_row20_col19, #T_a7b5f_row20_col25, #T_a7b5f_row28_col24, #T_a7b5f_row29_col17, #T_a7b5f_row30_col15, #T_a7b5f_row31_col29 {
  background-color: #506bda;
  color: #f1f1f1;
}
#T_a7b5f_row2_col16 {
  background-color: #b6cefa;
  color: #000000;
}
#T_a7b5f_row2_col17, #T_a7b5f_row6_col18, #T_a7b5f_row20_col18, #T_a7b5f_row27_col30, #T_a7b5f_row31_col20 {
  background-color: #89acfd;
  color: #000000;
}
#T_a7b5f_row2_col19, #T_a7b5f_row3_col2, #T_a7b5f_row5_col10, #T_a7b5f_row7_col9, #T_a7b5f_row14_col27, #T_a7b5f_row16_col12, #T_a7b5f_row17_col12, #T_a7b5f_row17_col23, #T_a7b5f_row18_col2, #T_a7b5f_row19_col21, #T_a7b5f_row20_col14, #T_a7b5f_row23_col26, #T_a7b5f_row26_col23, #T_a7b5f_row28_col23, #T_a7b5f_row28_col25, #T_a7b5f_row31_col21 {
  background-color: #5977e3;
  color: #f1f1f1;
}
#T_a7b5f_row2_col20, #T_a7b5f_row3_col4, #T_a7b5f_row6_col22, #T_a7b5f_row7_col29, #T_a7b5f_row8_col2, #T_a7b5f_row9_col14, #T_a7b5f_row9_col24, #T_a7b5f_row13_col28, #T_a7b5f_row14_col29, #T_a7b5f_row16_col19, #T_a7b5f_row17_col5, #T_a7b5f_row17_col20, #T_a7b5f_row22_col25, #T_a7b5f_row23_col27, #T_a7b5f_row24_col30, #T_a7b5f_row26_col2, #T_a7b5f_row28_col4, #T_a7b5f_row29_col6, #T_a7b5f_row29_col14 {
  background-color: #5e7de7;
  color: #f1f1f1;
}
#T_a7b5f_row2_col25, #T_a7b5f_row6_col15, #T_a7b5f_row16_col30, #T_a7b5f_row26_col24 {
  background-color: #3e51c5;
  color: #f1f1f1;
}
#T_a7b5f_row3_col0, #T_a7b5f_row5_col8, #T_a7b5f_row10_col8, #T_a7b5f_row11_col30, #T_a7b5f_row12_col28, #T_a7b5f_row14_col20, #T_a7b5f_row16_col26, #T_a7b5f_row17_col10, #T_a7b5f_row21_col7, #T_a7b5f_row21_col30, #T_a7b5f_row26_col10, #T_a7b5f_row26_col25, #T_a7b5f_row26_col28, #T_a7b5f_row31_col27 {
  background-color: #7093f3;
  color: #f1f1f1;
}
#T_a7b5f_row3_col1 {
  background-color: #d44e41;
  color: #f1f1f1;
}
#T_a7b5f_row3_col7 {
  background-color: #b3cdfb;
  color: #000000;
}
#T_a7b5f_row3_col8, #T_a7b5f_row4_col9, #T_a7b5f_row5_col27, #T_a7b5f_row7_col4, #T_a7b5f_row11_col2, #T_a7b5f_row15_col0, #T_a7b5f_row16_col9, #T_a7b5f_row18_col7, #T_a7b5f_row18_col21, #T_a7b5f_row19_col0, #T_a7b5f_row21_col29, #T_a7b5f_row22_col9, #T_a7b5f_row23_col18, #T_a7b5f_row24_col7, #T_a7b5f_row24_col28, #T_a7b5f_row25_col3, #T_a7b5f_row25_col28, #T_a7b5f_row26_col17 {
  background-color: #6e90f2;
  color: #f1f1f1;
}
#T_a7b5f_row3_col9, #T_a7b5f_row3_col26, #T_a7b5f_row6_col20, #T_a7b5f_row6_col28, #T_a7b5f_row8_col0, #T_a7b5f_row16_col8, #T_a7b5f_row18_col20, #T_a7b5f_row22_col16, #T_a7b5f_row27_col2, #T_a7b5f_row31_col7 {
  background-color: #7b9ff9;
  color: #f1f1f1;
}
#T_a7b5f_row3_col13, #T_a7b5f_row5_col15, #T_a7b5f_row16_col29, #T_a7b5f_row18_col5, #T_a7b5f_row20_col21 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_a7b5f_row3_col16 {
  background-color: #f7b599;
  color: #000000;
}
#T_a7b5f_row3_col17 {
  background-color: #f4c5ad;
  color: #000000;
}
#T_a7b5f_row3_col20, #T_a7b5f_row5_col9, #T_a7b5f_row6_col7, #T_a7b5f_row10_col3, #T_a7b5f_row12_col18, #T_a7b5f_row29_col7 {
  background-color: #7a9df8;
  color: #f1f1f1;
}
#T_a7b5f_row3_col27, #T_a7b5f_row3_col28, #T_a7b5f_row8_col16, #T_a7b5f_row26_col18, #T_a7b5f_row27_col1, #T_a7b5f_row27_col3, #T_a7b5f_row28_col29 {
  background-color: #85a8fc;
  color: #f1f1f1;
}
#T_a7b5f_row3_col30, #T_a7b5f_row14_col6, #T_a7b5f_row18_col19, #T_a7b5f_row21_col17, #T_a7b5f_row21_col24, #T_a7b5f_row23_col5 {
  background-color: #3c4ec2;
  color: #f1f1f1;
}
#T_a7b5f_row4_col3, #T_a7b5f_row4_col10, #T_a7b5f_row7_col22, #T_a7b5f_row7_col26, #T_a7b5f_row12_col10, #T_a7b5f_row13_col7, #T_a7b5f_row15_col20, #T_a7b5f_row16_col5, #T_a7b5f_row17_col8, #T_a7b5f_row22_col6, #T_a7b5f_row23_col3, #T_a7b5f_row23_col6, #T_a7b5f_row27_col12, #T_a7b5f_row30_col11 {
  background-color: #688aef;
  color: #f1f1f1;
}
#T_a7b5f_row4_col5 {
  background-color: #b2ccfb;
  color: #000000;
}
#T_a7b5f_row4_col6, #T_a7b5f_row4_col15, #T_a7b5f_row4_col19, #T_a7b5f_row5_col31, #T_a7b5f_row6_col31, #T_a7b5f_row7_col15, #T_a7b5f_row7_col24, #T_a7b5f_row8_col6, #T_a7b5f_row8_col29, #T_a7b5f_row14_col4, #T_a7b5f_row15_col5, #T_a7b5f_row19_col11, #T_a7b5f_row24_col21, #T_a7b5f_row30_col4, #T_a7b5f_row31_col9 {
  background-color: #455cce;
  color: #f1f1f1;
}
#T_a7b5f_row4_col7 {
  background-color: #8db0fe;
  color: #000000;
}
#T_a7b5f_row4_col23, #T_a7b5f_row11_col15, #T_a7b5f_row12_col17, #T_a7b5f_row13_col4, #T_a7b5f_row15_col4, #T_a7b5f_row15_col19, #T_a7b5f_row17_col24, #T_a7b5f_row18_col11, #T_a7b5f_row20_col10, #T_a7b5f_row21_col3, #T_a7b5f_row23_col25, #T_a7b5f_row24_col11, #T_a7b5f_row24_col15, #T_a7b5f_row24_col19, #T_a7b5f_row25_col14, #T_a7b5f_row27_col15, #T_a7b5f_row27_col19, #T_a7b5f_row29_col13, #T_a7b5f_row30_col5, #T_a7b5f_row30_col24, #T_a7b5f_row31_col1, #T_a7b5f_row31_col2, #T_a7b5f_row31_col26 {
  background-color: #4b64d5;
  color: #f1f1f1;
}
#T_a7b5f_row4_col28, #T_a7b5f_row5_col7, #T_a7b5f_row9_col11, #T_a7b5f_row12_col8, #T_a7b5f_row17_col0, #T_a7b5f_row21_col8, #T_a7b5f_row30_col7 {
  background-color: #7597f6;
  color: #f1f1f1;
}
#T_a7b5f_row5_col0, #T_a7b5f_row9_col16, #T_a7b5f_row11_col9, #T_a7b5f_row12_col9, #T_a7b5f_row19_col16, #T_a7b5f_row26_col3 {
  background-color: #84a7fc;
  color: #f1f1f1;
}
#T_a7b5f_row5_col4, #T_a7b5f_row7_col28 {
  background-color: #b1cbfc;
  color: #000000;
}
#T_a7b5f_row5_col6, #T_a7b5f_row5_col12, #T_a7b5f_row9_col15, #T_a7b5f_row10_col23, #T_a7b5f_row11_col4, #T_a7b5f_row11_col13, #T_a7b5f_row11_col31, #T_a7b5f_row12_col19, #T_a7b5f_row12_col22, #T_a7b5f_row13_col5, #T_a7b5f_row14_col5, #T_a7b5f_row15_col6, #T_a7b5f_row15_col24, #T_a7b5f_row15_col31, #T_a7b5f_row21_col10, #T_a7b5f_row21_col22, #T_a7b5f_row22_col15, #T_a7b5f_row22_col26, #T_a7b5f_row23_col22, #T_a7b5f_row29_col1, #T_a7b5f_row29_col5 {
  background-color: #4961d2;
  color: #f1f1f1;
}
#T_a7b5f_row5_col13, #T_a7b5f_row6_col21, #T_a7b5f_row8_col17, #T_a7b5f_row9_col29, #T_a7b5f_row9_col30, #T_a7b5f_row10_col21, #T_a7b5f_row19_col24, #T_a7b5f_row28_col0 {
  background-color: #3f53c6;
  color: #f1f1f1;
}
#T_a7b5f_row6_col8, #T_a7b5f_row13_col30, #T_a7b5f_row18_col3, #T_a7b5f_row18_col27, #T_a7b5f_row19_col2, #T_a7b5f_row19_col25, #T_a7b5f_row19_col26, #T_a7b5f_row19_col28, #T_a7b5f_row21_col14, #T_a7b5f_row22_col3, #T_a7b5f_row26_col27, #T_a7b5f_row27_col11, #T_a7b5f_row29_col8, #T_a7b5f_row29_col11, #T_a7b5f_row29_col25 {
  background-color: #6180e9;
  color: #f1f1f1;
}
#T_a7b5f_row6_col23, #T_a7b5f_row10_col11, #T_a7b5f_row10_col12, #T_a7b5f_row11_col29, #T_a7b5f_row15_col28, #T_a7b5f_row21_col28, #T_a7b5f_row25_col29, #T_a7b5f_row26_col9, #T_a7b5f_row28_col17 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_a7b5f_row7_col3, #T_a7b5f_row27_col29, #T_a7b5f_row28_col20 {
  background-color: #a2c1ff;
  color: #000000;
}
#T_a7b5f_row7_col20 {
  background-color: #a5c3fe;
  color: #000000;
}
#T_a7b5f_row7_col27 {
  background-color: #b5cdfa;
  color: #000000;
}
#T_a7b5f_row8_col9, #T_a7b5f_row26_col7, #T_a7b5f_row30_col20, #T_a7b5f_row30_col27 {
  background-color: #86a9fc;
  color: #f1f1f1;
}
#T_a7b5f_row9_col0 {
  background-color: #f7a889;
  color: #000000;
}
#T_a7b5f_row9_col8, #T_a7b5f_row18_col30 {
  background-color: #92b4fe;
  color: #000000;
}
#T_a7b5f_row9_col10, #T_a7b5f_row27_col20 {
  background-color: #a7c5fe;
  color: #000000;
}
#T_a7b5f_row9_col13, #T_a7b5f_row10_col15, #T_a7b5f_row19_col15, #T_a7b5f_row19_col30, #T_a7b5f_row29_col4, #T_a7b5f_row30_col17, #T_a7b5f_row31_col6, #T_a7b5f_row31_col22, #T_a7b5f_row31_col23, #T_a7b5f_row31_col24 {
  background-color: #4257c9;
  color: #f1f1f1;
}
#T_a7b5f_row9_col18, #T_a7b5f_row11_col7, #T_a7b5f_row16_col28, #T_a7b5f_row17_col28 {
  background-color: #82a6fb;
  color: #f1f1f1;
}
#T_a7b5f_row10_col0, #T_a7b5f_row10_col9 {
  background-color: #adc9fd;
  color: #000000;
}
#T_a7b5f_row10_col16 {
  background-color: #90b2fe;
  color: #000000;
}
#T_a7b5f_row11_col8, #T_a7b5f_row17_col26, #T_a7b5f_row24_col3, #T_a7b5f_row25_col16, #T_a7b5f_row28_col1 {
  background-color: #7699f6;
  color: #f1f1f1;
}
#T_a7b5f_row11_col12, #T_a7b5f_row29_col30 {
  background-color: #edd1c2;
  color: #000000;
}
#T_a7b5f_row12_col11 {
  background-color: #edd2c3;
  color: #000000;
}
#T_a7b5f_row12_col16, #T_a7b5f_row18_col0, #T_a7b5f_row22_col18, #T_a7b5f_row31_col28 {
  background-color: #7ea1fa;
  color: #f1f1f1;
}
#T_a7b5f_row13_col15, #T_a7b5f_row15_col13 {
  background-color: #bd1f2d;
  color: #f1f1f1;
}
#T_a7b5f_row13_col18, #T_a7b5f_row15_col18, #T_a7b5f_row17_col27, #T_a7b5f_row22_col28, #T_a7b5f_row27_col18, #T_a7b5f_row31_col16 {
  background-color: #81a4fb;
  color: #f1f1f1;
}
#T_a7b5f_row16_col1 {
  background-color: #f7ba9f;
  color: #000000;
}
#T_a7b5f_row16_col2 {
  background-color: #9ebeff;
  color: #000000;
}
#T_a7b5f_row16_col3 {
  background-color: #f5c0a7;
  color: #000000;
}
#T_a7b5f_row16_col7 {
  background-color: #a3c2fe;
  color: #000000;
}
#T_a7b5f_row16_col17 {
  background-color: #dd5f4b;
  color: #f1f1f1;
}
#T_a7b5f_row17_col1 {
  background-color: #f7b89c;
  color: #000000;
}
#T_a7b5f_row17_col2, #T_a7b5f_row18_col29 {
  background-color: #9dbdff;
  color: #000000;
}
#T_a7b5f_row17_col3 {
  background-color: #f6bea4;
  color: #000000;
}
#T_a7b5f_row17_col7 {
  background-color: #a1c0ff;
  color: #000000;
}
#T_a7b5f_row17_col16 {
  background-color: #d75445;
  color: #f1f1f1;
}
#T_a7b5f_row17_col18 {
  background-color: #c1d4f4;
  color: #000000;
}
#T_a7b5f_row18_col17 {
  background-color: #9abbff;
  color: #000000;
}
#T_a7b5f_row20_col7 {
  background-color: #a9c6fd;
  color: #000000;
}
#T_a7b5f_row20_col27 {
  background-color: #98b9ff;
  color: #000000;
}
#T_a7b5f_row21_col18, #T_a7b5f_row25_col18, #T_a7b5f_row29_col20 {
  background-color: #93b5fe;
  color: #000000;
}
#T_a7b5f_row23_col16, #T_a7b5f_row24_col0 {
  background-color: #88abfd;
  color: #000000;
}
#T_a7b5f_row26_col16 {
  background-color: #97b8ff;
  color: #000000;
}
#T_a7b5f_row27_col7 {
  background-color: #c4d5f3;
  color: #000000;
}
#T_a7b5f_row27_col16 {
  background-color: #9bbcff;
  color: #000000;
}
#T_a7b5f_row27_col28 {
  background-color: #e9785d;
  color: #f1f1f1;
}
#T_a7b5f_row28_col7 {
  background-color: #b9d0f9;
  color: #000000;
}
#T_a7b5f_row28_col16 {
  background-color: #94b6ff;
  color: #000000;
}
#T_a7b5f_row28_col27 {
  background-color: #ea7b60;
  color: #f1f1f1;
}
#T_a7b5f_row28_col30, #T_a7b5f_row29_col28 {
  background-color: #96b7ff;
  color: #000000;
}
#T_a7b5f_row29_col18 {
  background-color: #bad0f8;
  color: #000000;
}
#T_a7b5f_row29_col27 {
  background-color: #a6c4fe;
  color: #000000;
}
#T_a7b5f_row30_col18 {
  background-color: #abc8fd;
  color: #000000;
}
#T_a7b5f_row30_col29 {
  background-color: #ecd3c5;
  color: #000000;
}
</style>
<table id="T_a7b5f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a7b5f_level0_col0" class="col_heading level0 col0" > Age (yrs)</th>
      <th id="T_a7b5f_level0_col1" class="col_heading level0 col1" >Weight (Kg)</th>
      <th id="T_a7b5f_level0_col2" class="col_heading level0 col2" >Height(Cm) </th>
      <th id="T_a7b5f_level0_col3" class="col_heading level0 col3" >BMI</th>
      <th id="T_a7b5f_level0_col4" class="col_heading level0 col4" >Pulse rate(bpm) </th>
      <th id="T_a7b5f_level0_col5" class="col_heading level0 col5" >RR (breaths/min)</th>
      <th id="T_a7b5f_level0_col6" class="col_heading level0 col6" >Hb(g/dl)</th>
      <th id="T_a7b5f_level0_col7" class="col_heading level0 col7" >Cycle(R/I)</th>
      <th id="T_a7b5f_level0_col8" class="col_heading level0 col8" >Cycle length(days)</th>
      <th id="T_a7b5f_level0_col9" class="col_heading level0 col9" >Marraige Status (Yrs)</th>
      <th id="T_a7b5f_level0_col10" class="col_heading level0 col10" >No. of aborptions</th>
      <th id="T_a7b5f_level0_col11" class="col_heading level0 col11" >  I   beta-HCG(mIU/mL)</th>
      <th id="T_a7b5f_level0_col12" class="col_heading level0 col12" >II    beta-HCG(mIU/mL)</th>
      <th id="T_a7b5f_level0_col13" class="col_heading level0 col13" >FSH(mIU/mL)</th>
      <th id="T_a7b5f_level0_col14" class="col_heading level0 col14" >LH(mIU/mL)</th>
      <th id="T_a7b5f_level0_col15" class="col_heading level0 col15" >FSH/LH</th>
      <th id="T_a7b5f_level0_col16" class="col_heading level0 col16" >Hip(inch)</th>
      <th id="T_a7b5f_level0_col17" class="col_heading level0 col17" >Waist(inch)</th>
      <th id="T_a7b5f_level0_col18" class="col_heading level0 col18" >Waist:Hip Ratio</th>
      <th id="T_a7b5f_level0_col19" class="col_heading level0 col19" >TSH (mIU/L)</th>
      <th id="T_a7b5f_level0_col20" class="col_heading level0 col20" >AMH(ng/mL)</th>
      <th id="T_a7b5f_level0_col21" class="col_heading level0 col21" >PRL(ng/mL)</th>
      <th id="T_a7b5f_level0_col22" class="col_heading level0 col22" >Vit D3 (ng/mL)</th>
      <th id="T_a7b5f_level0_col23" class="col_heading level0 col23" >PRG(ng/mL)</th>
      <th id="T_a7b5f_level0_col24" class="col_heading level0 col24" >RBS(mg/dl)</th>
      <th id="T_a7b5f_level0_col25" class="col_heading level0 col25" >BP _Systolic (mmHg)</th>
      <th id="T_a7b5f_level0_col26" class="col_heading level0 col26" >BP _Diastolic (mmHg)</th>
      <th id="T_a7b5f_level0_col27" class="col_heading level0 col27" >Follicle No. (L)</th>
      <th id="T_a7b5f_level0_col28" class="col_heading level0 col28" >Follicle No. (R)</th>
      <th id="T_a7b5f_level0_col29" class="col_heading level0 col29" >Avg. F size (L) (mm)</th>
      <th id="T_a7b5f_level0_col30" class="col_heading level0 col30" >Avg. F size (R) (mm)</th>
      <th id="T_a7b5f_level0_col31" class="col_heading level0 col31" >Endometrium (mm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a7b5f_level0_row0" class="row_heading level0 row0" > Age (yrs)</th>
      <td id="T_a7b5f_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_a7b5f_row0_col1" class="data row0 col1" >-0.029136</td>
      <td id="T_a7b5f_row0_col2" class="data row0 col2" >-0.118577</td>
      <td id="T_a7b5f_row0_col3" class="data row0 col3" >0.021316</td>
      <td id="T_a7b5f_row0_col4" class="data row0 col4" >0.045553</td>
      <td id="T_a7b5f_row0_col5" class="data row0 col5" >0.086628</td>
      <td id="T_a7b5f_row0_col6" class="data row0 col6" >-0.022778</td>
      <td id="T_a7b5f_row0_col7" class="data row0 col7" >-0.084871</td>
      <td id="T_a7b5f_row0_col8" class="data row0 col8" >0.055898</td>
      <td id="T_a7b5f_row0_col9" class="data row0 col9" >0.662254</td>
      <td id="T_a7b5f_row0_col10" class="data row0 col10" >0.221843</td>
      <td id="T_a7b5f_row0_col11" class="data row0 col11" >0.008511</td>
      <td id="T_a7b5f_row0_col12" class="data row0 col12" >0.043378</td>
      <td id="T_a7b5f_row0_col13" class="data row0 col13" >-0.017708</td>
      <td id="T_a7b5f_row0_col14" class="data row0 col14" >0.000603</td>
      <td id="T_a7b5f_row0_col15" class="data row0 col15" >0.012432</td>
      <td id="T_a7b5f_row0_col16" class="data row0 col16" >-0.001346</td>
      <td id="T_a7b5f_row0_col17" class="data row0 col17" >0.036263</td>
      <td id="T_a7b5f_row0_col18" class="data row0 col18" >0.066452</td>
      <td id="T_a7b5f_row0_col19" class="data row0 col19" >0.010245</td>
      <td id="T_a7b5f_row0_col20" class="data row0 col20" >-0.179648</td>
      <td id="T_a7b5f_row0_col21" class="data row0 col21" >-0.046077</td>
      <td id="T_a7b5f_row0_col22" class="data row0 col22" >0.004454</td>
      <td id="T_a7b5f_row0_col23" class="data row0 col23" >-0.021775</td>
      <td id="T_a7b5f_row0_col24" class="data row0 col24" >0.097086</td>
      <td id="T_a7b5f_row0_col25" class="data row0 col25" >0.072313</td>
      <td id="T_a7b5f_row0_col26" class="data row0 col26" >0.069329</td>
      <td id="T_a7b5f_row0_col27" class="data row0 col27" >-0.109965</td>
      <td id="T_a7b5f_row0_col28" class="data row0 col28" >-0.158865</td>
      <td id="T_a7b5f_row0_col29" class="data row0 col29" >-0.017435</td>
      <td id="T_a7b5f_row0_col30" class="data row0 col30" >-0.079646</td>
      <td id="T_a7b5f_row0_col31" class="data row0 col31" >-0.101969</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row1" class="row_heading level0 row1" >Weight (Kg)</th>
      <td id="T_a7b5f_row1_col0" class="data row1 col0" >-0.029136</td>
      <td id="T_a7b5f_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_a7b5f_row1_col2" class="data row1 col2" >0.419901</td>
      <td id="T_a7b5f_row1_col3" class="data row1 col3" >0.901755</td>
      <td id="T_a7b5f_row1_col4" class="data row1 col4" >0.020089</td>
      <td id="T_a7b5f_row1_col5" class="data row1 col5" >0.043901</td>
      <td id="T_a7b5f_row1_col6" class="data row1 col6" >0.009979</td>
      <td id="T_a7b5f_row1_col7" class="data row1 col7" >0.200470</td>
      <td id="T_a7b5f_row1_col8" class="data row1 col8" >-0.002284</td>
      <td id="T_a7b5f_row1_col9" class="data row1 col9" >0.043991</td>
      <td id="T_a7b5f_row1_col10" class="data row1 col10" >0.093310</td>
      <td id="T_a7b5f_row1_col11" class="data row1 col11" >0.015882</td>
      <td id="T_a7b5f_row1_col12" class="data row1 col12" >-0.000521</td>
      <td id="T_a7b5f_row1_col13" class="data row1 col13" >-0.025785</td>
      <td id="T_a7b5f_row1_col14" class="data row1 col14" >-0.029910</td>
      <td id="T_a7b5f_row1_col15" class="data row1 col15" >-0.004831</td>
      <td id="T_a7b5f_row1_col16" class="data row1 col16" >0.633920</td>
      <td id="T_a7b5f_row1_col17" class="data row1 col17" >0.639589</td>
      <td id="T_a7b5f_row1_col18" class="data row1 col18" >0.014976</td>
      <td id="T_a7b5f_row1_col19" class="data row1 col19" >0.071410</td>
      <td id="T_a7b5f_row1_col20" class="data row1 col20" >0.031050</td>
      <td id="T_a7b5f_row1_col21" class="data row1 col21" >-0.050016</td>
      <td id="T_a7b5f_row1_col22" class="data row1 col22" >0.008145</td>
      <td id="T_a7b5f_row1_col23" class="data row1 col23" >0.069688</td>
      <td id="T_a7b5f_row1_col24" class="data row1 col24" >0.114294</td>
      <td id="T_a7b5f_row1_col25" class="data row1 col25" >0.028066</td>
      <td id="T_a7b5f_row1_col26" class="data row1 col26" >0.130843</td>
      <td id="T_a7b5f_row1_col27" class="data row1 col27" >0.173501</td>
      <td id="T_a7b5f_row1_col28" class="data row1 col28" >0.124092</td>
      <td id="T_a7b5f_row1_col29" class="data row1 col29" >-0.021036</td>
      <td id="T_a7b5f_row1_col30" class="data row1 col30" >-0.073115</td>
      <td id="T_a7b5f_row1_col31" class="data row1 col31" >-0.010932</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row2" class="row_heading level0 row2" >Height(Cm) </th>
      <td id="T_a7b5f_row2_col0" class="data row2 col0" >-0.118577</td>
      <td id="T_a7b5f_row2_col1" class="data row2 col1" >0.419901</td>
      <td id="T_a7b5f_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_a7b5f_row2_col3" class="data row2 col3" >-0.006906</td>
      <td id="T_a7b5f_row2_col4" class="data row2 col4" >-0.074144</td>
      <td id="T_a7b5f_row2_col5" class="data row2 col5" >-0.028862</td>
      <td id="T_a7b5f_row2_col6" class="data row2 col6" >0.025259</td>
      <td id="T_a7b5f_row2_col7" class="data row2 col7" >-0.018208</td>
      <td id="T_a7b5f_row2_col8" class="data row2 col8" >0.009595</td>
      <td id="T_a7b5f_row2_col9" class="data row2 col9" >-0.066407</td>
      <td id="T_a7b5f_row2_col10" class="data row2 col10" >-0.026240</td>
      <td id="T_a7b5f_row2_col11" class="data row2 col11" >0.062079</td>
      <td id="T_a7b5f_row2_col12" class="data row2 col12" >0.036474</td>
      <td id="T_a7b5f_row2_col13" class="data row2 col13" >0.030883</td>
      <td id="T_a7b5f_row2_col14" class="data row2 col14" >-0.045619</td>
      <td id="T_a7b5f_row2_col15" class="data row2 col15" >0.022065</td>
      <td id="T_a7b5f_row2_col16" class="data row2 col16" >0.215357</td>
      <td id="T_a7b5f_row2_col17" class="data row2 col17" >0.209348</td>
      <td id="T_a7b5f_row2_col18" class="data row2 col18" >-0.008982</td>
      <td id="T_a7b5f_row2_col19" class="data row2 col19" >0.018501</td>
      <td id="T_a7b5f_row2_col20" class="data row2 col20" >-0.045208</td>
      <td id="T_a7b5f_row2_col21" class="data row2 col21" >-0.018178</td>
      <td id="T_a7b5f_row2_col22" class="data row2 col22" >-0.034997</td>
      <td id="T_a7b5f_row2_col23" class="data row2 col23" >0.049654</td>
      <td id="T_a7b5f_row2_col24" class="data row2 col24" >0.050437</td>
      <td id="T_a7b5f_row2_col25" class="data row2 col25" >-0.067029</td>
      <td id="T_a7b5f_row2_col26" class="data row2 col26" >0.009420</td>
      <td id="T_a7b5f_row2_col27" class="data row2 col27" >0.105574</td>
      <td id="T_a7b5f_row2_col28" class="data row2 col28" >0.074896</td>
      <td id="T_a7b5f_row2_col29" class="data row2 col29" >-0.025959</td>
      <td id="T_a7b5f_row2_col30" class="data row2 col30" >0.059686</td>
      <td id="T_a7b5f_row2_col31" class="data row2 col31" >-0.055987</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row3" class="row_heading level0 row3" >BMI</th>
      <td id="T_a7b5f_row3_col0" class="data row3 col0" >0.021316</td>
      <td id="T_a7b5f_row3_col1" class="data row3 col1" >0.901755</td>
      <td id="T_a7b5f_row3_col2" class="data row3 col2" >-0.006906</td>
      <td id="T_a7b5f_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_a7b5f_row3_col4" class="data row3 col4" >0.050536</td>
      <td id="T_a7b5f_row3_col5" class="data row3 col5" >0.061932</td>
      <td id="T_a7b5f_row3_col6" class="data row3 col6" >0.003534</td>
      <td id="T_a7b5f_row3_col7" class="data row3 col7" >0.232891</td>
      <td id="T_a7b5f_row3_col8" class="data row3 col8" >-0.006231</td>
      <td id="T_a7b5f_row3_col9" class="data row3 col9" >0.084015</td>
      <td id="T_a7b5f_row3_col10" class="data row3 col10" >0.109865</td>
      <td id="T_a7b5f_row3_col11" class="data row3 col11" >-0.009967</td>
      <td id="T_a7b5f_row3_col12" class="data row3 col12" >-0.015270</td>
      <td id="T_a7b5f_row3_col13" class="data row3 col13" >-0.040717</td>
      <td id="T_a7b5f_row3_col14" class="data row3 col14" >-0.013312</td>
      <td id="T_a7b5f_row3_col15" class="data row3 col15" >-0.012076</td>
      <td id="T_a7b5f_row3_col16" class="data row3 col16" >0.597058</td>
      <td id="T_a7b5f_row3_col17" class="data row3 col17" >0.607524</td>
      <td id="T_a7b5f_row3_col18" class="data row3 col18" >0.023602</td>
      <td id="T_a7b5f_row3_col19" class="data row3 col19" >0.072305</td>
      <td id="T_a7b5f_row3_col20" class="data row3 col20" >0.054023</td>
      <td id="T_a7b5f_row3_col21" class="data row3 col21" >-0.047721</td>
      <td id="T_a7b5f_row3_col22" class="data row3 col22" >0.027035</td>
      <td id="T_a7b5f_row3_col23" class="data row3 col23" >0.049460</td>
      <td id="T_a7b5f_row3_col24" class="data row3 col24" >0.093543</td>
      <td id="T_a7b5f_row3_col25" class="data row3 col25" >0.069549</td>
      <td id="T_a7b5f_row3_col26" class="data row3 col26" >0.140134</td>
      <td id="T_a7b5f_row3_col27" class="data row3 col27" >0.142903</td>
      <td id="T_a7b5f_row3_col28" class="data row3 col28" >0.104205</td>
      <td id="T_a7b5f_row3_col29" class="data row3 col29" >-0.011594</td>
      <td id="T_a7b5f_row3_col30" class="data row3 col30" >-0.111519</td>
      <td id="T_a7b5f_row3_col31" class="data row3 col31" >0.009320</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row4" class="row_heading level0 row4" >Pulse rate(bpm) </th>
      <td id="T_a7b5f_row4_col0" class="data row4 col0" >0.045553</td>
      <td id="T_a7b5f_row4_col1" class="data row4 col1" >0.020089</td>
      <td id="T_a7b5f_row4_col2" class="data row4 col2" >-0.074144</td>
      <td id="T_a7b5f_row4_col3" class="data row4 col3" >0.050536</td>
      <td id="T_a7b5f_row4_col4" class="data row4 col4" >1.000000</td>
      <td id="T_a7b5f_row4_col5" class="data row4 col5" >0.303727</td>
      <td id="T_a7b5f_row4_col6" class="data row4 col6" >-0.052265</td>
      <td id="T_a7b5f_row4_col7" class="data row4 col7" >0.101241</td>
      <td id="T_a7b5f_row4_col8" class="data row4 col8" >0.006411</td>
      <td id="T_a7b5f_row4_col9" class="data row4 col9" >0.038701</td>
      <td id="T_a7b5f_row4_col10" class="data row4 col10" >0.046227</td>
      <td id="T_a7b5f_row4_col11" class="data row4 col11" >-0.020437</td>
      <td id="T_a7b5f_row4_col12" class="data row4 col12" >-0.016226</td>
      <td id="T_a7b5f_row4_col13" class="data row4 col13" >-0.013071</td>
      <td id="T_a7b5f_row4_col14" class="data row4 col14" >-0.032315</td>
      <td id="T_a7b5f_row4_col15" class="data row4 col15" >-0.013104</td>
      <td id="T_a7b5f_row4_col16" class="data row4 col16" >0.062951</td>
      <td id="T_a7b5f_row4_col17" class="data row4 col17" >0.037880</td>
      <td id="T_a7b5f_row4_col18" class="data row4 col18" >-0.052890</td>
      <td id="T_a7b5f_row4_col19" class="data row4 col19" >-0.051482</td>
      <td id="T_a7b5f_row4_col20" class="data row4 col20" >-0.049843</td>
      <td id="T_a7b5f_row4_col21" class="data row4 col21" >0.020750</td>
      <td id="T_a7b5f_row4_col22" class="data row4 col22" >-0.001486</td>
      <td id="T_a7b5f_row4_col23" class="data row4 col23" >-0.017678</td>
      <td id="T_a7b5f_row4_col24" class="data row4 col24" >0.042002</td>
      <td id="T_a7b5f_row4_col25" class="data row4 col25" >-0.025751</td>
      <td id="T_a7b5f_row4_col26" class="data row4 col26" >0.008027</td>
      <td id="T_a7b5f_row4_col27" class="data row4 col27" >0.040559</td>
      <td id="T_a7b5f_row4_col28" class="data row4 col28" >0.049307</td>
      <td id="T_a7b5f_row4_col29" class="data row4 col29" >-0.048546</td>
      <td id="T_a7b5f_row4_col30" class="data row4 col30" >-0.034256</td>
      <td id="T_a7b5f_row4_col31" class="data row4 col31" >-0.040891</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row5" class="row_heading level0 row5" >RR (breaths/min)</th>
      <td id="T_a7b5f_row5_col0" class="data row5 col0" >0.086628</td>
      <td id="T_a7b5f_row5_col1" class="data row5 col1" >0.043901</td>
      <td id="T_a7b5f_row5_col2" class="data row5 col2" >-0.028862</td>
      <td id="T_a7b5f_row5_col3" class="data row5 col3" >0.061932</td>
      <td id="T_a7b5f_row5_col4" class="data row5 col4" >0.303727</td>
      <td id="T_a7b5f_row5_col5" class="data row5 col5" >1.000000</td>
      <td id="T_a7b5f_row5_col6" class="data row5 col6" >-0.041027</td>
      <td id="T_a7b5f_row5_col7" class="data row5 col7" >0.018850</td>
      <td id="T_a7b5f_row5_col8" class="data row5 col8" >0.004972</td>
      <td id="T_a7b5f_row5_col9" class="data row5 col9" >0.077701</td>
      <td id="T_a7b5f_row5_col10" class="data row5 col10" >-0.006090</td>
      <td id="T_a7b5f_row5_col11" class="data row5 col11" >-0.085028</td>
      <td id="T_a7b5f_row5_col12" class="data row5 col12" >-0.039017</td>
      <td id="T_a7b5f_row5_col13" class="data row5 col13" >-0.032388</td>
      <td id="T_a7b5f_row5_col14" class="data row5 col14" >-0.031211</td>
      <td id="T_a7b5f_row5_col15" class="data row5 col15" >-0.043339</td>
      <td id="T_a7b5f_row5_col16" class="data row5 col16" >0.075020</td>
      <td id="T_a7b5f_row5_col17" class="data row5 col17" >0.038310</td>
      <td id="T_a7b5f_row5_col18" class="data row5 col18" >-0.075567</td>
      <td id="T_a7b5f_row5_col19" class="data row5 col19" >-0.011968</td>
      <td id="T_a7b5f_row5_col20" class="data row5 col20" >-0.017480</td>
      <td id="T_a7b5f_row5_col21" class="data row5 col21" >0.007284</td>
      <td id="T_a7b5f_row5_col22" class="data row5 col22" >-0.009016</td>
      <td id="T_a7b5f_row5_col23" class="data row5 col23" >-0.076895</td>
      <td id="T_a7b5f_row5_col24" class="data row5 col24" >0.050811</td>
      <td id="T_a7b5f_row5_col25" class="data row5 col25" >0.016734</td>
      <td id="T_a7b5f_row5_col26" class="data row5 col26" >0.053751</td>
      <td id="T_a7b5f_row5_col27" class="data row5 col27" >0.070179</td>
      <td id="T_a7b5f_row5_col28" class="data row5 col28" >0.012752</td>
      <td id="T_a7b5f_row5_col29" class="data row5 col29" >-0.031527</td>
      <td id="T_a7b5f_row5_col30" class="data row5 col30" >-0.022035</td>
      <td id="T_a7b5f_row5_col31" class="data row5 col31" >-0.062941</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row6" class="row_heading level0 row6" >Hb(g/dl)</th>
      <td id="T_a7b5f_row6_col0" class="data row6 col0" >-0.022778</td>
      <td id="T_a7b5f_row6_col1" class="data row6 col1" >0.009979</td>
      <td id="T_a7b5f_row6_col2" class="data row6 col2" >0.025259</td>
      <td id="T_a7b5f_row6_col3" class="data row6 col3" >0.003534</td>
      <td id="T_a7b5f_row6_col4" class="data row6 col4" >-0.052265</td>
      <td id="T_a7b5f_row6_col5" class="data row6 col5" >-0.041027</td>
      <td id="T_a7b5f_row6_col6" class="data row6 col6" >1.000000</td>
      <td id="T_a7b5f_row6_col7" class="data row6 col7" >0.037428</td>
      <td id="T_a7b5f_row6_col8" class="data row6 col8" >-0.051992</td>
      <td id="T_a7b5f_row6_col9" class="data row6 col9" >0.006743</td>
      <td id="T_a7b5f_row6_col10" class="data row6 col10" >0.060702</td>
      <td id="T_a7b5f_row6_col11" class="data row6 col11" >-0.016649</td>
      <td id="T_a7b5f_row6_col12" class="data row6 col12" >-0.094425</td>
      <td id="T_a7b5f_row6_col13" class="data row6 col13" >-0.047398</td>
      <td id="T_a7b5f_row6_col14" class="data row6 col14" >-0.089107</td>
      <td id="T_a7b5f_row6_col15" class="data row6 col15" >-0.039827</td>
      <td id="T_a7b5f_row6_col16" class="data row6 col16" >-0.024712</td>
      <td id="T_a7b5f_row6_col17" class="data row6 col17" >-0.000779</td>
      <td id="T_a7b5f_row6_col18" class="data row6 col18" >0.057161</td>
      <td id="T_a7b5f_row6_col19" class="data row6 col19" >-0.026156</td>
      <td id="T_a7b5f_row6_col20" class="data row6 col20" >0.056172</td>
      <td id="T_a7b5f_row6_col21" class="data row6 col21" >-0.063091</td>
      <td id="T_a7b5f_row6_col22" class="data row6 col22" >0.063916</td>
      <td id="T_a7b5f_row6_col23" class="data row6 col23" >0.065769</td>
      <td id="T_a7b5f_row6_col24" class="data row6 col24" >0.024067</td>
      <td id="T_a7b5f_row6_col25" class="data row6 col25" >0.052229</td>
      <td id="T_a7b5f_row6_col26" class="data row6 col26" >0.002046</td>
      <td id="T_a7b5f_row6_col27" class="data row6 col27" >0.061814</td>
      <td id="T_a7b5f_row6_col28" class="data row6 col28" >0.073422</td>
      <td id="T_a7b5f_row6_col29" class="data row6 col29" >0.031995</td>
      <td id="T_a7b5f_row6_col30" class="data row6 col30" >0.024153</td>
      <td id="T_a7b5f_row6_col31" class="data row6 col31" >-0.065041</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row7" class="row_heading level0 row7" >Cycle(R/I)</th>
      <td id="T_a7b5f_row7_col0" class="data row7 col0" >-0.084871</td>
      <td id="T_a7b5f_row7_col1" class="data row7 col1" >0.200470</td>
      <td id="T_a7b5f_row7_col2" class="data row7 col2" >-0.018208</td>
      <td id="T_a7b5f_row7_col3" class="data row7 col3" >0.232891</td>
      <td id="T_a7b5f_row7_col4" class="data row7 col4" >0.101241</td>
      <td id="T_a7b5f_row7_col5" class="data row7 col5" >0.018850</td>
      <td id="T_a7b5f_row7_col6" class="data row7 col6" >0.037428</td>
      <td id="T_a7b5f_row7_col7" class="data row7 col7" >1.000000</td>
      <td id="T_a7b5f_row7_col8" class="data row7 col8" >-0.201044</td>
      <td id="T_a7b5f_row7_col9" class="data row7 col9" >-0.033557</td>
      <td id="T_a7b5f_row7_col10" class="data row7 col10" >-0.057937</td>
      <td id="T_a7b5f_row7_col11" class="data row7 col11" >0.063098</td>
      <td id="T_a7b5f_row7_col12" class="data row7 col12" >0.027930</td>
      <td id="T_a7b5f_row7_col13" class="data row7 col13" >-0.026083</td>
      <td id="T_a7b5f_row7_col14" class="data row7 col14" >-0.021393</td>
      <td id="T_a7b5f_row7_col15" class="data row7 col15" >-0.016167</td>
      <td id="T_a7b5f_row7_col16" class="data row7 col16" >0.174305</td>
      <td id="T_a7b5f_row7_col17" class="data row7 col17" >0.168856</td>
      <td id="T_a7b5f_row7_col18" class="data row7 col18" >-0.004003</td>
      <td id="T_a7b5f_row7_col19" class="data row7 col19" >-0.017435</td>
      <td id="T_a7b5f_row7_col20" class="data row7 col20" >0.194063</td>
      <td id="T_a7b5f_row7_col21" class="data row7 col21" >0.005251</td>
      <td id="T_a7b5f_row7_col22" class="data row7 col22" >0.096717</td>
      <td id="T_a7b5f_row7_col23" class="data row7 col23" >-0.033781</td>
      <td id="T_a7b5f_row7_col24" class="data row7 col24" >-0.006394</td>
      <td id="T_a7b5f_row7_col25" class="data row7 col25" >0.055790</td>
      <td id="T_a7b5f_row7_col26" class="data row7 col26" >0.080057</td>
      <td id="T_a7b5f_row7_col27" class="data row7 col27" >0.296114</td>
      <td id="T_a7b5f_row7_col28" class="data row7 col28" >0.251271</td>
      <td id="T_a7b5f_row7_col29" class="data row7 col29" >0.034112</td>
      <td id="T_a7b5f_row7_col30" class="data row7 col30" >0.016204</td>
      <td id="T_a7b5f_row7_col31" class="data row7 col31" >0.042168</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row8" class="row_heading level0 row8" >Cycle length(days)</th>
      <td id="T_a7b5f_row8_col0" class="data row8 col0" >0.055898</td>
      <td id="T_a7b5f_row8_col1" class="data row8 col1" >-0.002284</td>
      <td id="T_a7b5f_row8_col2" class="data row8 col2" >0.009595</td>
      <td id="T_a7b5f_row8_col3" class="data row8 col3" >-0.006231</td>
      <td id="T_a7b5f_row8_col4" class="data row8 col4" >0.006411</td>
      <td id="T_a7b5f_row8_col5" class="data row8 col5" >0.004972</td>
      <td id="T_a7b5f_row8_col6" class="data row8 col6" >-0.051992</td>
      <td id="T_a7b5f_row8_col7" class="data row8 col7" >-0.201044</td>
      <td id="T_a7b5f_row8_col8" class="data row8 col8" >1.000000</td>
      <td id="T_a7b5f_row8_col9" class="data row8 col9" >0.117850</td>
      <td id="T_a7b5f_row8_col10" class="data row8 col10" >0.004023</td>
      <td id="T_a7b5f_row8_col11" class="data row8 col11" >0.020187</td>
      <td id="T_a7b5f_row8_col12" class="data row8 col12" >0.018577</td>
      <td id="T_a7b5f_row8_col13" class="data row8 col13" >0.029645</td>
      <td id="T_a7b5f_row8_col14" class="data row8 col14" >-0.001686</td>
      <td id="T_a7b5f_row8_col15" class="data row8 col15" >0.025937</td>
      <td id="T_a7b5f_row8_col16" class="data row8 col16" >0.040354</td>
      <td id="T_a7b5f_row8_col17" class="data row8 col17" >-0.023479</td>
      <td id="T_a7b5f_row8_col18" class="data row8 col18" >-0.130093</td>
      <td id="T_a7b5f_row8_col19" class="data row8 col19" >-0.003191</td>
      <td id="T_a7b5f_row8_col20" class="data row8 col20" >-0.072822</td>
      <td id="T_a7b5f_row8_col21" class="data row8 col21" >0.016415</td>
      <td id="T_a7b5f_row8_col22" class="data row8 col22" >-0.058345</td>
      <td id="T_a7b5f_row8_col23" class="data row8 col23" >0.007190</td>
      <td id="T_a7b5f_row8_col24" class="data row8 col24" >0.000210</td>
      <td id="T_a7b5f_row8_col25" class="data row8 col25" >-0.011963</td>
      <td id="T_a7b5f_row8_col26" class="data row8 col26" >-0.075792</td>
      <td id="T_a7b5f_row8_col27" class="data row8 col27" >-0.086809</td>
      <td id="T_a7b5f_row8_col28" class="data row8 col28" >-0.161256</td>
      <td id="T_a7b5f_row8_col29" class="data row8 col29" >-0.052364</td>
      <td id="T_a7b5f_row8_col30" class="data row8 col30" >-0.013956</td>
      <td id="T_a7b5f_row8_col31" class="data row8 col31" >-0.016506</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row9" class="row_heading level0 row9" >Marraige Status (Yrs)</th>
      <td id="T_a7b5f_row9_col0" class="data row9 col0" >0.662254</td>
      <td id="T_a7b5f_row9_col1" class="data row9 col1" >0.043991</td>
      <td id="T_a7b5f_row9_col2" class="data row9 col2" >-0.066407</td>
      <td id="T_a7b5f_row9_col3" class="data row9 col3" >0.084015</td>
      <td id="T_a7b5f_row9_col4" class="data row9 col4" >0.038701</td>
      <td id="T_a7b5f_row9_col5" class="data row9 col5" >0.077701</td>
      <td id="T_a7b5f_row9_col6" class="data row9 col6" >0.006743</td>
      <td id="T_a7b5f_row9_col7" class="data row9 col7" >-0.033557</td>
      <td id="T_a7b5f_row9_col8" class="data row9 col8" >0.117850</td>
      <td id="T_a7b5f_row9_col9" class="data row9 col9" >1.000000</td>
      <td id="T_a7b5f_row9_col10" class="data row9 col10" >0.246713</td>
      <td id="T_a7b5f_row9_col11" class="data row9 col11" >0.111666</td>
      <td id="T_a7b5f_row9_col12" class="data row9 col12" >0.112905</td>
      <td id="T_a7b5f_row9_col13" class="data row9 col13" >-0.023461</td>
      <td id="T_a7b5f_row9_col14" class="data row9 col14" >0.035509</td>
      <td id="T_a7b5f_row9_col15" class="data row9 col15" >-0.002983</td>
      <td id="T_a7b5f_row9_col16" class="data row9 col16" >0.038325</td>
      <td id="T_a7b5f_row9_col17" class="data row9 col17" >0.057376</td>
      <td id="T_a7b5f_row9_col18" class="data row9 col18" >0.033334</td>
      <td id="T_a7b5f_row9_col19" class="data row9 col19" >-0.040241</td>
      <td id="T_a7b5f_row9_col20" class="data row9 col20" >-0.146482</td>
      <td id="T_a7b5f_row9_col21" class="data row9 col21" >-0.025819</td>
      <td id="T_a7b5f_row9_col22" class="data row9 col22" >0.041541</td>
      <td id="T_a7b5f_row9_col23" class="data row9 col23" >-0.052840</td>
      <td id="T_a7b5f_row9_col24" class="data row9 col24" >0.075519</td>
      <td id="T_a7b5f_row9_col25" class="data row9 col25" >0.028478</td>
      <td id="T_a7b5f_row9_col26" class="data row9 col26" >0.005745</td>
      <td id="T_a7b5f_row9_col27" class="data row9 col27" >-0.079079</td>
      <td id="T_a7b5f_row9_col28" class="data row9 col28" >-0.087227</td>
      <td id="T_a7b5f_row9_col29" class="data row9 col29" >-0.072229</td>
      <td id="T_a7b5f_row9_col30" class="data row9 col30" >-0.097536</td>
      <td id="T_a7b5f_row9_col31" class="data row9 col31" >-0.105882</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row10" class="row_heading level0 row10" >No. of aborptions</th>
      <td id="T_a7b5f_row10_col0" class="data row10 col0" >0.221843</td>
      <td id="T_a7b5f_row10_col1" class="data row10 col1" >0.093310</td>
      <td id="T_a7b5f_row10_col2" class="data row10 col2" >-0.026240</td>
      <td id="T_a7b5f_row10_col3" class="data row10 col3" >0.109865</td>
      <td id="T_a7b5f_row10_col4" class="data row10 col4" >0.046227</td>
      <td id="T_a7b5f_row10_col5" class="data row10 col5" >-0.006090</td>
      <td id="T_a7b5f_row10_col6" class="data row10 col6" >0.060702</td>
      <td id="T_a7b5f_row10_col7" class="data row10 col7" >-0.057937</td>
      <td id="T_a7b5f_row10_col8" class="data row10 col8" >0.004023</td>
      <td id="T_a7b5f_row10_col9" class="data row10 col9" >0.246713</td>
      <td id="T_a7b5f_row10_col10" class="data row10 col10" >1.000000</td>
      <td id="T_a7b5f_row10_col11" class="data row10 col11" >0.057762</td>
      <td id="T_a7b5f_row10_col12" class="data row10 col12" >0.046977</td>
      <td id="T_a7b5f_row10_col13" class="data row10 col13" >-0.018187</td>
      <td id="T_a7b5f_row10_col14" class="data row10 col14" >-0.018876</td>
      <td id="T_a7b5f_row10_col15" class="data row10 col15" >-0.026472</td>
      <td id="T_a7b5f_row10_col16" class="data row10 col16" >0.078418</td>
      <td id="T_a7b5f_row10_col17" class="data row10 col17" >0.073280</td>
      <td id="T_a7b5f_row10_col18" class="data row10 col18" >-0.004031</td>
      <td id="T_a7b5f_row10_col19" class="data row10 col19" >0.032359</td>
      <td id="T_a7b5f_row10_col20" class="data row10 col20" >-0.052920</td>
      <td id="T_a7b5f_row10_col21" class="data row10 col21" >-0.064655</td>
      <td id="T_a7b5f_row10_col22" class="data row10 col22" >0.015516</td>
      <td id="T_a7b5f_row10_col23" class="data row10 col23" >-0.023962</td>
      <td id="T_a7b5f_row10_col24" class="data row10 col24" >-0.013134</td>
      <td id="T_a7b5f_row10_col25" class="data row10 col25" >-0.083221</td>
      <td id="T_a7b5f_row10_col26" class="data row10 col26" >0.070745</td>
      <td id="T_a7b5f_row10_col27" class="data row10 col27" >-0.058061</td>
      <td id="T_a7b5f_row10_col28" class="data row10 col28" >-0.078688</td>
      <td id="T_a7b5f_row10_col29" class="data row10 col29" >-0.056589</td>
      <td id="T_a7b5f_row10_col30" class="data row10 col30" >-0.117573</td>
      <td id="T_a7b5f_row10_col31" class="data row10 col31" >-0.067758</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row11" class="row_heading level0 row11" >  I   beta-HCG(mIU/mL)</th>
      <td id="T_a7b5f_row11_col0" class="data row11 col0" >0.008511</td>
      <td id="T_a7b5f_row11_col1" class="data row11 col1" >0.015882</td>
      <td id="T_a7b5f_row11_col2" class="data row11 col2" >0.062079</td>
      <td id="T_a7b5f_row11_col3" class="data row11 col3" >-0.009967</td>
      <td id="T_a7b5f_row11_col4" class="data row11 col4" >-0.020437</td>
      <td id="T_a7b5f_row11_col5" class="data row11 col5" >-0.085028</td>
      <td id="T_a7b5f_row11_col6" class="data row11 col6" >-0.016649</td>
      <td id="T_a7b5f_row11_col7" class="data row11 col7" >0.063098</td>
      <td id="T_a7b5f_row11_col8" class="data row11 col8" >0.020187</td>
      <td id="T_a7b5f_row11_col9" class="data row11 col9" >0.111666</td>
      <td id="T_a7b5f_row11_col10" class="data row11 col10" >0.057762</td>
      <td id="T_a7b5f_row11_col11" class="data row11 col11" >1.000000</td>
      <td id="T_a7b5f_row11_col12" class="data row11 col12" >0.533687</td>
      <td id="T_a7b5f_row11_col13" class="data row11 col13" >0.001641</td>
      <td id="T_a7b5f_row11_col14" class="data row11 col14" >-0.006914</td>
      <td id="T_a7b5f_row11_col15" class="data row11 col15" >0.007892</td>
      <td id="T_a7b5f_row11_col16" class="data row11 col16" >0.014833</td>
      <td id="T_a7b5f_row11_col17" class="data row11 col17" >0.005173</td>
      <td id="T_a7b5f_row11_col18" class="data row11 col18" >-0.021490</td>
      <td id="T_a7b5f_row11_col19" class="data row11 col19" >-0.044710</td>
      <td id="T_a7b5f_row11_col20" class="data row11 col20" >0.014428</td>
      <td id="T_a7b5f_row11_col21" class="data row11 col21" >-0.013383</td>
      <td id="T_a7b5f_row11_col22" class="data row11 col22" >-0.012659</td>
      <td id="T_a7b5f_row11_col23" class="data row11 col23" >-0.002984</td>
      <td id="T_a7b5f_row11_col24" class="data row11 col24" >-0.023509</td>
      <td id="T_a7b5f_row11_col25" class="data row11 col25" >-0.081720</td>
      <td id="T_a7b5f_row11_col26" class="data row11 col26" >0.003923</td>
      <td id="T_a7b5f_row11_col27" class="data row11 col27" >0.048325</td>
      <td id="T_a7b5f_row11_col28" class="data row11 col28" >0.018266</td>
      <td id="T_a7b5f_row11_col29" class="data row11 col29" >0.050100</td>
      <td id="T_a7b5f_row11_col30" class="data row11 col30" >0.071863</td>
      <td id="T_a7b5f_row11_col31" class="data row11 col31" >-0.051920</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row12" class="row_heading level0 row12" >II    beta-HCG(mIU/mL)</th>
      <td id="T_a7b5f_row12_col0" class="data row12 col0" >0.043378</td>
      <td id="T_a7b5f_row12_col1" class="data row12 col1" >-0.000521</td>
      <td id="T_a7b5f_row12_col2" class="data row12 col2" >0.036474</td>
      <td id="T_a7b5f_row12_col3" class="data row12 col3" >-0.015270</td>
      <td id="T_a7b5f_row12_col4" class="data row12 col4" >-0.016226</td>
      <td id="T_a7b5f_row12_col5" class="data row12 col5" >-0.039017</td>
      <td id="T_a7b5f_row12_col6" class="data row12 col6" >-0.094425</td>
      <td id="T_a7b5f_row12_col7" class="data row12 col7" >0.027930</td>
      <td id="T_a7b5f_row12_col8" class="data row12 col8" >0.018577</td>
      <td id="T_a7b5f_row12_col9" class="data row12 col9" >0.112905</td>
      <td id="T_a7b5f_row12_col10" class="data row12 col10" >0.046977</td>
      <td id="T_a7b5f_row12_col11" class="data row12 col11" >0.533687</td>
      <td id="T_a7b5f_row12_col12" class="data row12 col12" >1.000000</td>
      <td id="T_a7b5f_row12_col13" class="data row12 col13" >0.016752</td>
      <td id="T_a7b5f_row12_col14" class="data row12 col14" >-0.008092</td>
      <td id="T_a7b5f_row12_col15" class="data row12 col15" >0.027150</td>
      <td id="T_a7b5f_row12_col16" class="data row12 col16" >0.015716</td>
      <td id="T_a7b5f_row12_col17" class="data row12 col17" >0.016541</td>
      <td id="T_a7b5f_row12_col18" class="data row12 col18" >0.001982</td>
      <td id="T_a7b5f_row12_col19" class="data row12 col19" >-0.036839</td>
      <td id="T_a7b5f_row12_col20" class="data row12 col20" >0.003668</td>
      <td id="T_a7b5f_row12_col21" class="data row12 col21" >-0.001337</td>
      <td id="T_a7b5f_row12_col22" class="data row12 col22" >-0.007552</td>
      <td id="T_a7b5f_row12_col23" class="data row12 col23" >0.005649</td>
      <td id="T_a7b5f_row12_col24" class="data row12 col24" >-0.004830</td>
      <td id="T_a7b5f_row12_col25" class="data row12 col25" >-0.059202</td>
      <td id="T_a7b5f_row12_col26" class="data row12 col26" >0.004021</td>
      <td id="T_a7b5f_row12_col27" class="data row12 col27" >0.064106</td>
      <td id="T_a7b5f_row12_col28" class="data row12 col28" >0.037524</td>
      <td id="T_a7b5f_row12_col29" class="data row12 col29" >0.000003</td>
      <td id="T_a7b5f_row12_col30" class="data row12 col30" >0.037088</td>
      <td id="T_a7b5f_row12_col31" class="data row12 col31" >0.017013</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row13" class="row_heading level0 row13" >FSH(mIU/mL)</th>
      <td id="T_a7b5f_row13_col0" class="data row13 col0" >-0.017708</td>
      <td id="T_a7b5f_row13_col1" class="data row13 col1" >-0.025785</td>
      <td id="T_a7b5f_row13_col2" class="data row13 col2" >0.030883</td>
      <td id="T_a7b5f_row13_col3" class="data row13 col3" >-0.040717</td>
      <td id="T_a7b5f_row13_col4" class="data row13 col4" >-0.013071</td>
      <td id="T_a7b5f_row13_col5" class="data row13 col5" >-0.032388</td>
      <td id="T_a7b5f_row13_col6" class="data row13 col6" >-0.047398</td>
      <td id="T_a7b5f_row13_col7" class="data row13 col7" >-0.026083</td>
      <td id="T_a7b5f_row13_col8" class="data row13 col8" >0.029645</td>
      <td id="T_a7b5f_row13_col9" class="data row13 col9" >-0.023461</td>
      <td id="T_a7b5f_row13_col10" class="data row13 col10" >-0.018187</td>
      <td id="T_a7b5f_row13_col11" class="data row13 col11" >0.001641</td>
      <td id="T_a7b5f_row13_col12" class="data row13 col12" >0.016752</td>
      <td id="T_a7b5f_row13_col13" class="data row13 col13" >1.000000</td>
      <td id="T_a7b5f_row13_col14" class="data row13 col14" >-0.001450</td>
      <td id="T_a7b5f_row13_col15" class="data row13 col15" >0.971951</td>
      <td id="T_a7b5f_row13_col16" class="data row13 col16" >-0.000158</td>
      <td id="T_a7b5f_row13_col17" class="data row13 col17" >0.013754</td>
      <td id="T_a7b5f_row13_col18" class="data row13 col18" >0.027308</td>
      <td id="T_a7b5f_row13_col19" class="data row13 col19" >-0.024805</td>
      <td id="T_a7b5f_row13_col20" class="data row13 col20" >-0.014568</td>
      <td id="T_a7b5f_row13_col21" class="data row13 col21" >0.017848</td>
      <td id="T_a7b5f_row13_col22" class="data row13 col22" >-0.000683</td>
      <td id="T_a7b5f_row13_col23" class="data row13 col23" >-0.003426</td>
      <td id="T_a7b5f_row13_col24" class="data row13 col24" >0.018798</td>
      <td id="T_a7b5f_row13_col25" class="data row13 col25" >-0.026830</td>
      <td id="T_a7b5f_row13_col26" class="data row13 col26" >0.023285</td>
      <td id="T_a7b5f_row13_col27" class="data row13 col27" >-0.002342</td>
      <td id="T_a7b5f_row13_col28" class="data row13 col28" >-0.025358</td>
      <td id="T_a7b5f_row13_col29" class="data row13 col29" >0.011547</td>
      <td id="T_a7b5f_row13_col30" class="data row13 col30" >0.020184</td>
      <td id="T_a7b5f_row13_col31" class="data row13 col31" >-0.049158</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row14" class="row_heading level0 row14" >LH(mIU/mL)</th>
      <td id="T_a7b5f_row14_col0" class="data row14 col0" >0.000603</td>
      <td id="T_a7b5f_row14_col1" class="data row14 col1" >-0.029910</td>
      <td id="T_a7b5f_row14_col2" class="data row14 col2" >-0.045619</td>
      <td id="T_a7b5f_row14_col3" class="data row14 col3" >-0.013312</td>
      <td id="T_a7b5f_row14_col4" class="data row14 col4" >-0.032315</td>
      <td id="T_a7b5f_row14_col5" class="data row14 col5" >-0.031211</td>
      <td id="T_a7b5f_row14_col6" class="data row14 col6" >-0.089107</td>
      <td id="T_a7b5f_row14_col7" class="data row14 col7" >-0.021393</td>
      <td id="T_a7b5f_row14_col8" class="data row14 col8" >-0.001686</td>
      <td id="T_a7b5f_row14_col9" class="data row14 col9" >0.035509</td>
      <td id="T_a7b5f_row14_col10" class="data row14 col10" >-0.018876</td>
      <td id="T_a7b5f_row14_col11" class="data row14 col11" >-0.006914</td>
      <td id="T_a7b5f_row14_col12" class="data row14 col12" >-0.008092</td>
      <td id="T_a7b5f_row14_col13" class="data row14 col13" >-0.001450</td>
      <td id="T_a7b5f_row14_col14" class="data row14 col14" >1.000000</td>
      <td id="T_a7b5f_row14_col15" class="data row14 col15" >-0.005724</td>
      <td id="T_a7b5f_row14_col16" class="data row14 col16" >-0.021727</td>
      <td id="T_a7b5f_row14_col17" class="data row14 col17" >-0.022304</td>
      <td id="T_a7b5f_row14_col18" class="data row14 col18" >-0.002980</td>
      <td id="T_a7b5f_row14_col19" class="data row14 col19" >-0.009398</td>
      <td id="T_a7b5f_row14_col20" class="data row14 col20" >0.021055</td>
      <td id="T_a7b5f_row14_col21" class="data row14 col21" >0.046904</td>
      <td id="T_a7b5f_row14_col22" class="data row14 col22" >-0.001543</td>
      <td id="T_a7b5f_row14_col23" class="data row14 col23" >-0.004080</td>
      <td id="T_a7b5f_row14_col24" class="data row14 col24" >-0.015216</td>
      <td id="T_a7b5f_row14_col25" class="data row14 col25" >-0.027699</td>
      <td id="T_a7b5f_row14_col26" class="data row14 col26" >0.023964</td>
      <td id="T_a7b5f_row14_col27" class="data row14 col27" >-0.001280</td>
      <td id="T_a7b5f_row14_col28" class="data row14 col28" >0.003430</td>
      <td id="T_a7b5f_row14_col29" class="data row14 col29" >0.035632</td>
      <td id="T_a7b5f_row14_col30" class="data row14 col30" >0.032495</td>
      <td id="T_a7b5f_row14_col31" class="data row14 col31" >0.010786</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row15" class="row_heading level0 row15" >FSH/LH</th>
      <td id="T_a7b5f_row15_col0" class="data row15 col0" >0.012432</td>
      <td id="T_a7b5f_row15_col1" class="data row15 col1" >-0.004831</td>
      <td id="T_a7b5f_row15_col2" class="data row15 col2" >0.022065</td>
      <td id="T_a7b5f_row15_col3" class="data row15 col3" >-0.012076</td>
      <td id="T_a7b5f_row15_col4" class="data row15 col4" >-0.013104</td>
      <td id="T_a7b5f_row15_col5" class="data row15 col5" >-0.043339</td>
      <td id="T_a7b5f_row15_col6" class="data row15 col6" >-0.039827</td>
      <td id="T_a7b5f_row15_col7" class="data row15 col7" >-0.016167</td>
      <td id="T_a7b5f_row15_col8" class="data row15 col8" >0.025937</td>
      <td id="T_a7b5f_row15_col9" class="data row15 col9" >-0.002983</td>
      <td id="T_a7b5f_row15_col10" class="data row15 col10" >-0.026472</td>
      <td id="T_a7b5f_row15_col11" class="data row15 col11" >0.007892</td>
      <td id="T_a7b5f_row15_col12" class="data row15 col12" >0.027150</td>
      <td id="T_a7b5f_row15_col13" class="data row15 col13" >0.971951</td>
      <td id="T_a7b5f_row15_col14" class="data row15 col14" >-0.005724</td>
      <td id="T_a7b5f_row15_col15" class="data row15 col15" >1.000000</td>
      <td id="T_a7b5f_row15_col16" class="data row15 col16" >0.012370</td>
      <td id="T_a7b5f_row15_col17" class="data row15 col17" >0.027588</td>
      <td id="T_a7b5f_row15_col18" class="data row15 col18" >0.029240</td>
      <td id="T_a7b5f_row15_col19" class="data row15 col19" >-0.028331</td>
      <td id="T_a7b5f_row15_col20" class="data row15 col20" >-0.007438</td>
      <td id="T_a7b5f_row15_col21" class="data row15 col21" >0.015264</td>
      <td id="T_a7b5f_row15_col22" class="data row15 col22" >-0.001750</td>
      <td id="T_a7b5f_row15_col23" class="data row15 col23" >-0.005109</td>
      <td id="T_a7b5f_row15_col24" class="data row15 col24" >0.006231</td>
      <td id="T_a7b5f_row15_col25" class="data row15 col25" >-0.019302</td>
      <td id="T_a7b5f_row15_col26" class="data row15 col26" >0.026920</td>
      <td id="T_a7b5f_row15_col27" class="data row15 col27" >0.005872</td>
      <td id="T_a7b5f_row15_col28" class="data row15 col28" >-0.007506</td>
      <td id="T_a7b5f_row15_col29" class="data row15 col29" >0.014923</td>
      <td id="T_a7b5f_row15_col30" class="data row15 col30" >0.024070</td>
      <td id="T_a7b5f_row15_col31" class="data row15 col31" >-0.053768</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row16" class="row_heading level0 row16" >Hip(inch)</th>
      <td id="T_a7b5f_row16_col0" class="data row16 col0" >-0.001346</td>
      <td id="T_a7b5f_row16_col1" class="data row16 col1" >0.633920</td>
      <td id="T_a7b5f_row16_col2" class="data row16 col2" >0.215357</td>
      <td id="T_a7b5f_row16_col3" class="data row16 col3" >0.597058</td>
      <td id="T_a7b5f_row16_col4" class="data row16 col4" >0.062951</td>
      <td id="T_a7b5f_row16_col5" class="data row16 col5" >0.075020</td>
      <td id="T_a7b5f_row16_col6" class="data row16 col6" >-0.024712</td>
      <td id="T_a7b5f_row16_col7" class="data row16 col7" >0.174305</td>
      <td id="T_a7b5f_row16_col8" class="data row16 col8" >0.040354</td>
      <td id="T_a7b5f_row16_col9" class="data row16 col9" >0.038325</td>
      <td id="T_a7b5f_row16_col10" class="data row16 col10" >0.078418</td>
      <td id="T_a7b5f_row16_col11" class="data row16 col11" >0.014833</td>
      <td id="T_a7b5f_row16_col12" class="data row16 col12" >0.015716</td>
      <td id="T_a7b5f_row16_col13" class="data row16 col13" >-0.000158</td>
      <td id="T_a7b5f_row16_col14" class="data row16 col14" >-0.021727</td>
      <td id="T_a7b5f_row16_col15" class="data row16 col15" >0.012370</td>
      <td id="T_a7b5f_row16_col16" class="data row16 col16" >1.000000</td>
      <td id="T_a7b5f_row16_col17" class="data row16 col17" >0.873589</td>
      <td id="T_a7b5f_row16_col18" class="data row16 col18" >-0.241990</td>
      <td id="T_a7b5f_row16_col19" class="data row16 col19" >0.035835</td>
      <td id="T_a7b5f_row16_col20" class="data row16 col20" >-0.066106</td>
      <td id="T_a7b5f_row16_col21" class="data row16 col21" >-0.082153</td>
      <td id="T_a7b5f_row16_col22" class="data row16 col22" >0.009742</td>
      <td id="T_a7b5f_row16_col23" class="data row16 col23" >0.050008</td>
      <td id="T_a7b5f_row16_col24" class="data row16 col24" >-0.001393</td>
      <td id="T_a7b5f_row16_col25" class="data row16 col25" >-0.012106</td>
      <td id="T_a7b5f_row16_col26" class="data row16 col26" >0.104944</td>
      <td id="T_a7b5f_row16_col27" class="data row16 col27" >0.119505</td>
      <td id="T_a7b5f_row16_col28" class="data row16 col28" >0.094799</td>
      <td id="T_a7b5f_row16_col29" class="data row16 col29" >-0.082276</td>
      <td id="T_a7b5f_row16_col30" class="data row16 col30" >-0.104384</td>
      <td id="T_a7b5f_row16_col31" class="data row16 col31" >0.027778</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row17" class="row_heading level0 row17" >Waist(inch)</th>
      <td id="T_a7b5f_row17_col0" class="data row17 col0" >0.036263</td>
      <td id="T_a7b5f_row17_col1" class="data row17 col1" >0.639589</td>
      <td id="T_a7b5f_row17_col2" class="data row17 col2" >0.209348</td>
      <td id="T_a7b5f_row17_col3" class="data row17 col3" >0.607524</td>
      <td id="T_a7b5f_row17_col4" class="data row17 col4" >0.037880</td>
      <td id="T_a7b5f_row17_col5" class="data row17 col5" >0.038310</td>
      <td id="T_a7b5f_row17_col6" class="data row17 col6" >-0.000779</td>
      <td id="T_a7b5f_row17_col7" class="data row17 col7" >0.168856</td>
      <td id="T_a7b5f_row17_col8" class="data row17 col8" >-0.023479</td>
      <td id="T_a7b5f_row17_col9" class="data row17 col9" >0.057376</td>
      <td id="T_a7b5f_row17_col10" class="data row17 col10" >0.073280</td>
      <td id="T_a7b5f_row17_col11" class="data row17 col11" >0.005173</td>
      <td id="T_a7b5f_row17_col12" class="data row17 col12" >0.016541</td>
      <td id="T_a7b5f_row17_col13" class="data row17 col13" >0.013754</td>
      <td id="T_a7b5f_row17_col14" class="data row17 col14" >-0.022304</td>
      <td id="T_a7b5f_row17_col15" class="data row17 col15" >0.027588</td>
      <td id="T_a7b5f_row17_col16" class="data row17 col16" >0.873589</td>
      <td id="T_a7b5f_row17_col17" class="data row17 col17" >1.000000</td>
      <td id="T_a7b5f_row17_col18" class="data row17 col18" >0.258365</td>
      <td id="T_a7b5f_row17_col19" class="data row17 col19" >-0.008295</td>
      <td id="T_a7b5f_row17_col20" class="data row17 col20" >-0.043132</td>
      <td id="T_a7b5f_row17_col21" class="data row17 col21" >-0.036364</td>
      <td id="T_a7b5f_row17_col22" class="data row17 col22" >0.018989</td>
      <td id="T_a7b5f_row17_col23" class="data row17 col23" >0.029175</td>
      <td id="T_a7b5f_row17_col24" class="data row17 col24" >0.012995</td>
      <td id="T_a7b5f_row17_col25" class="data row17 col25" >0.034558</td>
      <td id="T_a7b5f_row17_col26" class="data row17 col26" >0.125243</td>
      <td id="T_a7b5f_row17_col27" class="data row17 col27" >0.129846</td>
      <td id="T_a7b5f_row17_col28" class="data row17 col28" >0.093401</td>
      <td id="T_a7b5f_row17_col29" class="data row17 col29" >0.031048</td>
      <td id="T_a7b5f_row17_col30" class="data row17 col30" >-0.015271</td>
      <td id="T_a7b5f_row17_col31" class="data row17 col31" >0.012810</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row18" class="row_heading level0 row18" >Waist:Hip Ratio</th>
      <td id="T_a7b5f_row18_col0" class="data row18 col0" >0.066452</td>
      <td id="T_a7b5f_row18_col1" class="data row18 col1" >0.014976</td>
      <td id="T_a7b5f_row18_col2" class="data row18 col2" >-0.008982</td>
      <td id="T_a7b5f_row18_col3" class="data row18 col3" >0.023602</td>
      <td id="T_a7b5f_row18_col4" class="data row18 col4" >-0.052890</td>
      <td id="T_a7b5f_row18_col5" class="data row18 col5" >-0.075567</td>
      <td id="T_a7b5f_row18_col6" class="data row18 col6" >0.057161</td>
      <td id="T_a7b5f_row18_col7" class="data row18 col7" >-0.004003</td>
      <td id="T_a7b5f_row18_col8" class="data row18 col8" >-0.130093</td>
      <td id="T_a7b5f_row18_col9" class="data row18 col9" >0.033334</td>
      <td id="T_a7b5f_row18_col10" class="data row18 col10" >-0.004031</td>
      <td id="T_a7b5f_row18_col11" class="data row18 col11" >-0.021490</td>
      <td id="T_a7b5f_row18_col12" class="data row18 col12" >0.001982</td>
      <td id="T_a7b5f_row18_col13" class="data row18 col13" >0.027308</td>
      <td id="T_a7b5f_row18_col14" class="data row18 col14" >-0.002980</td>
      <td id="T_a7b5f_row18_col15" class="data row18 col15" >0.029240</td>
      <td id="T_a7b5f_row18_col16" class="data row18 col16" >-0.241990</td>
      <td id="T_a7b5f_row18_col17" class="data row18 col17" >0.258365</td>
      <td id="T_a7b5f_row18_col18" class="data row18 col18" >1.000000</td>
      <td id="T_a7b5f_row18_col19" class="data row18 col19" >-0.085023</td>
      <td id="T_a7b5f_row18_col20" class="data row18 col20" >0.057354</td>
      <td id="T_a7b5f_row18_col21" class="data row18 col21" >0.092356</td>
      <td id="T_a7b5f_row18_col22" class="data row18 col22" >0.017701</td>
      <td id="T_a7b5f_row18_col23" class="data row18 col23" >-0.039042</td>
      <td id="T_a7b5f_row18_col24" class="data row18 col24" >0.023170</td>
      <td id="T_a7b5f_row18_col25" class="data row18 col25" >0.088978</td>
      <td id="T_a7b5f_row18_col26" class="data row18 col26" >0.042507</td>
      <td id="T_a7b5f_row18_col27" class="data row18 col27" >0.026456</td>
      <td id="T_a7b5f_row18_col28" class="data row18 col28" >-0.002207</td>
      <td id="T_a7b5f_row18_col29" class="data row18 col29" >0.229355</td>
      <td id="T_a7b5f_row18_col30" class="data row18 col30" >0.176453</td>
      <td id="T_a7b5f_row18_col31" class="data row18 col31" >-0.026793</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row19" class="row_heading level0 row19" >TSH (mIU/L)</th>
      <td id="T_a7b5f_row19_col0" class="data row19 col0" >0.010245</td>
      <td id="T_a7b5f_row19_col1" class="data row19 col1" >0.071410</td>
      <td id="T_a7b5f_row19_col2" class="data row19 col2" >0.018501</td>
      <td id="T_a7b5f_row19_col3" class="data row19 col3" >0.072305</td>
      <td id="T_a7b5f_row19_col4" class="data row19 col4" >-0.051482</td>
      <td id="T_a7b5f_row19_col5" class="data row19 col5" >-0.011968</td>
      <td id="T_a7b5f_row19_col6" class="data row19 col6" >-0.026156</td>
      <td id="T_a7b5f_row19_col7" class="data row19 col7" >-0.017435</td>
      <td id="T_a7b5f_row19_col8" class="data row19 col8" >-0.003191</td>
      <td id="T_a7b5f_row19_col9" class="data row19 col9" >-0.040241</td>
      <td id="T_a7b5f_row19_col10" class="data row19 col10" >0.032359</td>
      <td id="T_a7b5f_row19_col11" class="data row19 col11" >-0.044710</td>
      <td id="T_a7b5f_row19_col12" class="data row19 col12" >-0.036839</td>
      <td id="T_a7b5f_row19_col13" class="data row19 col13" >-0.024805</td>
      <td id="T_a7b5f_row19_col14" class="data row19 col14" >-0.009398</td>
      <td id="T_a7b5f_row19_col15" class="data row19 col15" >-0.028331</td>
      <td id="T_a7b5f_row19_col16" class="data row19 col16" >0.035835</td>
      <td id="T_a7b5f_row19_col17" class="data row19 col17" >-0.008295</td>
      <td id="T_a7b5f_row19_col18" class="data row19 col18" >-0.085023</td>
      <td id="T_a7b5f_row19_col19" class="data row19 col19" >1.000000</td>
      <td id="T_a7b5f_row19_col20" class="data row19 col20" >-0.010854</td>
      <td id="T_a7b5f_row19_col21" class="data row19 col21" >0.025606</td>
      <td id="T_a7b5f_row19_col22" class="data row19 col22" >-0.009114</td>
      <td id="T_a7b5f_row19_col23" class="data row19 col23" >-0.020504</td>
      <td id="T_a7b5f_row19_col24" class="data row19 col24" >-0.030056</td>
      <td id="T_a7b5f_row19_col25" class="data row19 col25" >0.048215</td>
      <td id="T_a7b5f_row19_col26" class="data row19 col26" >0.055034</td>
      <td id="T_a7b5f_row19_col27" class="data row19 col27" >-0.028164</td>
      <td id="T_a7b5f_row19_col28" class="data row19 col28" >-0.016807</td>
      <td id="T_a7b5f_row19_col29" class="data row19 col29" >-0.091201</td>
      <td id="T_a7b5f_row19_col30" class="data row19 col30" >-0.088850</td>
      <td id="T_a7b5f_row19_col31" class="data row19 col31" >0.013863</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row20" class="row_heading level0 row20" >AMH(ng/mL)</th>
      <td id="T_a7b5f_row20_col0" class="data row20 col0" >-0.179648</td>
      <td id="T_a7b5f_row20_col1" class="data row20 col1" >0.031050</td>
      <td id="T_a7b5f_row20_col2" class="data row20 col2" >-0.045208</td>
      <td id="T_a7b5f_row20_col3" class="data row20 col3" >0.054023</td>
      <td id="T_a7b5f_row20_col4" class="data row20 col4" >-0.049843</td>
      <td id="T_a7b5f_row20_col5" class="data row20 col5" >-0.017480</td>
      <td id="T_a7b5f_row20_col6" class="data row20 col6" >0.056172</td>
      <td id="T_a7b5f_row20_col7" class="data row20 col7" >0.194063</td>
      <td id="T_a7b5f_row20_col8" class="data row20 col8" >-0.072822</td>
      <td id="T_a7b5f_row20_col9" class="data row20 col9" >-0.146482</td>
      <td id="T_a7b5f_row20_col10" class="data row20 col10" >-0.052920</td>
      <td id="T_a7b5f_row20_col11" class="data row20 col11" >0.014428</td>
      <td id="T_a7b5f_row20_col12" class="data row20 col12" >0.003668</td>
      <td id="T_a7b5f_row20_col13" class="data row20 col13" >-0.014568</td>
      <td id="T_a7b5f_row20_col14" class="data row20 col14" >0.021055</td>
      <td id="T_a7b5f_row20_col15" class="data row20 col15" >-0.007438</td>
      <td id="T_a7b5f_row20_col16" class="data row20 col16" >-0.066106</td>
      <td id="T_a7b5f_row20_col17" class="data row20 col17" >-0.043132</td>
      <td id="T_a7b5f_row20_col18" class="data row20 col18" >0.057354</td>
      <td id="T_a7b5f_row20_col19" class="data row20 col19" >-0.010854</td>
      <td id="T_a7b5f_row20_col20" class="data row20 col20" >1.000000</td>
      <td id="T_a7b5f_row20_col21" class="data row20 col21" >-0.072214</td>
      <td id="T_a7b5f_row20_col22" class="data row20 col22" >0.007660</td>
      <td id="T_a7b5f_row20_col23" class="data row20 col23" >-0.011195</td>
      <td id="T_a7b5f_row20_col24" class="data row20 col24" >0.008454</td>
      <td id="T_a7b5f_row20_col25" class="data row20 col25" >-0.003192</td>
      <td id="T_a7b5f_row20_col26" class="data row20 col26" >0.026038</td>
      <td id="T_a7b5f_row20_col27" class="data row20 col27" >0.203407</td>
      <td id="T_a7b5f_row20_col28" class="data row20 col28" >0.188537</td>
      <td id="T_a7b5f_row20_col29" class="data row20 col29" >0.134628</td>
      <td id="T_a7b5f_row20_col30" class="data row20 col30" >0.096079</td>
      <td id="T_a7b5f_row20_col31" class="data row20 col31" >0.104159</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row21" class="row_heading level0 row21" >PRL(ng/mL)</th>
      <td id="T_a7b5f_row21_col0" class="data row21 col0" >-0.046077</td>
      <td id="T_a7b5f_row21_col1" class="data row21 col1" >-0.050016</td>
      <td id="T_a7b5f_row21_col2" class="data row21 col2" >-0.018178</td>
      <td id="T_a7b5f_row21_col3" class="data row21 col3" >-0.047721</td>
      <td id="T_a7b5f_row21_col4" class="data row21 col4" >0.020750</td>
      <td id="T_a7b5f_row21_col5" class="data row21 col5" >0.007284</td>
      <td id="T_a7b5f_row21_col6" class="data row21 col6" >-0.063091</td>
      <td id="T_a7b5f_row21_col7" class="data row21 col7" >0.005251</td>
      <td id="T_a7b5f_row21_col8" class="data row21 col8" >0.016415</td>
      <td id="T_a7b5f_row21_col9" class="data row21 col9" >-0.025819</td>
      <td id="T_a7b5f_row21_col10" class="data row21 col10" >-0.064655</td>
      <td id="T_a7b5f_row21_col11" class="data row21 col11" >-0.013383</td>
      <td id="T_a7b5f_row21_col12" class="data row21 col12" >-0.001337</td>
      <td id="T_a7b5f_row21_col13" class="data row21 col13" >0.017848</td>
      <td id="T_a7b5f_row21_col14" class="data row21 col14" >0.046904</td>
      <td id="T_a7b5f_row21_col15" class="data row21 col15" >0.015264</td>
      <td id="T_a7b5f_row21_col16" class="data row21 col16" >-0.082153</td>
      <td id="T_a7b5f_row21_col17" class="data row21 col17" >-0.036364</td>
      <td id="T_a7b5f_row21_col18" class="data row21 col18" >0.092356</td>
      <td id="T_a7b5f_row21_col19" class="data row21 col19" >0.025606</td>
      <td id="T_a7b5f_row21_col20" class="data row21 col20" >-0.072214</td>
      <td id="T_a7b5f_row21_col21" class="data row21 col21" >1.000000</td>
      <td id="T_a7b5f_row21_col22" class="data row21 col22" >-0.006827</td>
      <td id="T_a7b5f_row21_col23" class="data row21 col23" >-0.009700</td>
      <td id="T_a7b5f_row21_col24" class="data row21 col24" >-0.041676</td>
      <td id="T_a7b5f_row21_col25" class="data row21 col25" >-0.009963</td>
      <td id="T_a7b5f_row21_col26" class="data row21 col26" >-0.026821</td>
      <td id="T_a7b5f_row21_col27" class="data row21 col27" >-0.010803</td>
      <td id="T_a7b5f_row21_col28" class="data row21 col28" >-0.008607</td>
      <td id="T_a7b5f_row21_col29" class="data row21 col29" >0.086311</td>
      <td id="T_a7b5f_row21_col30" class="data row21 col30" >0.071688</td>
      <td id="T_a7b5f_row21_col31" class="data row21 col31" >0.027439</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row22" class="row_heading level0 row22" >Vit D3 (ng/mL)</th>
      <td id="T_a7b5f_row22_col0" class="data row22 col0" >0.004454</td>
      <td id="T_a7b5f_row22_col1" class="data row22 col1" >0.008145</td>
      <td id="T_a7b5f_row22_col2" class="data row22 col2" >-0.034997</td>
      <td id="T_a7b5f_row22_col3" class="data row22 col3" >0.027035</td>
      <td id="T_a7b5f_row22_col4" class="data row22 col4" >-0.001486</td>
      <td id="T_a7b5f_row22_col5" class="data row22 col5" >-0.009016</td>
      <td id="T_a7b5f_row22_col6" class="data row22 col6" >0.063916</td>
      <td id="T_a7b5f_row22_col7" class="data row22 col7" >0.096717</td>
      <td id="T_a7b5f_row22_col8" class="data row22 col8" >-0.058345</td>
      <td id="T_a7b5f_row22_col9" class="data row22 col9" >0.041541</td>
      <td id="T_a7b5f_row22_col10" class="data row22 col10" >0.015516</td>
      <td id="T_a7b5f_row22_col11" class="data row22 col11" >-0.012659</td>
      <td id="T_a7b5f_row22_col12" class="data row22 col12" >-0.007552</td>
      <td id="T_a7b5f_row22_col13" class="data row22 col13" >-0.000683</td>
      <td id="T_a7b5f_row22_col14" class="data row22 col14" >-0.001543</td>
      <td id="T_a7b5f_row22_col15" class="data row22 col15" >-0.001750</td>
      <td id="T_a7b5f_row22_col16" class="data row22 col16" >0.009742</td>
      <td id="T_a7b5f_row22_col17" class="data row22 col17" >0.018989</td>
      <td id="T_a7b5f_row22_col18" class="data row22 col18" >0.017701</td>
      <td id="T_a7b5f_row22_col19" class="data row22 col19" >-0.009114</td>
      <td id="T_a7b5f_row22_col20" class="data row22 col20" >0.007660</td>
      <td id="T_a7b5f_row22_col21" class="data row22 col21" >-0.006827</td>
      <td id="T_a7b5f_row22_col22" class="data row22 col22" >1.000000</td>
      <td id="T_a7b5f_row22_col23" class="data row22 col23" >-0.007240</td>
      <td id="T_a7b5f_row22_col24" class="data row22 col24" >0.038980</td>
      <td id="T_a7b5f_row22_col25" class="data row22 col25" >0.042855</td>
      <td id="T_a7b5f_row22_col26" class="data row22 col26" >-0.024962</td>
      <td id="T_a7b5f_row22_col27" class="data row22 col27" >0.074172</td>
      <td id="T_a7b5f_row22_col28" class="data row22 col28" >0.091775</td>
      <td id="T_a7b5f_row22_col29" class="data row22 col29" >0.005441</td>
      <td id="T_a7b5f_row22_col30" class="data row22 col30" >-0.011812</td>
      <td id="T_a7b5f_row22_col31" class="data row22 col31" >-0.031385</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row23" class="row_heading level0 row23" >PRG(ng/mL)</th>
      <td id="T_a7b5f_row23_col0" class="data row23 col0" >-0.021775</td>
      <td id="T_a7b5f_row23_col1" class="data row23 col1" >0.069688</td>
      <td id="T_a7b5f_row23_col2" class="data row23 col2" >0.049654</td>
      <td id="T_a7b5f_row23_col3" class="data row23 col3" >0.049460</td>
      <td id="T_a7b5f_row23_col4" class="data row23 col4" >-0.017678</td>
      <td id="T_a7b5f_row23_col5" class="data row23 col5" >-0.076895</td>
      <td id="T_a7b5f_row23_col6" class="data row23 col6" >0.065769</td>
      <td id="T_a7b5f_row23_col7" class="data row23 col7" >-0.033781</td>
      <td id="T_a7b5f_row23_col8" class="data row23 col8" >0.007190</td>
      <td id="T_a7b5f_row23_col9" class="data row23 col9" >-0.052840</td>
      <td id="T_a7b5f_row23_col10" class="data row23 col10" >-0.023962</td>
      <td id="T_a7b5f_row23_col11" class="data row23 col11" >-0.002984</td>
      <td id="T_a7b5f_row23_col12" class="data row23 col12" >0.005649</td>
      <td id="T_a7b5f_row23_col13" class="data row23 col13" >-0.003426</td>
      <td id="T_a7b5f_row23_col14" class="data row23 col14" >-0.004080</td>
      <td id="T_a7b5f_row23_col15" class="data row23 col15" >-0.005109</td>
      <td id="T_a7b5f_row23_col16" class="data row23 col16" >0.050008</td>
      <td id="T_a7b5f_row23_col17" class="data row23 col17" >0.029175</td>
      <td id="T_a7b5f_row23_col18" class="data row23 col18" >-0.039042</td>
      <td id="T_a7b5f_row23_col19" class="data row23 col19" >-0.020504</td>
      <td id="T_a7b5f_row23_col20" class="data row23 col20" >-0.011195</td>
      <td id="T_a7b5f_row23_col21" class="data row23 col21" >-0.009700</td>
      <td id="T_a7b5f_row23_col22" class="data row23 col22" >-0.007240</td>
      <td id="T_a7b5f_row23_col23" class="data row23 col23" >1.000000</td>
      <td id="T_a7b5f_row23_col24" class="data row23 col24" >-0.003545</td>
      <td id="T_a7b5f_row23_col25" class="data row23 col25" >-0.020638</td>
      <td id="T_a7b5f_row23_col26" class="data row23 col26" >0.032131</td>
      <td id="T_a7b5f_row23_col27" class="data row23 col27" >0.018489</td>
      <td id="T_a7b5f_row23_col28" class="data row23 col28" >0.031483</td>
      <td id="T_a7b5f_row23_col29" class="data row23 col29" >-0.040698</td>
      <td id="T_a7b5f_row23_col30" class="data row23 col30" >-0.040910</td>
      <td id="T_a7b5f_row23_col31" class="data row23 col31" >-0.047987</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row24" class="row_heading level0 row24" >RBS(mg/dl)</th>
      <td id="T_a7b5f_row24_col0" class="data row24 col0" >0.097086</td>
      <td id="T_a7b5f_row24_col1" class="data row24 col1" >0.114294</td>
      <td id="T_a7b5f_row24_col2" class="data row24 col2" >0.050437</td>
      <td id="T_a7b5f_row24_col3" class="data row24 col3" >0.093543</td>
      <td id="T_a7b5f_row24_col4" class="data row24 col4" >0.042002</td>
      <td id="T_a7b5f_row24_col5" class="data row24 col5" >0.050811</td>
      <td id="T_a7b5f_row24_col6" class="data row24 col6" >0.024067</td>
      <td id="T_a7b5f_row24_col7" class="data row24 col7" >-0.006394</td>
      <td id="T_a7b5f_row24_col8" class="data row24 col8" >0.000210</td>
      <td id="T_a7b5f_row24_col9" class="data row24 col9" >0.075519</td>
      <td id="T_a7b5f_row24_col10" class="data row24 col10" >-0.013134</td>
      <td id="T_a7b5f_row24_col11" class="data row24 col11" >-0.023509</td>
      <td id="T_a7b5f_row24_col12" class="data row24 col12" >-0.004830</td>
      <td id="T_a7b5f_row24_col13" class="data row24 col13" >0.018798</td>
      <td id="T_a7b5f_row24_col14" class="data row24 col14" >-0.015216</td>
      <td id="T_a7b5f_row24_col15" class="data row24 col15" >0.006231</td>
      <td id="T_a7b5f_row24_col16" class="data row24 col16" >-0.001393</td>
      <td id="T_a7b5f_row24_col17" class="data row24 col17" >0.012995</td>
      <td id="T_a7b5f_row24_col18" class="data row24 col18" >0.023170</td>
      <td id="T_a7b5f_row24_col19" class="data row24 col19" >-0.030056</td>
      <td id="T_a7b5f_row24_col20" class="data row24 col20" >0.008454</td>
      <td id="T_a7b5f_row24_col21" class="data row24 col21" >-0.041676</td>
      <td id="T_a7b5f_row24_col22" class="data row24 col22" >0.038980</td>
      <td id="T_a7b5f_row24_col23" class="data row24 col23" >-0.003545</td>
      <td id="T_a7b5f_row24_col24" class="data row24 col24" >1.000000</td>
      <td id="T_a7b5f_row24_col25" class="data row24 col25" >0.052683</td>
      <td id="T_a7b5f_row24_col26" class="data row24 col26" >-0.032907</td>
      <td id="T_a7b5f_row24_col27" class="data row24 col27" >0.044342</td>
      <td id="T_a7b5f_row24_col28" class="data row24 col28" >0.027562</td>
      <td id="T_a7b5f_row24_col29" class="data row24 col29" >-0.046821</td>
      <td id="T_a7b5f_row24_col30" class="data row24 col30" >0.013137</td>
      <td id="T_a7b5f_row24_col31" class="data row24 col31" >-0.018700</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row25" class="row_heading level0 row25" >BP _Systolic (mmHg)</th>
      <td id="T_a7b5f_row25_col0" class="data row25 col0" >0.072313</td>
      <td id="T_a7b5f_row25_col1" class="data row25 col1" >0.028066</td>
      <td id="T_a7b5f_row25_col2" class="data row25 col2" >-0.067029</td>
      <td id="T_a7b5f_row25_col3" class="data row25 col3" >0.069549</td>
      <td id="T_a7b5f_row25_col4" class="data row25 col4" >-0.025751</td>
      <td id="T_a7b5f_row25_col5" class="data row25 col5" >0.016734</td>
      <td id="T_a7b5f_row25_col6" class="data row25 col6" >0.052229</td>
      <td id="T_a7b5f_row25_col7" class="data row25 col7" >0.055790</td>
      <td id="T_a7b5f_row25_col8" class="data row25 col8" >-0.011963</td>
      <td id="T_a7b5f_row25_col9" class="data row25 col9" >0.028478</td>
      <td id="T_a7b5f_row25_col10" class="data row25 col10" >-0.083221</td>
      <td id="T_a7b5f_row25_col11" class="data row25 col11" >-0.081720</td>
      <td id="T_a7b5f_row25_col12" class="data row25 col12" >-0.059202</td>
      <td id="T_a7b5f_row25_col13" class="data row25 col13" >-0.026830</td>
      <td id="T_a7b5f_row25_col14" class="data row25 col14" >-0.027699</td>
      <td id="T_a7b5f_row25_col15" class="data row25 col15" >-0.019302</td>
      <td id="T_a7b5f_row25_col16" class="data row25 col16" >-0.012106</td>
      <td id="T_a7b5f_row25_col17" class="data row25 col17" >0.034558</td>
      <td id="T_a7b5f_row25_col18" class="data row25 col18" >0.088978</td>
      <td id="T_a7b5f_row25_col19" class="data row25 col19" >0.048215</td>
      <td id="T_a7b5f_row25_col20" class="data row25 col20" >-0.003192</td>
      <td id="T_a7b5f_row25_col21" class="data row25 col21" >-0.009963</td>
      <td id="T_a7b5f_row25_col22" class="data row25 col22" >0.042855</td>
      <td id="T_a7b5f_row25_col23" class="data row25 col23" >-0.020638</td>
      <td id="T_a7b5f_row25_col24" class="data row25 col24" >0.052683</td>
      <td id="T_a7b5f_row25_col25" class="data row25 col25" >1.000000</td>
      <td id="T_a7b5f_row25_col26" class="data row25 col26" >0.102090</td>
      <td id="T_a7b5f_row25_col27" class="data row25 col27" >0.039463</td>
      <td id="T_a7b5f_row25_col28" class="data row25 col28" >0.025515</td>
      <td id="T_a7b5f_row25_col29" class="data row25 col29" >0.051273</td>
      <td id="T_a7b5f_row25_col30" class="data row25 col30" >0.038448</td>
      <td id="T_a7b5f_row25_col31" class="data row25 col31" >-0.018613</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row26" class="row_heading level0 row26" >BP _Diastolic (mmHg)</th>
      <td id="T_a7b5f_row26_col0" class="data row26 col0" >0.069329</td>
      <td id="T_a7b5f_row26_col1" class="data row26 col1" >0.130843</td>
      <td id="T_a7b5f_row26_col2" class="data row26 col2" >0.009420</td>
      <td id="T_a7b5f_row26_col3" class="data row26 col3" >0.140134</td>
      <td id="T_a7b5f_row26_col4" class="data row26 col4" >0.008027</td>
      <td id="T_a7b5f_row26_col5" class="data row26 col5" >0.053751</td>
      <td id="T_a7b5f_row26_col6" class="data row26 col6" >0.002046</td>
      <td id="T_a7b5f_row26_col7" class="data row26 col7" >0.080057</td>
      <td id="T_a7b5f_row26_col8" class="data row26 col8" >-0.075792</td>
      <td id="T_a7b5f_row26_col9" class="data row26 col9" >0.005745</td>
      <td id="T_a7b5f_row26_col10" class="data row26 col10" >0.070745</td>
      <td id="T_a7b5f_row26_col11" class="data row26 col11" >0.003923</td>
      <td id="T_a7b5f_row26_col12" class="data row26 col12" >0.004021</td>
      <td id="T_a7b5f_row26_col13" class="data row26 col13" >0.023285</td>
      <td id="T_a7b5f_row26_col14" class="data row26 col14" >0.023964</td>
      <td id="T_a7b5f_row26_col15" class="data row26 col15" >0.026920</td>
      <td id="T_a7b5f_row26_col16" class="data row26 col16" >0.104944</td>
      <td id="T_a7b5f_row26_col17" class="data row26 col17" >0.125243</td>
      <td id="T_a7b5f_row26_col18" class="data row26 col18" >0.042507</td>
      <td id="T_a7b5f_row26_col19" class="data row26 col19" >0.055034</td>
      <td id="T_a7b5f_row26_col20" class="data row26 col20" >0.026038</td>
      <td id="T_a7b5f_row26_col21" class="data row26 col21" >-0.026821</td>
      <td id="T_a7b5f_row26_col22" class="data row26 col22" >-0.024962</td>
      <td id="T_a7b5f_row26_col23" class="data row26 col23" >0.032131</td>
      <td id="T_a7b5f_row26_col24" class="data row26 col24" >-0.032907</td>
      <td id="T_a7b5f_row26_col25" class="data row26 col25" >0.102090</td>
      <td id="T_a7b5f_row26_col26" class="data row26 col26" >1.000000</td>
      <td id="T_a7b5f_row26_col27" class="data row26 col27" >0.025043</td>
      <td id="T_a7b5f_row26_col28" class="data row26 col28" >0.037844</td>
      <td id="T_a7b5f_row26_col29" class="data row26 col29" >0.037028</td>
      <td id="T_a7b5f_row26_col30" class="data row26 col30" >0.024013</td>
      <td id="T_a7b5f_row26_col31" class="data row26 col31" >-0.015342</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row27" class="row_heading level0 row27" >Follicle No. (L)</th>
      <td id="T_a7b5f_row27_col0" class="data row27 col0" >-0.109965</td>
      <td id="T_a7b5f_row27_col1" class="data row27 col1" >0.173501</td>
      <td id="T_a7b5f_row27_col2" class="data row27 col2" >0.105574</td>
      <td id="T_a7b5f_row27_col3" class="data row27 col3" >0.142903</td>
      <td id="T_a7b5f_row27_col4" class="data row27 col4" >0.040559</td>
      <td id="T_a7b5f_row27_col5" class="data row27 col5" >0.070179</td>
      <td id="T_a7b5f_row27_col6" class="data row27 col6" >0.061814</td>
      <td id="T_a7b5f_row27_col7" class="data row27 col7" >0.296114</td>
      <td id="T_a7b5f_row27_col8" class="data row27 col8" >-0.086809</td>
      <td id="T_a7b5f_row27_col9" class="data row27 col9" >-0.079079</td>
      <td id="T_a7b5f_row27_col10" class="data row27 col10" >-0.058061</td>
      <td id="T_a7b5f_row27_col11" class="data row27 col11" >0.048325</td>
      <td id="T_a7b5f_row27_col12" class="data row27 col12" >0.064106</td>
      <td id="T_a7b5f_row27_col13" class="data row27 col13" >-0.002342</td>
      <td id="T_a7b5f_row27_col14" class="data row27 col14" >-0.001280</td>
      <td id="T_a7b5f_row27_col15" class="data row27 col15" >0.005872</td>
      <td id="T_a7b5f_row27_col16" class="data row27 col16" >0.119505</td>
      <td id="T_a7b5f_row27_col17" class="data row27 col17" >0.129846</td>
      <td id="T_a7b5f_row27_col18" class="data row27 col18" >0.026456</td>
      <td id="T_a7b5f_row27_col19" class="data row27 col19" >-0.028164</td>
      <td id="T_a7b5f_row27_col20" class="data row27 col20" >0.203407</td>
      <td id="T_a7b5f_row27_col21" class="data row27 col21" >-0.010803</td>
      <td id="T_a7b5f_row27_col22" class="data row27 col22" >0.074172</td>
      <td id="T_a7b5f_row27_col23" class="data row27 col23" >0.018489</td>
      <td id="T_a7b5f_row27_col24" class="data row27 col24" >0.044342</td>
      <td id="T_a7b5f_row27_col25" class="data row27 col25" >0.039463</td>
      <td id="T_a7b5f_row27_col26" class="data row27 col26" >0.025043</td>
      <td id="T_a7b5f_row27_col27" class="data row27 col27" >1.000000</td>
      <td id="T_a7b5f_row27_col28" class="data row27 col28" >0.799516</td>
      <td id="T_a7b5f_row27_col29" class="data row27 col29" >0.249781</td>
      <td id="T_a7b5f_row27_col30" class="data row27 col30" >0.149002</td>
      <td id="T_a7b5f_row27_col31" class="data row27 col31" >0.078605</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row28" class="row_heading level0 row28" >Follicle No. (R)</th>
      <td id="T_a7b5f_row28_col0" class="data row28 col0" >-0.158865</td>
      <td id="T_a7b5f_row28_col1" class="data row28 col1" >0.124092</td>
      <td id="T_a7b5f_row28_col2" class="data row28 col2" >0.074896</td>
      <td id="T_a7b5f_row28_col3" class="data row28 col3" >0.104205</td>
      <td id="T_a7b5f_row28_col4" class="data row28 col4" >0.049307</td>
      <td id="T_a7b5f_row28_col5" class="data row28 col5" >0.012752</td>
      <td id="T_a7b5f_row28_col6" class="data row28 col6" >0.073422</td>
      <td id="T_a7b5f_row28_col7" class="data row28 col7" >0.251271</td>
      <td id="T_a7b5f_row28_col8" class="data row28 col8" >-0.161256</td>
      <td id="T_a7b5f_row28_col9" class="data row28 col9" >-0.087227</td>
      <td id="T_a7b5f_row28_col10" class="data row28 col10" >-0.078688</td>
      <td id="T_a7b5f_row28_col11" class="data row28 col11" >0.018266</td>
      <td id="T_a7b5f_row28_col12" class="data row28 col12" >0.037524</td>
      <td id="T_a7b5f_row28_col13" class="data row28 col13" >-0.025358</td>
      <td id="T_a7b5f_row28_col14" class="data row28 col14" >0.003430</td>
      <td id="T_a7b5f_row28_col15" class="data row28 col15" >-0.007506</td>
      <td id="T_a7b5f_row28_col16" class="data row28 col16" >0.094799</td>
      <td id="T_a7b5f_row28_col17" class="data row28 col17" >0.093401</td>
      <td id="T_a7b5f_row28_col18" class="data row28 col18" >-0.002207</td>
      <td id="T_a7b5f_row28_col19" class="data row28 col19" >-0.016807</td>
      <td id="T_a7b5f_row28_col20" class="data row28 col20" >0.188537</td>
      <td id="T_a7b5f_row28_col21" class="data row28 col21" >-0.008607</td>
      <td id="T_a7b5f_row28_col22" class="data row28 col22" >0.091775</td>
      <td id="T_a7b5f_row28_col23" class="data row28 col23" >0.031483</td>
      <td id="T_a7b5f_row28_col24" class="data row28 col24" >0.027562</td>
      <td id="T_a7b5f_row28_col25" class="data row28 col25" >0.025515</td>
      <td id="T_a7b5f_row28_col26" class="data row28 col26" >0.037844</td>
      <td id="T_a7b5f_row28_col27" class="data row28 col27" >0.799516</td>
      <td id="T_a7b5f_row28_col28" class="data row28 col28" >1.000000</td>
      <td id="T_a7b5f_row28_col29" class="data row28 col29" >0.156640</td>
      <td id="T_a7b5f_row28_col30" class="data row28 col30" >0.188049</td>
      <td id="T_a7b5f_row28_col31" class="data row28 col31" >0.079817</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row29" class="row_heading level0 row29" >Avg. F size (L) (mm)</th>
      <td id="T_a7b5f_row29_col0" class="data row29 col0" >-0.017435</td>
      <td id="T_a7b5f_row29_col1" class="data row29 col1" >-0.021036</td>
      <td id="T_a7b5f_row29_col2" class="data row29 col2" >-0.025959</td>
      <td id="T_a7b5f_row29_col3" class="data row29 col3" >-0.011594</td>
      <td id="T_a7b5f_row29_col4" class="data row29 col4" >-0.048546</td>
      <td id="T_a7b5f_row29_col5" class="data row29 col5" >-0.031527</td>
      <td id="T_a7b5f_row29_col6" class="data row29 col6" >0.031995</td>
      <td id="T_a7b5f_row29_col7" class="data row29 col7" >0.034112</td>
      <td id="T_a7b5f_row29_col8" class="data row29 col8" >-0.052364</td>
      <td id="T_a7b5f_row29_col9" class="data row29 col9" >-0.072229</td>
      <td id="T_a7b5f_row29_col10" class="data row29 col10" >-0.056589</td>
      <td id="T_a7b5f_row29_col11" class="data row29 col11" >0.050100</td>
      <td id="T_a7b5f_row29_col12" class="data row29 col12" >0.000003</td>
      <td id="T_a7b5f_row29_col13" class="data row29 col13" >0.011547</td>
      <td id="T_a7b5f_row29_col14" class="data row29 col14" >0.035632</td>
      <td id="T_a7b5f_row29_col15" class="data row29 col15" >0.014923</td>
      <td id="T_a7b5f_row29_col16" class="data row29 col16" >-0.082276</td>
      <td id="T_a7b5f_row29_col17" class="data row29 col17" >0.031048</td>
      <td id="T_a7b5f_row29_col18" class="data row29 col18" >0.229355</td>
      <td id="T_a7b5f_row29_col19" class="data row29 col19" >-0.091201</td>
      <td id="T_a7b5f_row29_col20" class="data row29 col20" >0.134628</td>
      <td id="T_a7b5f_row29_col21" class="data row29 col21" >0.086311</td>
      <td id="T_a7b5f_row29_col22" class="data row29 col22" >0.005441</td>
      <td id="T_a7b5f_row29_col23" class="data row29 col23" >-0.040698</td>
      <td id="T_a7b5f_row29_col24" class="data row29 col24" >-0.046821</td>
      <td id="T_a7b5f_row29_col25" class="data row29 col25" >0.051273</td>
      <td id="T_a7b5f_row29_col26" class="data row29 col26" >0.037028</td>
      <td id="T_a7b5f_row29_col27" class="data row29 col27" >0.249781</td>
      <td id="T_a7b5f_row29_col28" class="data row29 col28" >0.156640</td>
      <td id="T_a7b5f_row29_col29" class="data row29 col29" >1.000000</td>
      <td id="T_a7b5f_row29_col30" class="data row29 col30" >0.522794</td>
      <td id="T_a7b5f_row29_col31" class="data row29 col31" >-0.014174</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row30" class="row_heading level0 row30" >Avg. F size (R) (mm)</th>
      <td id="T_a7b5f_row30_col0" class="data row30 col0" >-0.079646</td>
      <td id="T_a7b5f_row30_col1" class="data row30 col1" >-0.073115</td>
      <td id="T_a7b5f_row30_col2" class="data row30 col2" >0.059686</td>
      <td id="T_a7b5f_row30_col3" class="data row30 col3" >-0.111519</td>
      <td id="T_a7b5f_row30_col4" class="data row30 col4" >-0.034256</td>
      <td id="T_a7b5f_row30_col5" class="data row30 col5" >-0.022035</td>
      <td id="T_a7b5f_row30_col6" class="data row30 col6" >0.024153</td>
      <td id="T_a7b5f_row30_col7" class="data row30 col7" >0.016204</td>
      <td id="T_a7b5f_row30_col8" class="data row30 col8" >-0.013956</td>
      <td id="T_a7b5f_row30_col9" class="data row30 col9" >-0.097536</td>
      <td id="T_a7b5f_row30_col10" class="data row30 col10" >-0.117573</td>
      <td id="T_a7b5f_row30_col11" class="data row30 col11" >0.071863</td>
      <td id="T_a7b5f_row30_col12" class="data row30 col12" >0.037088</td>
      <td id="T_a7b5f_row30_col13" class="data row30 col13" >0.020184</td>
      <td id="T_a7b5f_row30_col14" class="data row30 col14" >0.032495</td>
      <td id="T_a7b5f_row30_col15" class="data row30 col15" >0.024070</td>
      <td id="T_a7b5f_row30_col16" class="data row30 col16" >-0.104384</td>
      <td id="T_a7b5f_row30_col17" class="data row30 col17" >-0.015271</td>
      <td id="T_a7b5f_row30_col18" class="data row30 col18" >0.176453</td>
      <td id="T_a7b5f_row30_col19" class="data row30 col19" >-0.088850</td>
      <td id="T_a7b5f_row30_col20" class="data row30 col20" >0.096079</td>
      <td id="T_a7b5f_row30_col21" class="data row30 col21" >0.071688</td>
      <td id="T_a7b5f_row30_col22" class="data row30 col22" >-0.011812</td>
      <td id="T_a7b5f_row30_col23" class="data row30 col23" >-0.040910</td>
      <td id="T_a7b5f_row30_col24" class="data row30 col24" >0.013137</td>
      <td id="T_a7b5f_row30_col25" class="data row30 col25" >0.038448</td>
      <td id="T_a7b5f_row30_col26" class="data row30 col26" >0.024013</td>
      <td id="T_a7b5f_row30_col27" class="data row30 col27" >0.149002</td>
      <td id="T_a7b5f_row30_col28" class="data row30 col28" >0.188049</td>
      <td id="T_a7b5f_row30_col29" class="data row30 col29" >0.522794</td>
      <td id="T_a7b5f_row30_col30" class="data row30 col30" >1.000000</td>
      <td id="T_a7b5f_row30_col31" class="data row30 col31" >-0.045628</td>
    </tr>
    <tr>
      <th id="T_a7b5f_level0_row31" class="row_heading level0 row31" >Endometrium (mm)</th>
      <td id="T_a7b5f_row31_col0" class="data row31 col0" >-0.101969</td>
      <td id="T_a7b5f_row31_col1" class="data row31 col1" >-0.010932</td>
      <td id="T_a7b5f_row31_col2" class="data row31 col2" >-0.055987</td>
      <td id="T_a7b5f_row31_col3" class="data row31 col3" >0.009320</td>
      <td id="T_a7b5f_row31_col4" class="data row31 col4" >-0.040891</td>
      <td id="T_a7b5f_row31_col5" class="data row31 col5" >-0.062941</td>
      <td id="T_a7b5f_row31_col6" class="data row31 col6" >-0.065041</td>
      <td id="T_a7b5f_row31_col7" class="data row31 col7" >0.042168</td>
      <td id="T_a7b5f_row31_col8" class="data row31 col8" >-0.016506</td>
      <td id="T_a7b5f_row31_col9" class="data row31 col9" >-0.105882</td>
      <td id="T_a7b5f_row31_col10" class="data row31 col10" >-0.067758</td>
      <td id="T_a7b5f_row31_col11" class="data row31 col11" >-0.051920</td>
      <td id="T_a7b5f_row31_col12" class="data row31 col12" >0.017013</td>
      <td id="T_a7b5f_row31_col13" class="data row31 col13" >-0.049158</td>
      <td id="T_a7b5f_row31_col14" class="data row31 col14" >0.010786</td>
      <td id="T_a7b5f_row31_col15" class="data row31 col15" >-0.053768</td>
      <td id="T_a7b5f_row31_col16" class="data row31 col16" >0.027778</td>
      <td id="T_a7b5f_row31_col17" class="data row31 col17" >0.012810</td>
      <td id="T_a7b5f_row31_col18" class="data row31 col18" >-0.026793</td>
      <td id="T_a7b5f_row31_col19" class="data row31 col19" >0.013863</td>
      <td id="T_a7b5f_row31_col20" class="data row31 col20" >0.104159</td>
      <td id="T_a7b5f_row31_col21" class="data row31 col21" >0.027439</td>
      <td id="T_a7b5f_row31_col22" class="data row31 col22" >-0.031385</td>
      <td id="T_a7b5f_row31_col23" class="data row31 col23" >-0.047987</td>
      <td id="T_a7b5f_row31_col24" class="data row31 col24" >-0.018700</td>
      <td id="T_a7b5f_row31_col25" class="data row31 col25" >-0.018613</td>
      <td id="T_a7b5f_row31_col26" class="data row31 col26" >-0.015342</td>
      <td id="T_a7b5f_row31_col27" class="data row31 col27" >0.078605</td>
      <td id="T_a7b5f_row31_col28" class="data row31 col28" >0.079817</td>
      <td id="T_a7b5f_row31_col29" class="data row31 col29" >-0.014174</td>
      <td id="T_a7b5f_row31_col30" class="data row31 col30" >-0.045628</td>
      <td id="T_a7b5f_row31_col31" class="data row31 col31" >1.000000</td>
    </tr>
  </tbody>
</table>




## Correlation analysis
So far we did nott split the data on PCOS diagnosis. Our next task is to do that. Let's make to different dataframes: 
* contains PCOS diagnosis
* does not contain PCOS diagnosis

This should not be strictly necessary, but I am just playing around pandas here.


```python
df_PCOS = df_PCOS_all[df_PCOS_all['PCOS (Y/N)']]
df_no_PCOS = df_PCOS_all[~df_PCOS_all['PCOS (Y/N)']]
```


```python
df_no_PCOS
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sl. No</th>
      <th>Patient File No.</th>
      <th>PCOS (Y/N)</th>
      <th>Age (yrs)</th>
      <th>Weight (Kg)</th>
      <th>Height(Cm)</th>
      <th>BMI</th>
      <th>Blood Group</th>
      <th>Pulse rate(bpm)</th>
      <th>RR (breaths/min)</th>
      <th>...</th>
      <th>Fast food (Y/N)</th>
      <th>Reg.Exercise(Y/N)</th>
      <th>BP _Systolic (mmHg)</th>
      <th>BP _Diastolic (mmHg)</th>
      <th>Follicle No. (L)</th>
      <th>Follicle No. (R)</th>
      <th>Avg. F size (L) (mm)</th>
      <th>Avg. F size (R) (mm)</th>
      <th>Endometrium (mm)</th>
      <th>Unnamed: 44</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>28</td>
      <td>44.6</td>
      <td>152.000</td>
      <td>19.300000</td>
      <td>15</td>
      <td>78</td>
      <td>22</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>110</td>
      <td>80</td>
      <td>3</td>
      <td>3</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>8.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>False</td>
      <td>36</td>
      <td>65.0</td>
      <td>161.500</td>
      <td>24.921163</td>
      <td>15</td>
      <td>74</td>
      <td>20</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>120</td>
      <td>70</td>
      <td>3</td>
      <td>5</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>3.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>False</td>
      <td>37</td>
      <td>65.0</td>
      <td>148.000</td>
      <td>29.674945</td>
      <td>13</td>
      <td>72</td>
      <td>20</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>120</td>
      <td>70</td>
      <td>2</td>
      <td>2</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>7.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>False</td>
      <td>25</td>
      <td>52.0</td>
      <td>161.000</td>
      <td>20.060954</td>
      <td>11</td>
      <td>72</td>
      <td>18</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>120</td>
      <td>80</td>
      <td>3</td>
      <td>4</td>
      <td>16.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>6</td>
      <td>False</td>
      <td>36</td>
      <td>74.1</td>
      <td>165.000</td>
      <td>27.217631</td>
      <td>15</td>
      <td>78</td>
      <td>28</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>110</td>
      <td>70</td>
      <td>9</td>
      <td>6</td>
      <td>16.0</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>535</th>
      <td>536</td>
      <td>536</td>
      <td>False</td>
      <td>26</td>
      <td>80.0</td>
      <td>161.544</td>
      <td>30.700000</td>
      <td>18</td>
      <td>70</td>
      <td>18</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>110</td>
      <td>80</td>
      <td>7</td>
      <td>9</td>
      <td>13.0</td>
      <td>17.5</td>
      <td>9.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>536</th>
      <td>537</td>
      <td>537</td>
      <td>False</td>
      <td>35</td>
      <td>50.0</td>
      <td>164.592</td>
      <td>18.500000</td>
      <td>17</td>
      <td>72</td>
      <td>16</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>110</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
      <td>17.5</td>
      <td>10.0</td>
      <td>6.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>537</th>
      <td>538</td>
      <td>538</td>
      <td>False</td>
      <td>30</td>
      <td>63.2</td>
      <td>158.000</td>
      <td>25.300000</td>
      <td>15</td>
      <td>72</td>
      <td>18</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>110</td>
      <td>70</td>
      <td>9</td>
      <td>7</td>
      <td>19.0</td>
      <td>18.0</td>
      <td>8.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>538</th>
      <td>539</td>
      <td>539</td>
      <td>False</td>
      <td>36</td>
      <td>54.0</td>
      <td>152.000</td>
      <td>23.400000</td>
      <td>13</td>
      <td>74</td>
      <td>20</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>110</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
      <td>18.0</td>
      <td>9.0</td>
      <td>7.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>539</th>
      <td>540</td>
      <td>540</td>
      <td>False</td>
      <td>27</td>
      <td>50.0</td>
      <td>150.000</td>
      <td>22.200000</td>
      <td>15</td>
      <td>74</td>
      <td>20</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>110</td>
      <td>70</td>
      <td>7</td>
      <td>6</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>11.5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>363 rows × 45 columns</p>
</div>




```python
# We have to use Perason's correlation coefficient because the dataframe contains both boolean and integer values.
corr = df_PCOS_all.drop(columns_to_drop, axis = 1).corr(method ='pearson')
# Let's try to extract the variables which have the highest and lowest correlation with PCOS status.
print(f' The three highest {list(corr.nlargest(3, "PCOS (Y/N)")["PCOS (Y/N)"].index)}')
print(f' The three smallest PCOS correlated variables are {list(corr.nsmallest(3, "PCOS (Y/N)")["PCOS (Y/N)"].index)}')
```

     The three highest ['PCOS (Y/N)', 'Follicle No. (R)', 'Follicle No. (L)']
     The three smallest PCOS correlated variables are ['Cycle length(days)', ' Age (yrs)', 'Marraige Status (Yrs)']


#### Checking the positive and negative correlations seperately
Do the correlation matrices of PCOS diagnosis vs no PCOS diagnosis look very different? 


```python
corrpos = df_PCOS.drop(columns_to_drop, axis = 1).corr()
corrneg = df_no_PCOS.drop(columns_to_drop, axis = 1).corr()
sns.heatmap(corrpos, cmap="Pastel1")
plt.title('Correlations in PCOS diagosed data');
```


    
![png](PCOS_files/PCOS_24_0.png)
    



```python
sns.heatmap(corrneg, cmap="Pastel1")
plt.title('Correlations in no PCOS diagosed data');
```


    
![png](PCOS_files/PCOS_25_0.png)
    



```python
sns.heatmap(corrpos-corrneg, cmap="Pastel1")
plt.title('Difference between PCOS diagnosed and no PCOS diagnose data');
```


    
![png](PCOS_files/PCOS_26_0.png)
    


For simplicity I would want to plot the 10 highest PCOS correlated and 5 lowest correlated variables as a heatmap. 


```python
#print(df_PCOS_all.describe())
#columns_to_drop = ['Sl. No', 'Patient File No.', 'Blood Group']
largest_vars = corr.nlargest(10, "PCOS (Y/N)")["PCOS (Y/N)"].index
smallest_vars = corr.nsmallest(5, "PCOS (Y/N)")["PCOS (Y/N)"].index
total_vars = largest_vars.append(smallest_vars)
sns.heatmap(df_PCOS_all.drop(columns_to_drop, axis=1)[total_vars].corr(), \
            cmap = "Pastel1")
plt.title('10 largest and 5 smallest PCOS correlated quantities');
```


    
![png](PCOS_files/PCOS_28_0.png)
    


### Visualise PCOS true and PCOS false data distributions


```python
df_PCOS_numeric = df_PCOS.select_dtypes(exclude = 'bool')
df_PCOS_numeric = df_PCOS_numeric.drop(columns_to_drop, axis = 1)
df_no_PCOS_numeric = df_no_PCOS.select_dtypes(exclude = 'bool')
df_no_PCOS_numeric = df_no_PCOS_numeric.drop(columns_to_drop, axis = 1)
variable_dropdown = Dropdown(options = df_PCOS_numeric.columns)
output_plot = Output()

def make_histograms(*args):
    output_plot.clear_output(wait = True)
    with output_plot:
        fig, ax1 = plt.subplots()
        sns.histplot(x = variable_dropdown.value, data = df_PCOS, kde = True, ax = ax1 )
        sns.histplot(x = variable_dropdown.value, data = df_no_PCOS, kde = True, ax = ax1)
        #ax1.get_legend()
        #legend = ax1.get_legend()
        #handles = legend.legend_handles
        #ax1.legend(handles, ['PCOS True', 'PCOS False'], title='Stat.ind.')
        plt.show()
        
variable_dropdown.observe(make_histograms, names = 'value')
display(variable_dropdown, output_plot)
```


    Dropdown(options=(' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI', 'Pulse rate(bpm) ', 'RR (breaths/min)', '…



    Output()


Among all the plots we can look at above the plots related to left and right follicle measurements are distinctly different. Let's look at them in a bit more detail.


```python
fig, ax = plt.subplots(2,2, figsize = (8,8))
sns.violinplot(df_PCOS_all, x = 'PCOS (Y/N)', y = 'Avg. F size (L) (mm)', ax = ax[0][0])
sns.violinplot(df_PCOS_all, x = 'PCOS (Y/N)', y = 'Avg. F size (R) (mm)', ax = ax[0][1])
sns.violinplot(df_PCOS_all, x = 'PCOS (Y/N)', y = 'Follicle No. (L)', ax = ax[1][0])
sns.violinplot(df_PCOS_all, x = 'PCOS (Y/N)', y = 'Follicle No. (R)', ax = ax[1][1])
```




    <Axes: xlabel='PCOS (Y/N)', ylabel='Follicle No. (R)'>




    
![png](PCOS_files/PCOS_32_1.png)
    


# Statistical analysis

My main goal is to be able to understand which factors are correlated with PCOS. These are not 'causes' of PCOS but rather symtoms which can help diagnose the condition. Note that correlations within binary data are also computed in .corr() method of pandas. Here I evaluate them explicitly. 

## Evaluating Cramér's V coefficient

Cramér's V coefficient is used for evaluating correlation between two variables. The variables may have more than binary categories. For variables A, B if 

$n_{i,j}$ = number of times $A_i, B_j$ were observed, 

$\chi^2 = \sum_{i,j}\frac{(n_{i,j} - {\frac{n_i, n_j}{n}})}{\frac{n_i, n_j}{n}}$

where $n_i = \sum_j n_{i,j}$ is the number of times the value $A_i$ is observed and $n_i = \sum_j n_{i,j}$ is the number of times the value $B_j$ is observed. 

The coefficient is the computed as 

$V = \sqrt\frac{\chi^2/n}{min(k-1)(r-1)}$

where $k$ = number of columns, $r$ = number of rows, $n$ = total number of observations


```python
from scipy.stats import chi2_contingency
from tabulate import tabulate
import numpy as np
df_PCOS_bool = df_PCOS_all.select_dtypes(include = 'bool')
results = []
for col in df_PCOS_bool.columns:
    if col == 'PCOS (Y/N)': continue
    pearson_corr = df_PCOS_bool['PCOS (Y/N)'].astype(int).corr(df_PCOS_bool[col].astype(int))
    # Now let's compute the Cramér's V coefficient
    contigency_table = pd.crosstab(df_PCOS_bool['PCOS (Y/N)'].astype(int), df_PCOS_bool[col].astype(int))
    print('----------------------------------')
    print(contigency_table )
    
    chi2, _, _, _ = chi2_contingency(contigency_table)
    n = contigency_table.values.sum()
    cramers_v = np.sqrt(chi2/n)
    results.append([col, pearson_corr, cramers_v])
print('----------------------------------')
print(tabulate(results, headers = ['criteria', 'Pearsons coefficient', 'cramers_v']))
```

    ----------------------------------
    Pregnant(Y/N)    0    1
    PCOS (Y/N)             
    0              221  142
    1              113   64
    ----------------------------------
    Weight gain(Y/N)    0    1
    PCOS (Y/N)                
    0                 280   83
    1                  56  121
    ----------------------------------
    hair growth(Y/N)    0    1
    PCOS (Y/N)                
    0                 316   47
    1                  76  101
    ----------------------------------
    Skin darkening (Y/N)    0    1
    PCOS (Y/N)                    
    0                     307   56
    1                      67  110
    ----------------------------------
    Hair loss(Y/N)    0    1
    PCOS (Y/N)              
    0               220  143
    1                75  102
    ----------------------------------
    Pimples(Y/N)    0    1
    PCOS (Y/N)            
    0             222  141
    1              54  123
    ----------------------------------
    Fast food (Y/N)    0    1
    PCOS (Y/N)               
    0                223  140
    1                 38  139
    ----------------------------------
    Reg.Exercise(Y/N)    0   1
    PCOS (Y/N)                
    0                  281  82
    1                  126  51
    ----------------------------------
    criteria                Pearsons coefficient    cramers_v
    --------------------  ----------------------  -----------
    Pregnant(Y/N)                     -0.0286064    0.0245456
    Weight gain(Y/N)                   0.440488     0.436419
    hair growth(Y/N)                   0.464245     0.459823
    Skin darkening (Y/N)               0.475283     0.471008
    Hair loss(Y/N)                     0.171913     0.167951
    Pimples(Y/N)                       0.287802     0.283856
    Fast food (Y/N)                    0.375389     0.371442
    Reg.Exercise(Y/N)                  0.0678092    0.0632309


It seems that hair growth, skin darknin and hair loss are correlated with PCOS diagnosis. Fast food seems to have a correlation but I wonder if that is an artefact.

# Training a classifier (Random Forest)

I will use a random forrest classifier here because there is no specific knowledge to choose other classifiers. In addition, the data consists of both integers and boolaen types. I believe that random forest classifier should work.

$precision = \frac{TP}{TP+FP}$ 

$recall = \frac{TP}{FP+FN}$

$F_1 score = \frac{precision \times recall}{precision + recall}$


```python
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

columns_to_convert = df_PCOS_all.filter(like = 'Y/N').columns
df_PCOS_all[columns_to_convert] = df_PCOS_all[columns_to_convert].astype('int')

label_col = "PCOS (Y/N)"
X = df_PCOS_all.drop(columns = [label_col])
X = X.drop(columns = columns_to_drop, axis = 1)

y = df_PCOS_all[label_col]

float_columns = X.select_dtypes(include = ['float64']).columns

scaler=StandardScaler()
X[float_columns] = scaler.fit_transform(X[float_columns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy is {accuracy}')

print('\n Classification report is')
print(classification_report(y_test, y_pred))

# Optional: Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in conf_matrix]))

```

    Accuracy is 0.8888888888888888
    
     Classification report is
                  precision    recall  f1-score   support
    
               0       0.87      0.97      0.92        71
               1       0.93      0.73      0.82        37
    
        accuracy                           0.89       108
       macro avg       0.90      0.85      0.87       108
    weighted avg       0.89      0.89      0.89       108
    
    
     Confusion Matrix:
    69	2
    10	27


The above scores don't look too bad. 

Let us try to make a precisio vs recall curve. The following curve doesn't look right, so I have to do something to fix it.


```python
y_scores = cross_val_predict(clf, X_train, y_train)#, method = 'decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
print(precisions,recalls, thresholds)
plt.plot(thresholds, precisions[:-1])
plt.plot(thresholds, recalls[:-1])
plt.scatter(thresholds, precisions[:-1])
plt.scatter(thresholds, recalls[:-1])
```

    [0.3287037  0.89430894 1.        ] [1.         0.77464789 0.        ] [0 1]





    <matplotlib.collections.PathCollection at 0x15f8de990>




    
![png](PCOS_files/PCOS_41_2.png)
    


# Hypothesis testing (next step)
see https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e for hypothesis testing


```python

```


```python

```
