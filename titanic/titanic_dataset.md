```python
import numpy as np
import pandas as pd
import re
```


```python
path = "../dataset/titanic/"
```


```python
dt_train = pd.read_csv(path+"train.csv")
```


```python
#taking a look in the data
dt_train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#SibSp is a column that have the numbers of siblins and spouses on boat of someone
#We can see that some people have 0 - 8 others SibSp with them
dt_train['SibSp'].unique()
```




    array([1, 0, 3, 4, 2, 5, 8], dtype=int64)




```python
#A way to know who is the SibSp of someone is looking for their surname
matchs = []
for n in dt_train['Name']:
    if "Thayer" in n:
        matchs.append(n)
matchs
```




    ['Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
     'Thayer, Mr. John Borland Jr',
     'Thayer, Mrs. John Borland (Marian Longstreth Morris)',
     'Thayer, Mr. John Borland']




```python
# If a woman is married the name of her husband appear before and her name appear in brackets
# Taking a look in the Miss Marian
dt_train[dt_train['Name'] == "Thayer, Mrs. John Borland (Marian Longstreth Morris)"]
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>581</th>
      <td>582</td>
      <td>1</td>
      <td>1</td>
      <td>Thayer, Mrs. John Borland (Marian Longstreth M...</td>
      <td>female</td>
      <td>39.0</td>
      <td>1</td>
      <td>1</td>
      <td>17421</td>
      <td>110.8833</td>
      <td>C68</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Finding her relatives we can see that Miss Marian has a husband and a son with her
dt_train.iloc[[550,581,698]]
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>550</th>
      <td>551</td>
      <td>1</td>
      <td>1</td>
      <td>Thayer, Mr. John Borland Jr</td>
      <td>male</td>
      <td>17.0</td>
      <td>0</td>
      <td>2</td>
      <td>17421</td>
      <td>110.8833</td>
      <td>C70</td>
      <td>C</td>
    </tr>
    <tr>
      <th>581</th>
      <td>582</td>
      <td>1</td>
      <td>1</td>
      <td>Thayer, Mrs. John Borland (Marian Longstreth M...</td>
      <td>female</td>
      <td>39.0</td>
      <td>1</td>
      <td>1</td>
      <td>17421</td>
      <td>110.8833</td>
      <td>C68</td>
      <td>C</td>
    </tr>
    <tr>
      <th>698</th>
      <td>699</td>
      <td>0</td>
      <td>1</td>
      <td>Thayer, Mr. John Borland</td>
      <td>male</td>
      <td>49.0</td>
      <td>1</td>
      <td>1</td>
      <td>17421</td>
      <td>110.8833</td>
      <td>C68</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now, we will create a new column saying if a person is married or not
## NOTE: In this column doesn't matter if a person is married and spouse isn't on the boat, 
## that people will be considered married
married = np.zeros((len(dt_train['Name']), 1)).astype(int)
for n in dt_train['Name']:
    count = 0
    pos = dt_train[dt_train['Name']==n].index.values.astype(int)[0]
    if "(" in n:
        married[pos] = 1
    else:
        for t in dt_train['Name']:
            if n in t:
                count += 1
        if count > 1:
            married[pos] = 1
married[:5]
```




    array([[0],
           [1],
           [0],
           [1],
           [0]])




```python
# Saving on the data the new column
dt_train['Married'] = married
```


```python
# Now, we'll create a new column consider if the spouse of a person is on the boat.
spouses_on_boat = np.zeros((len(dt_train['Name']),1)).astype(int)
for i in range(len(dt_train['SibSp'])):
    s = dt_train.iloc[i]
    if int(s['Married']) and s['SibSp'] > 0:
        spouses_on_boat[i] = 1
dt_train['Spouse_on_boat'] = spouses_on_boat # Saving the new column
```


```python
# Creating a column that will say if a person has sisters on the boat
sisters = np.zeros((len(dt_train['Name']), 1)).astype(int)
for i in range(len(dt_train['SibSp'])):
    s = dt_train.iloc[i]['SibSp'] - dt_train.iloc[i]['Spouse_on_boat']
    sisters[i] = s
dt_train['Sisters'] = sisters # Saving the new column
```


```python
# Taking a look in the data we will see data has some NaN in "Age" column

dt_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   PassengerId     891 non-null    int64  
     1   Survived        891 non-null    int64  
     2   Pclass          891 non-null    int64  
     3   Name            891 non-null    object 
     4   Sex             891 non-null    object 
     5   Age             714 non-null    float64
     6   SibSp           891 non-null    int64  
     7   Parch           891 non-null    int64  
     8   Ticket          891 non-null    object 
     9   Fare            891 non-null    float64
     10  Cabin           204 non-null    object 
     11  Embarked        889 non-null    object 
     12  Married         891 non-null    int32  
     13  Spouse_on_boat  891 non-null    int32  
     14  Sisters         891 non-null    int32  
    dtypes: float64(2), int32(3), int64(5), object(5)
    memory usage: 94.1+ KB
    


```python
# How "Cabin" has only 204 registers we will remove from data
dt_train = dt_train.drop(columns=['Cabin']) #another way: dt_train.drop('Cabin', axis=1, inplace=True)
```


```python
# How we have 714 of 891 registers in "age" column, we will fill in the column with mean of all ages
dt_train['Age'].fillna(round(dt_train['Age'].mean(), 2), inplace=True)
```


```python
# Removing the off data in 'Embarked' column.
dt_train = dt_train.dropna(subset=['Embarked'])
```


```python
# Columun 'Embarked' has only 3 types. It's a categorical colum
dt_train['Embarked'].unique()
```




    array(['S', 'C', 'Q'], dtype=object)




```python
embarked_cat = dt_train[['Embarked']]
embarked_cat
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
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>889 rows × 1 columns</p>
</div>




```python
# Changing male to 0 and female to 1
dt_train = dt_train.replace(['male', 'female'], [0, 1])
dt_train['Sex'].unique()
```




    array([0, 1], dtype=int64)




```python
# Now, we have the correlation matrix and four of three columns we have modify are very correlation with "survived" column
corr_matrix = dt_train.corr()
corr_matrix['Survived'].sort_values(ascending=True)
```




    Pclass           -0.335549
    Sisters          -0.103881
    Age              -0.074675
    SibSp            -0.034040
    PassengerId      -0.005028
    Parch             0.083151
    Fare              0.255290
    Spouse_on_boat    0.260131
    Married           0.338783
    Sex               0.541585
    Survived          1.000000
    Name: Survived, dtype: float64




```python
# Now, fare comlumn has a large difference between the mean and the max number
dt_train.describe()['Fare']
```




    count    889.000000
    mean      32.096681
    std       49.697504
    min        0.000000
    25%        7.895800
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64




```python
# Only 3 persons has fare > 263 and it can be a problem in ml algoritms
dt_train[dt_train['Fare'] > 263].count()
```




    PassengerId       3
    Survived          3
    Pclass            3
    Name              3
    Sex               3
    Age               3
    SibSp             3
    Parch             3
    Ticket            3
    Fare              3
    Embarked          3
    Married           3
    Spouse_on_boat    3
    Sisters           3
    dtype: int64




```python
#taking a look in a graphic
import matplotlib.pyplot as plt
```


```python
fig = plt.figure()
fig_fare = fig.add_subplot(1,1,1)
fig_fare.plot(dt_train['Fare'])
fig_fare.plot(np.full((889, 1), 263), label='Discrepant values', linestyle="--")
fig_fare.legend(loc='best')
fig_fare.set_title('Fare')
fig_fare.set_yticks([0, 100, 200, 263, 300, 400, 500])
```




    [<matplotlib.axis.YTick at 0x1d5312c4490>,
     <matplotlib.axis.YTick at 0x1d53200cfa0>,
     <matplotlib.axis.YTick at 0x1d532002e50>,
     <matplotlib.axis.YTick at 0x1d5322b3550>,
     <matplotlib.axis.YTick at 0x1d5322b3a60>,
     <matplotlib.axis.YTick at 0x1d5322b3f70>,
     <matplotlib.axis.YTick at 0x1d5322ba4c0>]




    
![png](output_23_1.png)
    



```python
# removing discrepant values from 'Fare'
dt_train = dt_train[dt_train['Fare'] <= 263]
```


```python
fig = plt.figure()
fig_fare = fig.add_subplot(1,1,1)
fig_fare.plot(dt_train['Fare'])
fig_fare.plot(np.full((889, 1), 263), label='Discrepant values', linestyle="--")
fig_fare.legend(loc='best')
fig_fare.set_title('Fare')
fig_fare.set_yticks([0, 100, 200, 263, 300, 400, 500])
```




    [<matplotlib.axis.YTick at 0x1d53236b4f0>,
     <matplotlib.axis.YTick at 0x1d53236b0d0>,
     <matplotlib.axis.YTick at 0x1d532363ee0>,
     <matplotlib.axis.YTick at 0x1d5323955b0>,
     <matplotlib.axis.YTick at 0x1d532395ac0>,
     <matplotlib.axis.YTick at 0x1d532395fd0>,
     <matplotlib.axis.YTick at 0x1d5323957f0>]




    
![png](output_25_1.png)
    



```python
dt_train.head(10)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Married</th>
      <th>Spouse_on_boat</th>
      <th>Sisters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>29.7</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>0</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>0</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>1</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>1</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>C</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Miss Elisabeth Vilhemina Berg has two relatives on the boat
# but we need to know if her relatives are her parents or her childrens
# Take a look from Ticket number:

dt_train[dt_train['Ticket']=="347742"]
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Married</th>
      <th>Spouse_on_boat</th>
      <th>Sisters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>1</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>172</th>
      <td>173</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Miss. Eleanor Ileen</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>869</th>
      <td>870</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Master. Harold Theodor</td>
      <td>0</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
###NOTE:it is still in implementation

## So, we can see Miss Elisabeth has two children of 4 and 1 years old.

# Now, we'll create two new columns named "Son" and "Parent".
# How the name says, if the passenger is a "son" of someone it is 1, if not, 0.
# The same will occurs with Parent row.

#relative_or_not = []
#for t in dt_train['Ticket']:
#    relative_or_not = 
```


```python
from sklearn.preprocessing import LabelEncoder
#Creating a sparse matrix for categorical column 'Embarked'
le = LabelEncoder()
dt_train['Embarked'] = le.fit_transform(dt_train['Embarked'])
le.classes_
```




    array(['C', 'Q', 'S'], dtype=object)




```python
# removing text columns from data
dt_train.drop('Name', axis=1, inplace=True)
dt_train.drop('Ticket', axis=1, inplace=True)
dt_train.head(6)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Married</th>
      <th>Spouse_on_boat</th>
      <th>Sisters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>29.7</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import StandardScaler
# Scaling the data to use in ml algoritm
scaler = StandardScaler()
target = dt_train["Survived"]
dt_train.drop("PassengerId", axis=1, inplace=True)
#dt_train.drop("Survived", axis=1, inplace=True) # take off target column from data
dt_train_scaled = scaler.fit_transform(dt_train)
dt_train_scaled = pd.DataFrame(dt_train_scaled)
dt_train_scaled
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.783482</td>
      <td>0.821948</td>
      <td>-0.735438</td>
      <td>-0.588223</td>
      <td>0.429180</td>
      <td>-0.474007</td>
      <td>-0.564532</td>
      <td>0.583017</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>0.519261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.276354</td>
      <td>-1.581503</td>
      <td>1.359733</td>
      <td>0.644562</td>
      <td>0.429180</td>
      <td>-0.474007</td>
      <td>0.992225</td>
      <td>-1.955329</td>
      <td>2.269973</td>
      <td>3.089922</td>
      <td>-0.393567</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.276354</td>
      <td>0.821948</td>
      <td>1.359733</td>
      <td>-0.280026</td>
      <td>-0.476185</td>
      <td>-0.474007</td>
      <td>-0.548122</td>
      <td>0.583017</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>-0.393567</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.276354</td>
      <td>-1.581503</td>
      <td>1.359733</td>
      <td>0.413415</td>
      <td>0.429180</td>
      <td>-0.474007</td>
      <td>0.550159</td>
      <td>0.583017</td>
      <td>2.269973</td>
      <td>3.089922</td>
      <td>-0.393567</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.783482</td>
      <td>0.821948</td>
      <td>-0.735438</td>
      <td>0.413415</td>
      <td>-0.476185</td>
      <td>-0.474007</td>
      <td>-0.545083</td>
      <td>0.583017</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>-0.393567</td>
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
    </tr>
    <tr>
      <th>881</th>
      <td>-0.783482</td>
      <td>-0.379778</td>
      <td>-0.735438</td>
      <td>-0.202977</td>
      <td>-0.476185</td>
      <td>-0.474007</td>
      <td>-0.424740</td>
      <td>0.583017</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>-0.393567</td>
    </tr>
    <tr>
      <th>882</th>
      <td>1.276354</td>
      <td>-1.581503</td>
      <td>1.359733</td>
      <td>-0.819370</td>
      <td>-0.476185</td>
      <td>-0.474007</td>
      <td>-0.011441</td>
      <td>0.583017</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>-0.393567</td>
    </tr>
    <tr>
      <th>883</th>
      <td>-0.783482</td>
      <td>0.821948</td>
      <td>1.359733</td>
      <td>0.005055</td>
      <td>0.429180</td>
      <td>2.003694</td>
      <td>-0.170683</td>
      <td>0.583017</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>0.519261</td>
    </tr>
    <tr>
      <th>884</th>
      <td>1.276354</td>
      <td>-1.581503</td>
      <td>-0.735438</td>
      <td>-0.280026</td>
      <td>-0.476185</td>
      <td>-0.474007</td>
      <td>-0.011441</td>
      <td>-1.955329</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>-0.393567</td>
    </tr>
    <tr>
      <th>885</th>
      <td>-0.783482</td>
      <td>0.821948</td>
      <td>-0.735438</td>
      <td>0.182268</td>
      <td>-0.476185</td>
      <td>-0.474007</td>
      <td>-0.552376</td>
      <td>-0.686156</td>
      <td>-0.440534</td>
      <td>-0.323633</td>
      <td>-0.393567</td>
    </tr>
  </tbody>
</table>
<p>886 rows × 11 columns</p>
</div>




```python
corr_matrix = dt_train_corr_matrix.corr()
corr_matrix[0].sort_values(ascending=True)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-36-8d0337895b26> in <module>
    ----> 1 corr_matrix = dt_train_corr_matrix.corr()
          2 corr_matrix[0].sort_values(ascending=True)
    

    NameError: name 'dt_train_corr_matrix' is not defined



```python
dt_train_scaled.drop(0, axis=1, inplace=True)

from sklearn.linear_model import LogisticRegression
#aplying the Logistic Regression algoritm
log_r = LogisticRegression()
log_r.fit(dt_train_scaled, target)
```


```python
proba = log_r.predict_proba(dt_train_scaled)
```


```python
proba_fig = plt.figure()
p = proba_fig.add_subplot()
p.hist(proba)
```


```python
log_r.predict(dt_train_scaled)[:5]
```


```python
#Now, we need to use the algoritm (already learned) on the data test.
#Perhaps we need to create a Pipeline and it will be more efficiently
### ...
### ...
### ...
```
