# Python-Practise
Python for Data Science

## *LECTURE 1*

## Introduction to Python

### Instrall Python Interpreter by itself
* Python
    * Object-oriented language -> entity = behavior + data
    * Dynamic -> type-checking at runtime
* Language Basics (quick demo)
    * Commenting code
    * Variables
        * case sensitive
        * = is the assignment operator
    * Data Objects
        * Lists
        * Arrays
        * Dictionaries
    * Conditional Statements
    * Loops
    * Functions
        * packages are adds on to the basic installation of python
        * import opens a library of functions within a package

##  Language Essentials

## <font color = 'green'> Basic Variable Types </font>

* Python is case sensitive and there is no need to do any complicated initialization of variables
* Python variables are dynamically typed. This means that you can define variables when they are used
* Python has several variable types, a few of which are **string, integer, float, boolean**

## <font color = 'green'> Collection data types </font>
1. **lists**: is a collection which is ordered and changeable. Allows duplicate members.
    * Lists are container variables.
    * You can store any basic variable data type in a list (different or the same)
    * Lists are mutable
    * Each item can be accessed using a list index
    * Index starts from 0 instead of 1
    * Length of a list can be obtained using 
    ```python
    len( ) function
    ```
    
2. **tuple**: is a collection which is ordered and unchangeable. Allows duplicate members.
    * Similar to lists except use ( ) instead of [ ] and tuples are immutable
3. **dictionary**: is a collection which is unordered, changeable and indexed. No duplicate members.
    * Dictionaries are composed of key value pairs. 
* The data is indexed by key instead of indexes, hence values can be accessed using the associated keys.

## <font color = 'green'> Operators </font>

Operators are symbols that perform certain operations on variables or values within variables. Following are some of Python's operators:

**Arithmetic operators** do basic mathematical operations:
* ***** (multiplication)
* **+** (addition)
* **-** (subtraction) 
* **/** (division)
* **%** (modules)
* ****** (exponentation)

**Relational operators** carry out comparison operations.
* **<** (less than)
* **>** (greater than)
* **<=** (less than or equal)
* **>=** (greater than or equal)
* **==** (equal to) 
* **!=** (not equal) (python2 allowed <> but python3 no longer does)

**Assignment** Operator
* **=** (assign)
* **+=** add and assign 
* (so on)

**Logical operators**
* **AND**
* **OR**
* **NOT**

List operators
    * +

## <font color = 'green'> String Formatting </font>

Python supports embedding variable values within strings. We can specify a placeholder for variables using one of two styles
* C style formatting 
```python
name="Sharat"
print("Hello %s!" % name)
```
* Python style formatting
``` python
name="Ani"
print("Hello {}!".format(name))
```
* Python style formatting with named substitutions
```python
name="Sharat"
print("Hello {user_name}!".format(user_name=name))
```

## <font color = 'green'> Conditional Evaluation </font>
if, elif, else construct can be used to do conditional evaluation of code. 

```python
if statement:
    # execute if statement is true
elif other_statement:
    # execute other_statement
    # this branch is optional
else:
    # execute if both statements are false
    # this branch is optional
```
    
* Note the **indentation**. Python uses indendation to separate code blocks
* Note the trailing **':'**. Python uses : to indicate end of condition evaluation
* Omitting either of these will lead to syntax error
* **If** and **else** are the core of these statements. You can have many **elif**-s or have none.

## <font color = 'green'> Loops </font>
Python supports two looping constructs
* **for** loops are used to iterating over a sequence (lists, strings, dictionaries, etc.)
```python
for var in list:
    # you can exit using break (terminates iterations)
    # you can loop again using continue (skips current iteration of loop)
    # you can do something with var
```

* **while**
```python
while statement:
    # you can do something while statement is true
    # you can exit using break
    # you can loop over using continue
    # you can do something to possible change statement value
```

## <font color = 'green'> Functions & Methods </font>

You can either create your own functions or you can use already existing functions (or both). 

Functions can be used to split the program logic into smaller reusable components. Functions are paramterized by their `name` and `parameters`

`Arguments` are the things you supply to the function when you call it but `parameters` are the names you use when you define a function

```python
def foo(bar):
    print("bar value:{}".format(bar))

foo("hello")
foo("world")
```

## <font color = 'green'> List Comprehensions </font>

List comprehensions provide a concise way to `create lists`. The `expression` comes first right after the `square brackets`, then the `for` or `if` clauses, then the `condition or none`.

The result will be a new list resulting from evaluating the expression in the context of the `for` or `if` clauses which follow it.


## *LECTURE 2*
## Pandas & NumPy

## 1. Pandas - Panel Data

`Pandas` is a Python library for analyzing data. It provides classes, methods, and functions to `read`, `manipulate`, and `analyze` tabular data.
* Tabular or relational data is organized into rows and columns
* Rows contain individual elements
* Columns contain properties of each element

To use pandas in your code, use:
```python
import pandas as pd
```

### Creating Series and DataFrames
1. Lists
    * lists can turn into dataframes
```python
lst = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks'] 
df = pd.DataFrame(lst, index =['a', 'b', 'c', 'd', 'e', 'f', 'g'], columns =['Names'])
```

2. Dictionaries
    * dictionaries can turn into dataframes

```python
data = {'team':['Leicester', 'Manchester City', 'Arsenal'], 
        'player':['Vardy', 'Aguero', 'Sanchez'], 
        'goals':[24,22,19]
       }

my_df = pd.DataFrame(data)
my_df

#or
football = DataFrame(data, columns=['player','team','goals','played'], 
                     index=['one','two','three'])

#or from a list dictionaries
df = pd.DataFrame([{'Item':'Book', 'Cost':10},{'Item':'Pen','Cost':2}])

```

3. Series
    * Each series represents a column
```python
items = pd.Series(['Book','Pen'])
costs = pd.Series([10,2])
df = pd.DataFrame({'Item':items,'Cost':costs})
```

4. From a file
    * Pandas can import data from sql, csv, tsv, ecel, json, etc.

```python
pd.read_csv('data.csv')
```

### Accessing Data
1. Lists
    * each element of the list represents a row
    * elements of a python list can be accessed by a numerical index 
```python
list[i]
```

2. Dictionaries
    * elements of a python dictionary can be accessed using keys
```python
dict['key']
```

3. Series
    * elements of a python series can be accessed both like lists and like dictionaries
```python
series['key']
series[i]
```

4. DataFrames
    * 
```python
bn_csv['column']
series[i]
```   
###  Rreding in csv files 
```python 
pd.read_csv()
```

### Reading in excel files 

* using `read_excel()` function from pandas

```python 
pandas.read_excel()
```
* using other python libraries (e.g., openpyxl)
    * you would need to first 
    ```ptyhon
pip install openpyxl
conda install openpyxl
    ```
    * then bring in the library
    ```python 
    from openpyxl import load_workbook
    workbook = load_workbook(filename="sample.xlsx")
    workbook.sheetnames
    ```

### Reading in JSON files
### Reading in data by web scraping
### Reading in data from SQL DBs
### Reading in images
```python
import cv2
my_img = cv2.imread()
```

### Dataframe properties


df.columns #names of columns
df.size # 
df.shape # number of rows and columns
df.info() # column descriptions
df.describe() # descriptive statistics

### Modifying Data
* Modifying individual columns is the same as modifying series
```python
cost = df['Cost]
cost = cost+10
```
**Note** that modifying the series modifies the dataframe

* Rows can be modified using slicing/indexing sections of the data frame

### Querying data through Boolean
### Descriptive Statistics
### Split-apply-combine with `groupby`

## 2. NumPy - Numeric Python 

NumPy is the fundamental package for `scientific computing` with Python. `ndarray`(n-dimensional array) is the primary data structure in `NumPy`. It is a `table of elements` (usually numbers), all of the same type, `indexed` by a tuple of non-negative integers. NumPy Array is an alternative to `python lists` with wider abilities: perform calculations over entire arrays. In NumPy `dimensions` are called `axes`.

To use numpy in your code, use:
```python
#run this in command line (terminal)
pip install numpy
#or
conda install numpy

#then add this to your notebook
import numpy as np
```

### NumPy Array Type

NumPy assumes that your array elements are of the single type: array of floats, boolean, strings, etc. So numpy array is a new kind of a python type, hence, it has its own methods. If you create a list with different types of data, some of the elements' types will be changed to end up with a homogeneous list (see below). This is known as `type coercion`.

```python
my_list_a = [100.0, True]
my_liist_b = [1, 3, "hello"]
my_list_a + my_liist_b

#compare the output of the above code with the one below

my_array_a = np.array([100, 13, 7])
my_array_b = np.array(['bob', True, 103])
```

### NumPy <font color='green'> Array</font>
NumPy’s array class is called `ndarray`. It is also known by the alias `array`. Note that `numpy.array` is not the same as the Standard Python Library class `array.array`, which only handles one-dimensional `arrays` and offers less functionality.

ndarray:
   * supports multiple numeric types (e.g., float, int, complex)
   * numpy arrays have attributes (ndim, shape, size, dtype) unlike standard python arrays
   * supports operators (vectorization)
   * allows indexing, slicing, reshaping


### <font color='green'>`Arrays` & Dimenstionality</font>
### <font color='green'>Creating a `numpy array`:</font>
### <font color='green'>Operations with `numpy arrays`</font>
### <font color='green'>Descriptive Statistics</font>
### <font color='green'>Indexing, Slicing, Iterating</font>



## *LECTURE 3*

## Visualizations

1. Matplotlib
2. Seaborn
3. Bokeh
4. Plotly


## Predictive Analytics
1. Linear Model (OLS)
2. Logistic Regression
3. Cluster Analysis
4. Decision Tree
5. Neural Nets

### Matplotlib
Matplotlib is a Python 2D plotting library which produces various visualizations. This library has and is still being used extensively for scitntific publications. 
In this class we will be using Matplotlib within Jupyter but you can use it in python scripts, web app servers, etc. 

You can find more graphs here: https://matplotlib.org/3.1.1/gallery/index.html#mplot3d-examples-index

### Plotting a Line
### Bar Graph
### Histogram
### Pie Chart

### Seaborn

Seaborn is a Python data visualization library based on matplotlib. It's static just like matplotlib but it provides a more indepth, attractive, and informative statistical graphics than common matplotlib graphics.

See some of the example: https://seaborn.pydata.org/examples/index.html

### Histogram & Density plots

A density plot is a representation of the probability of density distribution, a distribution of a numeric variable where x is still the numeric expression of the metric and y is the probability. It uses a kernel density estimate to show the probability density function of the variable: the porbability of the metric being equal to a specific number in the distribution. It is a smoothed version of the histogram and does not have the bars or the related change in the distribution as we change the number of bins.

### Heatmap

Heatmaps are 2D graphical representations of data values in a data matrix represented by color.

### Scatterplot

Pairplots create two basic figures: histograms and scatter plots. The histogram on the diagonal allows us to see the distribution of a single variable while the scatter plots on the upper and lower triangles show the relationship (or lack thereof) between two or more variables.

### Bokeh
### Philosophy: Grammar of Graphics
### Line Graph
### Bar Charts

### Plotly

Unlike matplotlip and others, this is a visualization library that offers interactivity.

You need to install it first:

```python
#pip
pip install plotly
#or
pip install plotly --upgrade

#conda
conda install plotly

```

Then you need to import plotly graph objects
```python 
import plotly.graph_objects as go
```

Find more examples here: https://plot.ly/python/creating-and-updating-figures/


### Apache Superset
## Predictive Analytics
### Linear Regression (OLS model)
### Clustering (k-means)
Grouping observations based on similarity. We are not predicting anything.

Features – continuous

### Classification Algorithms

A) Logistic Regression

B) Decision Tree

C) Neural Network

All these methods share common steps:
* categorical data needs to be encoded as dummy data
* data needs to be split into training and test

### TensorFlow Playground
