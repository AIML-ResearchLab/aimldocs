# What is NumPy?
NumPy, short for Numerical Python, is an open-source Python library. It supports multi-dimensional arrays (matrices) and provides a wide range of mathematical functions for array operations. It is used in scientific computing, and in areas like data analysis, machine learning, etc.

# Why to Use NumPy?
In Python we have lists that serve the purpose of arrays, but they are slow to process.
NumPy aims to provide an array object that is up to 50x faster than traditional Python lists.
The array object in NumPy is called ndarray, it provides a lot of supporting functions that make working with ndarray very easy.
Arrays are very frequently used in data science, where speed and resources are very important.

- NumPy provides various math functions for calculations like addition, algebra, and data analysis.
- NumPy provides various objects representing arrays and multi-dimensional arrays which can be used to handle large data such as images, sounds, etc.
- NumPy also works with other libraries like **SciPy** (for scientific computing), **Pandas** (for data analysis), and **scikit-learn** (for machine learning).
- NumPy is fast and reliable, which makes it a great choice for numerical computing in Python.

# Why is NumPy Faster Than Lists?
NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
This behavior is called locality of reference in computer science.
This is the main reason why NumPy is faster than lists. Also it is optimized to work with latest CPU architectures.

# Which Language is NumPy written in?
NumPy is a Python library and is written partially in Python, but most of the parts that require fast computation are written in C or C++.


# NumPy Applications

The following are some common application areas where NumPy is extensively used:

- **Data Analysis:** In Data analysis, while handling data, we can create data (in the form of array objects), filter the data, and perform various operations such as mean, finding the standard deviations, etc.
- **Machine Learning & AI:** Popular machine learning tools like TensorFlow and PyTorch use NumPy to manage input data, handle model parameters, and process the output values.
- **Array Manipulation:** NumPy allows you to create, resize, slice, index, stack, split, and combine arrays.
- **Finance & Economics:** NumPy is used for financial analysis, including portfolio optimization, risk assessment, time series analysis, and statistical modelling.
- **Image & Signal Processing:** NumPy helps process and analyze images and signals for various applications.
- **Data Visualization:** NumPy independently does not create visualizations, but it works with libraries like **Matplotlib** and **Seaborn** to generate charts and graphs from numerical data.

## Example: 1
Checking NumPy Version

```
import numpy as np

print(np.__version__)
```

## Example: 2
Create a NumPy array:

```
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
```

Output:
```
[1 2 3 4 5]

print(type(arr)) => <class 'numpy.ndarray'>
```

## Dimensions in Arrays
A dimension in arrays is one level of array depth (nested arrays).

**nested array:** are arrays that have arrays as their elements.

**Scalar:** A scalar is a single value — just one number, string, or boolean.It has no dimensions — it's just a standalone value.
```
x = 5          # scalar (integer)
y = 3.14       # scalar (float)
z = "hello"    # scalar (string)
```

**Array:**An array is a collection of values — it can hold many numbers or elements, and it has one or more dimensions.
```
import numpy as np

a = np.array([1, 2, 3])         # 1D array (vector)
b = np.array([[1, 2], [3, 4]])  # 2D array (matrix)
```

An array can be:
- **1D** (like a list)
- **2D** (like a table or matrix)
- **nD** (higher-dimensional)

## Computation
Computation means carrying out calculations — it’s the process of solving problems using mathematical operations (addition, multiplication, etc.), often with a computer.

## Matrix (Mathematical Structure)
A matrix is a grid of numbers arranged in rows and columns.
**Example:**

```
A = | 1  2 |
    | 3  4 |
```
- This is a **2x2** matrix (2 rows, 2 columns).

```
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
```

## Types of matrix computations:
- **Addition:** A + B
- **Scalar multiplication:** 3 * A
- **Matrix multiplication:** np.dot(A, B) or A @ B
- **Transpose:** A.T
- **Inverse:** np.linalg.inv(A) (if invertible)

In AI/ML, matrices are everywhere:
    - Images are matrices of pixels
    - Neural networks use weight matrices
    - Data tables are often treated as matrices


## 0-D Arrays:

0-D arrays, or Scalars, are the elements in an array. Each value in an array is a 0-D array.

## Example: 3
Create a 0-D array with value 42

```
import numpy as np

arr = np.array(42)

print(arr)
```

**Output:**
```
42
```

## 1-D Arrays:
An array that has 0-D arrays as its elements is called uni-dimensional or 1-D array.

These are the most common and basic arrays.

## Example: 4
Create a 1-D array containing the values 1,2,3,4,5

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)
```

**Output:**
```
[1 2 3 4 5]
```

## 2-D Arrays
An array that has 1-D arrays as its elements is called a 2-D array.
These are often used to represent matrix or 2nd order tensors.


NumPy has a whole sub module dedicated towards matrix operations called **numpy.mat**

## Example: 5
Create a 2-D array containing two arrays with the values 1,2,3 and 4,5,6

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)
```

**Output:**
```
[[1 2 3]
 [4 5 6]]
```

## 3-D arrays
An array that has 2-D arrays (matrices) as its elements is called 3-D array.
These are often used to represent a 3rd order tensor.

## Example: 6

Create a 3-D array with two 2-D arrays, both containing two arrays with the values 1,2,3 and 4,5,6

```
import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr)
```

**Output:**
```
[[[1 2 3]
  [4 5 6]]

 [[1 2 3]
  [4 5 6]]]
```

## Check Number of Dimensions?
NumPy Arrays provides the ndim attribute that returns an integer that tells us how many dimensions the array have.

## Example: 7

Check how many dimensions the arrays have:

```
import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)
```

**Output:**
```
0
1
2
3
```

## Higher Dimensional Arrays
An array can have any number of dimensions.

When the array is created, you can define the number of dimensions by using the ndmin argument.

## Example: 8

Create an array with 5 dimensions and verify that it has 5 dimensions:

- NumPy support maximum array of dimensions  is = 64
  MAXDIMS (=64)

```
import numpy as np

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('number of dimensions :', arr.ndim)
```

**Output:**
```
[[[[[1 2 3 4]]]]]
number of dimensions : 5
```

- 5th dim: In this array the innermost dimension (5th dim) has 4 elements.
- 4th dim: the 4th dim has 1 element that is the **vector**.
- 3rd dim: the 3rd dim has 1 element that is the **matrix** with the **vector**
- 2nd dim: the 2nd dim has 1 element that is 3D array.
- 1st dim: 1st dim has 1 element that is a 4D array.

## Key Concept: N-D Arrays Don't Automatically Treat Contents as Matrices

In NumPy:
    - A matrix is just a 2D array (with shape (rows, cols)).
    - A vector is a 1D array (shape (n,)).
    - A scalar has shape () (0D).

- Everything beyond 2D is a **tensor** — and all dimensions are just levels of **nesting**.
- NumPy only treats an array as a **matrix if it’s 2D**, like: [[1, 2], [3, 4]].
- Does NumPy Consider a 3D Array a Matrix?
    - NumPy does not consider a **3D array** a **matrix**.
    - Why?
        - In linear algebra, a matrix is strictly a 2D structure: it has rows and columns.

## NumPy follows this convention.

```
Shape	            Interpretation
(3,)	            1D array (vector)
(3, 4)	            2D array (matrix)
(2, 3, 4)	        3D array (tensor)
(1, 1, 1, 1, 4)	    5D tensor
```

**A 3D array in NumPy is:**
A stack of matrices (or a cube of numbers).

Example:

```
import numpy as np

arr = np.array([
  [[1, 2], [3, 4]],
  [[5, 6], [7, 8]]
])
```

**Shape:**

```
(2, 2, 2)
```
**Interpretation:**
    - 2 matrices
    - Each matrix is 2x2


# Matrix vs Tensor in NumPy
```
Concept	        Description	                                    NumPy term
Scalar	        0D single value	                                np.array(5)
Vector	        1D array	                                    np.array([1,2])
Matrix	        2D array (rows × cols)	                        np.array([[1,2], [3,4]])
Tensor	        3D+ array (e.g., 3D, 4D...)	                    np.array([[[...]]])
```


# NumPy Array Indexing
Array indexing is the same as accessing an array element.
You can access an array element by referring to its index number.
The indexes in NumPy arrays start with 0, meaning that the first element has index 0, and the second has index 1 etc.

**What is Indexing?**
**Indexing** is how you access elements inside a NumPy array using their **position** (like in a list). NumPy supports powerful indexing for:

- 1D, 2D, 3D+ arrays
- Slicing
- Boolean conditions
- Fancy indexing

## ✅ 1D Array Indexing

```
import numpy as np
a = np.array([10, 20, 30, 40, 50])
```
**Access:**

```
print(a[0])
10

print(a[-1])
50  (last element)
```

## ✅ 2D Array Indexing (Matrix)

```
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
```

```
1st row:

b[0, 0]  # (1st row, 1st column)
b[0, 1]   # (1st row, 2nd column), like wise.

2nd row:

b[1, 0]   # (2nd row, 1st column)
b[1, 2]  # (2nd row, 3rd column)

3rd row:

b[2, 0]   # (3rd row, 1st column)
b[2, 1]   # (3rd row, 2nd column)

whole 3rd row:

b[2]     # → [7, 8, 9] (whole 3rd row)

whole 2nd column:

b[:, 1]  # → [2, 5, 8] (whole 2nd column)

```

## ✅ 3D Array Indexing (Tensor)

```
c = np.array([
  [[1, 2], [3, 4]],
  [[5, 6], [7, 8]]
])
```

**Shape:** (2, 2, 2)

**Access:**

```
c[0, 0, 0] -> 1

c[0, 0, 1] -> 2

c[1, 1, 0] -> 7
```
# NumPy Array Slicing
Slicing arrays

Slicing in python means taking elements from one given index to another given index.
We pass slice instead of index like this: [start:end].
We can also define the step, like this: [start:end:step].
If we don't pass start its considered 0
If we don't pass end its considered length of array in that dimension
If we don't pass step its considered 1


## ✂️ Slicing

```
a = np.array([10, 20, 30, 40, 50])
a[1:4]     # → [20 30 40]
a[:3]      # → [10 20 30]
a[::2]     # → [10 30 50] (every 2nd element)
```

**2D slicing:**

```
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

b[0:2, 1:]  # → [[2, 3], [5, 6]]

- 0:2 → select rows 0 and 1 (i.e., the first two rows)
- 1: → select columns starting from index 1 (i.e., the second and third columns) 
```


## 🧠 Boolean Indexing

```
a = np.array([1, 2, 3, 4, 5])
a[a > 3]  # → [4 5]
```


## Negative Slicing
Use the minus operator to refer to an index from the end:

Example
Slice from the index 3 from the end to index 1 from the end:

```
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])

Output: [5 6]
```
## STEP
Use the step value to determine the step of the slicing:

Example
Return every other element from index 1 to index 5:

```
arr = np.array([1, 2, 3, 4, 5, 6, 7])

Output: [2 4]
```

Example
Return every other element from the entire array:

```
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[::2])

Output: [1 3 5 7]
```

## Slicing 2-D Arrays

Example
From the second element, slice elements from index 1 to index 4 (not included):

```
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4])

Output: [7 8 9]
```

Example
From both elements, return index 2:

```
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 2])

Output: [3 8]
```

Example
From both elements, slice index 1 to index 4 (not included), this will return a 2-D array:

```
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 1:4])

Output: [[2 3 4]
        [7 8 9]]
```


# NumPy Data Types

## Data Types in Python
By default Python have these data types:

- **strings** - used to represent text data, the text is given under quote marks. e.g. "ABCD"
- **integer** - used to represent integer numbers. e.g. -1, -2, -3
- **float** - used to represent real numbers. e.g. 1.2, 42.42
- **boolean** - used to represent True or False.
- **complex** - used to represent complex numbers. e.g. 1.0 + 2.0j, 1.5 + 2.5j

## Data Types in NumPy
NumPy has some extra data types, and refer to data types with one character, like i for integers, u for unsigned integers etc.

Below is a list of all data types in NumPy and the characters used to represent them.

- **i** - integer
- **b** - boolean
- **u** - unsigned integer
- **f** - float
- **c** - complex float
- **m** - timedelta
- **M** - datetime
- **O** - object
- **S** - string
- **U** - unicode string
- **V** - fixed chunk of memory for other type ( void )

## Checking the Data Type of an Array
The NumPy array object has a property called **dtype** that returns the data type of the array:

Example:

```
import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr.dtype)
```

## Converting Data Type on Existing Arrays
The best way to change the data type of an existing array, is to make a copy of the array with the **astype()** method.

The **astype()** function creates a copy of the array, and allows you to specify the data type as a parameter.

The data type can be specified using a string, like **'f'** for float, **'i'** for integer etc. or you can use the data type directly like **float** for float and **int** for integer.


Example
Change data type from float to integer by using 'i' as parameter value:

```
import numpy as np

arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('i')

print(newarr)
print(newarr.dtype)
```

# NumPy Array Copy vs View

## The Difference Between Copy and View
The main difference between a copy and a view of an array is that the copy is a new array, and the view is just a view of the original array.

The copy owns the data and any changes made to the copy will not affect original array, and any changes made to the original array will not affect the copy.

The view does not own the data and any changes made to the view will affect the original array, and any changes made to the original array will affect the view.


## COPY:

ExampleGet
Make a copy, change the original array, and display both arrays:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr)
print(x)


Output:
[42  2  3  4  5]
[1 2 3 4 5]
```

## VIEW:
Example
Make a view, change the original array, and display both arrays:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42

print(arr)
print(x)


Output:
[42  2  3  4  5]
[42  2  3  4  5]
```

Make a view, change the view, and display both arrays:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31

print(arr)
print(x)

Output:
[31  2  3  4  5]
[31  2  3  4  5]
```

## Check if Array Owns its Data
As mentioned above, copies owns the data, and views does not own the data, but how can we check this?

Every NumPy array has the attribute **base** that returns **None** if the array owns the data.
Otherwise, the base  attribute refers to the original object.

Example

```
Print the value of the base attribute to check if an array owns it's data or not:

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(x.base)
print(y.base)

Output:
None
[1 2 3 4 5]
```

# NumPy Array Shape
Shape of an Array

The shape of an array is the number of elements in each dimension.

## Get the Shape of an Array
NumPy arrays have an attribute called **shape** that returns a tuple with each index having the number of corresponding elements.

Example
Print the shape of a 2-D array:

```
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]]
                )

print(arr.shape)

Output:
(2, 4)


The shape tells us:

2 → there are 2 rows → the first dimension (axis 0)

4 → each row has 4 elements → the second dimension (axis 1)

[
  [1, 2, 3, 4],  ← 1st row
  [5, 6, 7, 8]   ← 2nd row
]


    Columns →
    0   1   2   3
R  +---------------
o  | 1   2   3   4
w  | 5   6   7   8
s
↓

```

Example
Create an array with 5 dimensions using ndmin using a vector with values 1,2,3,4 and verify that last dimension has value 4:

```
import numpy as np

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)

Output:
[[[[[1 2 3 4]]]]]
shape of array : (1, 1, 1, 1, 4)
```

# NumPy Array Reshaping
Reshaping arrays

Reshaping means changing the shape of an array.

The shape of an array is the number of elements in each dimension.

By reshaping we can add or remove dimensions or change number of elements in each dimension.

## Reshape From 1-D to 2-D

**Example**
Convert the following 1-D array with 12 elements into a 2-D array.

The outermost dimension will have 4 arrays, each with 3 elements:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)

Output:
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```

## Reshape From 1-D to 3-D

Example
Convert the following 1-D array with 12 elements into a 3-D array.

The outermost dimension will have 2 arrays that contains 3 arrays, each with 2 elements:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr)

Output:
[[[ 1  2]
  [ 3  4]
  [ 5  6]]

 [[ 7  8]
  [ 9 10]
  [11 12]]]
```

## Can We Reshape Into any Shape?
Yes, as long as the elements required for reshaping are equal in both shapes.

We can reshape an 8 elements 1D array into 4 elements in 2 rows 2D array but we cannot reshape it into a 3 elements 3 rows 2D array as that would require 3x3 = 9 elements.

Example
Try converting 1D array with 8 elements to a 2D array with 3 elements in each dimension (will raise an error):

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(3, 3)

print(newarr)

Output:
ValueError: cannot reshape array of size 8 into shape (3,3)
```

## Returns Copy or View?

Example
Check if the returned array is a copy or a view:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(arr.reshape(2, 4).base)


Output:
[1 2 3 4 5 6 7 8]  -> The example above returns the original array, so it is a view.

```

## Unknown Dimension

You are allowed to have one "unknown" dimension.

Meaning that you do not have to specify an exact number for one of the dimensions in the reshape method.

Pass -1 as the value, and NumPy will calculate this number for you.

Example
Convert 1D array with 8 elements to 3D array with 2x2 elements:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(2, 2, -1)

print(newarr)

Output:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
```

## Flattening the arrays

Flattening array means converting a multidimensional array into a 1D array.

We can use **reshape(-1)** to do this.

Example

Convert the 2D array into a 1D array:

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)

Output:
[1 2 3 4 5 6]
```

Note: There are a lot of functions for changing the shapes of arrays in numpy **flatten**, **ravel** and also for rearranging the elements **rot90, flip, fliplr, flipud** etc. These fall under Intermediate to Advanced section of numpy.


## NumPy Array Iterating

**Iterating Arrays**

Iterating means going through elements one by one.

As we deal with multi-dimensional arrays in numpy, we can do this using basic for loop of python.

If we iterate on a 1-D array it will go through each element one by one.

Example
Iterate on the elements of the following 1-D array:

```
import numpy as np

arr = np.array([1, 2, 3])

for x in arr:
  print(x)

Output:
1
2
3
```

**Iterating 2-D Arrays**

In a 2-D array it will go through all the rows.

Example
Iterate on the elements of the following 2-D array:

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x)

Output:
[1 2 3]
[4 5 6]
```

If we iterate on a n-D array it will go through n-1th dimension one by one.

To return the actual values, the scalars, we have to iterate the arrays in each dimension.

Example
Iterate on each scalar element of the 2-D array:

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  for y in x:
    print(y)

Output:
1
2
3
4
5
6
```

## Iterating 3-D Arrays
In a 3-D array it will go through all the 2-D arrays.

Example
Iterate on the elements of the following 3-D array:

```
import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x)

Output:
[[1 2 3]
 [4 5 6]]
[[ 7  8  9]
 [10 11 12]]
```

To return the actual values, the scalars, we have to iterate the arrays in each dimension.

Example
Iterate down to the scalars:

```
import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  for y in x:
    for z in y:
      print(z)

Output:
1
2
3
4
5
6
7
8
9
10
11
12
```

## Iterating Arrays Using nditer()

The function **nditer()** is a helping function that can be used from very basic to very advanced iterations. It solves some basic issues which we face in iteration, lets go through it with examples.

Iterating on Each Scalar Element

In basic for loops, iterating through each scalar of an array we need to use n for loops which can be difficult to write for arrays with very high dimensionality.

Example
Iterate through the following 3-D array:

```
import numpy as np

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
  print(x)

Output:
1
2
3
4
5
6
7
8
```

## Iterating Array With Different Data Types
We can use **op_dtypes** argument and pass it the expected datatype to change the datatype of elements while iterating.

NumPy does not change the data type of the element in-place (where the element is in array) so it needs some other space to perform this action, that extra space is called buffer, and in order to enable it in **nditer()** we pass flags=['buffered'].

Example
Iterate through the array as a string:

```
import numpy as np

arr = np.array([1, 2, 3])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)

Output:
np.bytes_(b'1')
np.bytes_(b'2')
np.bytes_(b'3')
```

## Iterating With Different Step Size
We can use filtering and followed by iteration.

Example
Iterate through every scalar element of the 2D array skipping 1 element:

```
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
  print(x)

Output:
1
3
5
7
```

## Enumerated Iteration Using ndenumerate()

Enumeration means mentioning sequence number of somethings one by one.

Sometimes we require corresponding index of the element while iterating, the **ndenumerate()** method can be used for those usecases.

Example
Enumerate on following 1D arrays elements:

```
import numpy as np

arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr):
  print(idx, x)


Output:
(0,) 1
(1,) 2
(2,) 3
```

Example
Enumerate on following 2D array's elements:

```
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr):
  print(idx, x)

Output:
(0, 0) 1
(0, 1) 2
(0, 2) 3
(0, 3) 4
(1, 0) 5
(1, 1) 6
(1, 2) 7
(1, 3) 8
```

# NumPy Joining Array

## Joining NumPy Arrays
Joining means putting contents of two or more arrays in a single array.

In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.

We pass a sequence of arrays that we want to join to the **concatenate()** function, along with the axis. If axis is not explicitly passed, it is taken as 0.


ExampleGet your own Python Server
Join two arrays

```
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))

print(arr)

Output:
[1 2 3 4 5 6]
```

Example
Join two 2-D arrays along rows (axis=1):

```
import numpy as np

arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)

Output:
[[1 2 5 6]
 [3 4 7 8]]

```

## Joining Arrays Using Stack Functions
Stacking is same as concatenation, the only difference is that stacking is done along a new axis.

We can concatenate two 1-D arrays along the second axis which would result in putting them one over the other, ie. stacking.

We pass a sequence of arrays that we want to join to the **stack()** method along with the axis. If axis is not explicitly passed it is taken as 0.

Example

```
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)


Output:
[[1 4]
 [2 5]
 [3 6]]

```

## Stacking Along Rows
NumPy provides a helper function: **hstack()** to stack along rows.

Example

```
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))

print(arr)

Output:
[1 2 3 4 5 6]

```

## Stacking Along Columns
NumPy provides a helper function: **vstack()**  to stack along columns.

Example

```
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))

print(arr)

Output:
[[1 2 3]
 [4 5 6]]

```

## Stacking Along Height (depth)
NumPy provides a helper function: **dstack()** to stack along height, which is the same as depth.

Example

```
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.dstack((arr1, arr2))

print(arr)


Output:
[[[1 4]
  [2 5]
  [3 6]]]

```

# NumPy Splitting Array

## Splitting NumPy Arrays

Splitting is reverse operation of Joining.

Joining merges multiple arrays into one and Splitting breaks one array into multiple.

We use **array_split()** for splitting arrays, we pass it the array we want to split and the number of splits.

Example
Split the array in 3 parts:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)


Output:
[array([1, 2]), array([3, 4]), array([5, 6])]
```


If the array has less elements than required, it will adjust from the end accordingly.

Example
Split the array in 4 parts:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 4)

print(newarr)


Output:
[array([1, 2]), array([3, 4]), array([5]), array([6])]
```

**Note:** We also have the method **split()** available but it will not adjust the elements when elements are less in source array for splitting like in example above, **array_split()** worked properly but **split()** would fail.


## Split Into Arrays

The return value of the **array_split()** method is an array containing each of the split as an array.

If you split an array into 3 arrays, you can access them from the result just like any array element:

Example
Access the splitted arrays:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr[0])
print(newarr[1])
print(newarr[2])

Output:
[1 2]
[3 4]
[5 6]

```

## Splitting 2-D Arrays
Use the same syntax when splitting 2-D arrays.

Use the **array_split()** method, pass in the array you want to split and the number of splits you want to do.

Example
Split the 2-D array into three 2-D arrays.

```
import numpy as np

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

newarr = np.array_split(arr, 3)

print(newarr)

Output:
[array([[1, 2],
       [3, 4]]), array([[5, 6],
       [7, 8]]), array([[ 9, 10],
       [11, 12]])]

```
The example above returns three 2-D arrays.


Example
Split the 2-D array into three 2-D arrays.

```


[array([[1, 2, 3],
       [4, 5, 6]]), array([[ 7,  8,  9],
       [10, 11, 12]]), array([[13, 14, 15],
       [16, 17, 18]])]


Output:
[array([[1, 2, 3],
       [4, 5, 6]]), array([[ 7,  8,  9],
       [10, 11, 12]]), array([[13, 14, 15],
       [16, 17, 18]])]

```

The example above returns three 2-D arrays.

In addition, you can specify which axis you want to do the split around.

The example below also returns three 2-D arrays, but they are split along the row (axis=1).

Example
Split the 2-D array into three 2-D arrays along rows.

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3, axis=1)

print(newarr)

Output:


[array([[ 1],
       [ 4],
       [ 7],
       [10],
       [13],
       [16]]), array([[ 2],
       [ 5],
       [ 8],
       [11],
       [14],
       [17]]), array([[ 3],
       [ 6],
       [ 9],
       [12],
       [15],
       [18]])]

```

An alternate solution is using **hsplit()** opposite of **hstack()**

Example
Use the hsplit() method to split the 2-D array into three 2-D arrays along rows.

```
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.hsplit(arr, 3)

print(newarr)


Output:
[array([[ 1],
       [ 4],
       [ 7],
       [10],
       [13],
       [16]]), array([[ 2],
       [ 5],
       [ 8],
       [11],
       [14],
       [17]]), array([[ 3],
       [ 6],
       [ 9],
       [12],
       [15],
       [18]])]

```

**Note:** Similar alternates to **vstack()** and **dstack()** are available as **vsplit()** and **dsplit()**.



# NumPy Searching Arrays

## Searching Arrays
You can search an array for a certain value, and return the indexes that get a match.

To search an array, use the **where()** method.

Example
Find the indexes where the value is 4:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)

print(x)

Output:
(array([3, 5, 6]),)  -> Which means that the value 4 is present at index 3, 5, and 6.

```

Example
Find the indexes where the values are even:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 0)

print(x)

Output:
(array([1, 3, 5, 7]),)

```

Example
Find the indexes where the values are odd:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 1)

print(x)

Output:
(array([0, 2, 4, 6]),)

```

## Search Sorted
There is a method called **searchsorted()** which performs a binary search in the array, and returns the index where the specified value would be inserted to maintain the search order.


The **searchsorted()** method is assumed to be used on sorted arrays.

Example
Find the indexes where the value 7 should be inserted:

```
import numpy as np

arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7)

print(x)


Output:
1        -> The number 7 should be inserted on index 1 to remain the sort order.

```

The method starts the search from the left and returns the first index where the number 7 is no longer larger than the next value.


## Search From the Right Side
By default the left most index is returned, but we can give side='right' to return the right most index instead.

Example
Find the indexes where the value 7 should be inserted, starting from the right:

```
import numpy as np

arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7, side='right')

print(x)

Output:
2        -> The number 7 should be inserted on index 2 to remain the sort order.

```

The method starts the search from the right and returns the first index where the number 7 is no longer less than the next value.

## Multiple Values
To search for more than one value, use an array with the specified values.

Example
Find the indexes where the values 2, 4, and 6 should be inserted:

```
import numpy as np

arr = np.array([1, 3, 5, 7])

x = np.searchsorted(arr, [2, 4, 6])

print(x)


Output:
[1 2 3]         

arr = [1, 3, 5, 7]
        ↑  ↑  ↑  ↑
Index:  0  1  2  3

```

The return value is an array: [1 2 3] containing the three indexes where 2, 4, 6 would be inserted in the original array to maintain the order.

# NumPy Sorting Arrays

## Sorting Arrays
Sorting means putting elements in an ordered sequence.

Ordered sequence is any sequence that has an order corresponding to elements, like numeric or alphabetical, ascending or descending.

The NumPy ndarray object has a function called sort(), that will sort a specified array.

Example
Sort the array:

```
import numpy as np

arr = np.array([3, 2, 0, 1])

print(np.sort(arr))

Output:
[0 1 2 3]

```

Note: This method returns a copy of the array, leaving the original array unchanged.


Example
Sort the array alphabetically:

```
import numpy as np

arr = np.array(['banana', 'cherry', 'apple'])

print(np.sort(arr))


Output:

['apple' 'banana' 'cherry']

```

Example
Sort a boolean array:


```
import numpy as np

arr = np.array([True, False, True])

print(np.sort(arr))


Output:
[False  True  True]

```

## Sorting a 2-D Array
If you use the sort() method on a 2-D array, both arrays will be sorted:

Example
Sort a 2-D array:

```
import numpy as np

arr = np.array([[3, 2, 4], [5, 0, 1]])

print(np.sort(arr))

Output:
[[2 3 4]
 [0 1 5]]

```

# NumPy Filter Array

## Filtering Arrays
Getting some elements out of an existing array and creating a new array out of them is called filtering.

In NumPy, you filter an array using a boolean index list.

A boolean index list is a list of booleans corresponding to indexes in the array.

If the value at an index is **True** that element is contained in the filtered array, if the value at that index is **False** that element is excluded from the filtered array.


Example
Create an array from the elements on index 0 and 2:

```
import numpy as np

arr = np.array([41, 42, 43, 44])

x = [True, False, True, False]

newarr = arr[x]

print(newarr)

Output:
[41 43]


The example above will return [41, 43], why?

Because the new array contains only the values where the filter array had the value True, in this case, index 0 and 2.

```


## Creating the Filter Array
In the example above we hard-coded the True and False values, but the common use is to create a filter array based on conditions.



Example
Create a filter array that will return only values higher than 42:

```
import numpy as np

arr = np.array([41, 42, 43, 44])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)

Output:
[False, False, True, True]
[43 44]

```

Example
Create a filter array that will return only even elements from the original array:


```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is completely divisble by 2, set the value to True, otherwise False
  if element % 2 == 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)

Output:
[False, True, False, True, False, True, False]
[2 4 6]

```

## Creating Filter Directly From Array
The above example is quite a common task in NumPy and NumPy provides a nice way to tackle it.

We can directly substitute the array instead of the iterable variable in our condition and it will work just as we expect it to.

Example
Create a filter array that will return only values higher than 42:

```
import numpy as np

arr = np.array([41, 42, 43, 44])

filter_arr = arr > 42

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)


Output:
[False False  True  True]
[43 44]

```

Example
Create a filter array that will return only even elements from the original array:

```
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

filter_arr = arr % 2 == 0

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)

Output:
[False  True False  True False  True False]
[2 4 6]

```

# 🧠 1. Array Creation
Used for creating datasets, weight matrices, etc.

```
np.array()         # Convert list to array
np.zeros(), np.ones()  # Initialize weights
np.eye()           # Identity matrix
np.arange(), np.linspace()  # Range of values
```

# ⚙️ 2. Array Operations & Math
Used for vectorized operations (fast!)

```
+ - * /            # Element-wise ops
np.dot(), np.matmul()  # Matrix multiplication
np.sum(), np.mean(), np.std(), np.var()
np.max(), np.min(), np.argmax(), np.argmin()
np.exp(), np.log(), np.sqrt()
np.clip()          # Limit values (e.g. activation limits)
```

# 📐 3. Reshaping & Indexing
Used to prepare data for ML models (like reshaping images, slicing time series)

```
np.reshape(), np.ravel(), np.flatten()
np.transpose(), np.swapaxes()
np.concatenate(), np.stack(), np.split()
np.where(), np.argwhere()
np.unique()
```

# 📊 4. Random Numbers (for initializing weights, data splitting, etc.)

```
np.random.rand(), np.random.randn()   # Uniform & normal dist
np.random.randint()
np.random.shuffle(), np.random.permutation()
np.random.seed()   # Set seed for reproducibility
```

# 📉 5. Logical & Boolean Indexing
Used for masking, filtering, conditional operations.

```
a[a > 0]                  # Filter positives
np.any(), np.all()
np.isfinite(), np.isnan() # Data cleaning
```

# 🧪 6. Linear Algebra (used in neural networks, PCA, etc.)

```
np.linalg.inv()      # Inverse
np.linalg.pinv()     # Pseudo-inverse
np.linalg.norm()     # Vector norms
np.linalg.eig(), np.linalg.svd()
```

# 🧠 Common ML/AI Tasks Using NumPy:

- **Feature scaling / normalization:** np.mean(), np.std()

- **Distance calculations:** np.linalg.norm()

- **Vectorized loss functions** (MSE, Cross Entropy, etc.)

- **Gradient descent implementation**

- **Custom ML algorithms (k-NN, k-means, PCA)**












