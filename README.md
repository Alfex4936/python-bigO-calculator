## Table of Contents

<table>
    <tr><td width=40% valign=top>
        
* [Methods](https://github.com/Alfex4936/python-bigO-calculator#methods-parameters)
* [Usage](https://github.com/Alfex4936/python-bigO-calculator#usage)
* [QuickSort example](https://github.com/Alfex4936/python-bigO-calculator#quick-sort-example)
* [test_all() exmaple](https://github.com/Alfex4936/python-bigO-calculator#test_allmysort-example)
* [runtime() exmaple](https://github.com/Alfex4936/python-bigO-calculator#runtimemysort-example)
* [compare() exmaple](https://github.com/Alfex4936/python-bigO-calculator#comparemysort-thissort-example)
* [@isSorted decorator](https://github.com/Alfex4936/python-bigO-calculator#issorted)
* [Array generators](https://github.com/Alfex4936/python-bigO-calculator#array-generators)
* [Test arrays sample](https://github.com/Alfex4936/python-bigO-calculator#test-arrays-sample-size--20)
</td></tr>
</table>

# Big-O Caculator

A big-O calculator to estimate time complexity of sorting functions.

inspired by : https://github.com/ismaelJimenez/cpp.leastsq

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAlfex4936%2Fpython-bigO-calculator&count_bg=%233D6BC8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) [![Downloads](https://pepy.tech/badge/big-o-calculator)](https://pepy.tech/project/big-o-calculator)

## Installation

```bash
pip install big-O-calculator
```

## What it does

You can test time complexity, calculate runtime, compare two sorting algorithms

Results may vary.

(n : [10, 100, 1_000, 10_000, 100_000])

```py
Big-O calculator

Methods:
    def test(function, array="random", limit=True, prtResult=True):
        It will run only specified array test, returns Tuple[str, estimatedTime]

    def test_all(function):
        It will run all test cases, prints (best, average, worst cases), returns dict

    def runtime(function, array="random", size, epoch=1):
        It will simply returns execution time to sort length of size of test array, returns Tuple[float, List[Any]]

    def compare(function1, function2, array, size, epoch=3):
        It will compare two functions on {array} case, returns dict
```

## Methods parameters
```py
def test(**args):
    functionName [Callable]: a function to call.
    array [str]: "random", "big", "sorted", "reversed", "partial", "Ksorted", "string", "almost_equal", "equal", "hole".
    limit [bool] = True: To break before it takes "forever" to sort an array. (ex. selectionSort)
    prtResult [bool] = True: Whether to print result by itself

def test_all(**args):
    functionName [Callable]: a function to call.

def runtime(**args):
    functionName [Callable]: a function to call.
    array: "random", "big", "sorted", "partial", "reversed", "Ksorted" ,
        "hole", "equal", "almost_equal" or your custom array.
    size [int]: How big test array should be.
    epoch [int]: How many tests to run and calculte average.
    prtResult (bool): Whether to print the result by itself. (default = True)

def compare(**args):
    function1 [Callable]: a function to compare.
    function2 [Callable]: a function to compare.
    array [str]|[List]: "random", "big", "sorted", "partial", "reversed", "Ksorted", 
        "hole", "equal", "almost_equal", "all" or your custom array.
    size [int]: How big test array should be.
```
Info:
    To see the result of function, return the array.

    These methods will also check if the function sorts correctly.

    "K" in Ksorted uses testSize.bit_length().

## Usage

```py
from bigO import BigO
from bigO import algorithm

lib = BigO()

lib.test(bubbleSort, "random")
lib.test_all(bubbleSort)
lib.runtime(bubbleSort, "random", 5000)
lib.runtime(algorithm.insertSort, "reversed", 32)
lib.compare(algorithm.insertSort, algorithm.bubbleSort, "all", 5000)
```

## Quick Sort Example

```py
from bigO import BigO
from random import randint

def quickSort(array):  # in-place | not-stable
    """
    Best : O(nlogn) Time | O(logn) Space
    Average : O(nlogn) Time | O(logn) Space
    Worst : O(n^2) Time | O(logn) Space
    """
    if len(array) <= 1:
        return array
    smaller, equal, larger = [], [], []
    pivot = array[randint(0, len(array) - 1)]
    for x in array:
        if x < pivot:
            smaller.append(x)
        elif x == pivot:
            equal.append(x)
        else:
            larger.append(x)
    return quickSort(smaller) + equal + quickSort(larger)


lib = BigO()
complexity = lib.test(quickSort, "random")
complexity = lib.test(quickSort, "sorted")
complexity = lib.test(quickSort, "reversed")
complexity = lib.test(quickSort, "partial")
complexity = lib.test(quickSort, "Ksorted")

''' Result
Running quickSort(random array)...
Completed quickSort(random array): O(nlog(n))

Running quickSort(sorted array)...
Completed quickSort(sorted array): O(nlog(n))

Running quickSort(reversed array)...
Completed quickSort(reversed array): O(nlog(n))

Running quickSort(partial array)...
Completed quickSort(partial array): O(nlog(n))

Running quickSort(Ksorted array)...
Completed quickSort(ksorted array): O(nlog(n))
'''
```

## Selection Sort Example
```py
from bigO import BigO

def selectionSort(array):  # in-place, unstable
    '''
    Best : O(n^2) Time | O(1) Space
    Average : O(n^2) Time | O(1) Space
    Worst : O(n^2) Time | O(1) Space
    '''
    currentIdx = 0
    while currentIdx < len(array) - 1:
        smallestIdx = currentIdx
        for i in range(currentIdx + 1, len(array)):
            if array[smallestIdx] > array[i]:
                smallestIdx = i
        array[currentIdx], array[smallestIdx] = array[smallestIdx], array[currentIdx]
        currentIdx += 1
    return array


lib = BigO()
complexity = lib.test(selectionSort, "random")
complexity = lib.test(selectionSort, "sorted")
complexity = lib.test(selectionSort, "reversed")
complexity = lib.test(selectionSort, "partial")
complexity = lib.test(selectionSort, "Ksorted")

''' Result
Running selectionSort(random array)...
Completed selectionSort(random array): O(n^2)
Running selectionSort(sorted array)...
Completed selectionSort(sorted array): O(n^2)
Running selectionSort(reversed array)...
Completed selectionSort(reversed array): O(n^2)
Running selectionSort(partial array)...
Completed selectionSort(partial array): O(n^2)
Running selectionSort(Ksorted array)...
Completed selectionSort(ksorted array): O(n^2)
'''
```

## test_all(mySort) Example

We can test all "random", "sorted", "reversed", "partial", "Ksorted", "almost_equal" at once,
and it shows, best, average and worst time complexity

```py
from bigO import BigO

lib = BigO()

lib.test_all(bubbleSort)
lib.test_all(insertSort)

result = lib.test_all(selectionSort)

print(result)  # Dictionary


''' Result
Running bubbleSort(tests)
Best : O(n) Time
Average : O(n^2) Time
Worst : O(n^2) Time
Running insertSort(tests)
Best : O(n) Time
Average : O(n^2) Time
Worst : O(n^2) Time
Running selectionSort(tests)
Best : O(n^2) Time
Average : O(n^2) Time
Worst : O(n^2) Time

{'random': 'O(n^2)', 'sorted': 'O(n^2)', 'reversed': 'O(n^2)', 'partial': 'O(n^2)', 'Ksorted': 'O(n^2)'}
'''
```

## runtime(mySort) Example
array: "random", "big", "sorted", "partial", "reversed", "Ksorted",
        "hole", "equal", "almost_equal" or your custom array.

```py
from bigO import BigO
from bigO import algorithm

lib = BigO()

timeTook, result = lib.runtime(algorithm.bubbleSort, "random", 5000)

custom = ["abc", "bbc", "ccd", "ef", "az"]

timeTook, result = lib.runtime(algorithm.bubbleSort, custom)

''' Result
Running bubbleSort(len 5000 random array)
Took 2.61346s to sort bubbleSort(random)

Running bubbleSort(len 5 custom array)
Took 0.00001s to sort bubbleSort(custom)
'''
```

## compare(mySort, thisSort) Example
array: "random", "big", "sorted", "partial", "reversed", "Ksorted",
        "hole", "equal", "almost_equal", "all" or your custom array.

```py
lib = BigO()

result = lib.compare(algorithm.bubbleSort, algorithm.insertSort, "reversed", 5000)
result = lib.compare(algorithm.insertSort, algorithm.insertSortOptimized, "reversed", 5000)
result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "reversed", 50000)
result = lib.compare(algorithm.timSort, algorithm.introSort, "reversed", 50000)
result = lib.compare(sorted, algorithm.introSort, "reversed", 50000)

result = lib.compare(algorithm.bubbleSort, algorithm.insertSort, "all", 5000)

print(result)

'''Result
bubbleSort is 3.6% faster than insertSort on reversed case
Time Difference: 0.04513s
insertSortOptimized is 5959.3% faster than insertSort on reversed case
Time Difference: 1.25974s
quickSortHoare is 153.6% faster than quickSort on reversed case
Time Difference: 0.09869s
introSort is 206.6% faster than timSort on reversed case
Time Difference: 0.12597s
sorted is 12436.9% faster than introSort on reversed case
Time Difference: 0.06862s

Running bubbleSort(tests) vs insertSort(tests)
insertSort is 32.6% faster than bubbleSort on 6 of 8 cases
Time Difference: 0.11975s

{'bubbleSort': 0.4875642249999998, 'insertSort': 0.3678110916666666}
'''
```

## @isSorted
If it sorts correctly, it shows: "mySort sorts correctly."

Otherwise, it shows like,
"mySort doesn't sort correctly."
"At N index: [...100, -72, 121...]

```py
from bigO import BigO
from bigO import utils

@utils.isSorted
def bubbleSort(array):  # in-place | stable
    isSorted = False
    counter = 1  # not correct
    while not isSorted:
        isSorted = True
        for i in range(len(array) - 1 - counter):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False

        counter += 1

    return array


if __name__ == "__main__":
    bubbleSort(BigO.genRandomArray(100))

''' Result
bubbleSort doesn't sort correctly.
At 99 index: [...99, -76]
'''
```

## Array generators
```py
from bigO import BigO

lib = BigO()

arr = lib.genRandomArray(100)
arr = lib.genRandomBigArray(100)
arr = lib.genRandomString(100)
arr = lib.genSortedArray(100)
arr = lib.genReversedArray(100)
arr = lib.genPartialArray(100)
arr = lib.genKsortedArray(100)
arr = lib.genAlmostEqualArray(100)
arr = lib.genEqualArray(100)
arr = lib.genHoleArray(100)

```

## Test arrays sample (size = 20)
Results vary.

```py
random = [15, 15, -11, -16, -19, -16, -14, 14, 19, 2, 18, -10, 5, -17, -4, -2, 9, 12, 8, 12]
randomBig = [-996061023766482, 347955820115093, ...]
string = ['rwe55pi8hkwpjv5rhhoo', '5ecvybo5xi8p25wanh3t', ...]
sorted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
reversed = [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
partial = [-18, 14, 7, -11, 17, 5, 6, 7, 8, 9, 14, 9, -13, 0, 14, -17, -18, -9, -16, 14]
Ksorted = [-4, -5, -6, -7, -8, -9, -10, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
almost_equal = [19, 19, 19, 20, 20, 19, 20, 20, 21, 19, 20, 21, 21, 19, 19, 21, 20, 19, 21, 19]
equal = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
hole = [-7, -7, -7, -7, -7, -7, -9999, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7]
```

## Built-in algorithms list
Visit [here](https://github.com/Alfex4936/python-bigO-calculator/blob/master/bigO/algorithm.py) to see codes

BinaryInsertSort, BubbleSort, CountSort, gnomeSort, heapSort, 
InsertSort, InsertSortOptimized, IntroSort,
mergeSort, quickSort(random pivot), quickSortHoare(Hoare+Tail recur+InsertionSort), timSort(simplified)


## Benchmarks
Visit [here](https://github.com/Alfex4936/python-bigO-calculator/tree/master/benchmarks) for more results

<div align="center">
<p>
    <img src="https://raw.githubusercontent.com/Alfex4936/python-bigO-calculator/master/benchmarks/random.webp">
</p>
</div>
