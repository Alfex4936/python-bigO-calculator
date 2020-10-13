# Big-O Caculator

A big-O calculator to estimate time complexity of sorting functions.

inspired by : https://github.com/ismaelJimenez/cpp.leastsq

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
    test(function, array="random", limit=True, prtResult=True):
        It will run only specified array test, returns Tuple[str, estimatedTime]

    test_all(function):
        It will run all test cases, prints (best, average, worst cases), returns dict

    runtime(function, array="random", size, epoch=1):
        It will simply returns execution time to sort length of size of test array, returns Tuple[float, List[Any]]

    compare(function1, function2, array, size, epoch=3):
        It will compare two functions on {array} case, returns dict
```

## Methods parameters
```py
test(**args):
    functionName [Callable]: a function to call.
    array [str]: "random", "sorted", "reversed", "partial", "Ksorted", "string", "almost_equal", "equal", "hole".
    limit [bool] = True: To break before it takes "forever" to sort an array. (ex. selectionSort)
    prtResult [bool] = True: Whether to print result by itself

test_all(**args):
    functionName [Callable]: a function to call.

runtime(**args):
    functionName [Callable]: a function to call.
    array: "random", "sorted", "partial", "reversed", "Ksorted" ,
        "hole", "equal", "almost_equal" or your custom array.
    size [int]: How big test array should be.
    epoch [int]: How many tests to run and calculte average.
    prtResult (bool): Whether to print the result by itself. (default = True)

compare(**args):
    function1 [Callable]: a function to compare.
    function2 [Callable]: a function to compare.
    array [str]|[List]: "random", "sorted", "partial", "reversed", "Ksorted", 
        "hole", "equal", "almost_equal", "all" or your custom array.
    size [int]: How big test array should be.
```
Info:
    To see the result of function, return the array.

    These methods will also check if the function sorts correctly.

    "K" in Ksorted uses testSize.bit_length().

## Usage

```py
from bigO import bigO
from bigO import algorithm

lib = bigO.bigO()

lib.test(bubbleSort, "random")
lib.test_all(bubbleSort)
lib.runtime(bubbleSort, "random", 5000)
lib.runtime(algorithm.insertSort, "reversed", 32)
lib.compare(algorithm.insertSort, algorithm.bubbleSort, "all", 5000)
```

## Quick Sort Example

```py
from bigO import bigO
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


tester = bigO.bigO()
complexity, time = tester.test(quickSort, "random")
complexity, time = tester.test(quickSort, "sorted")
complexity, time = tester.test(quickSort, "reversed")
complexity, time = tester.test(quickSort, "partial")
complexity, time = tester.test(quickSort, "Ksorted")

''' Result
Running quickSort(random array)...
Completed quickSort(random array): O(nlog(n))
Time took: 0.35816s
Running quickSort(sorted array)...
Completed quickSort(sorted array): O(nlog(n))
Time took: 0.37821s
Running quickSort(reversed array)...
Completed quickSort(reversed array): O(nlog(n))
Time took: 0.38500s
Running quickSort(partial array)...
Completed quickSort(partial array): O(nlog(n))
Time took: 0.35820s
Running quickSort(Ksorted array)...
Completed quickSort(ksorted array): O(nlog(n))
Time took: 0.38140s
'''
```

## Selection Sort Example
```py
from bigO import bigO

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


tester = bigO.bigO()
complexity, time = tester.test(selectionSort, "random")
complexity, time = tester.test(selectionSort, "sorted")
complexity, time = tester.test(selectionSort, "reversed")
complexity, time = tester.test(selectionSort, "partial")
complexity, time = tester.test(selectionSort, "Ksorted")

''' Result
Running selectionSort(random array)...
Completed selectionSort(random array): O(n^2)
Time took: 4.04027s
Running selectionSort(reversed array)...
Completed selectionSort(reversed array): O(n^2)
Time took: 4.04918s
Running selectionSort(sorted array)...
Completed selectionSort(sorted array): O(n^2)
Time took: 3.97238s
Running selectionSort(partial array)...
Completed selectionSort(partial array): O(n^2)
Time took: 4.02878s
Running selectionSort(Ksorted array)...
Completed selectionSort(ksorted array): O(n^2)
Time took: 4.05617s
'''
```

## test_all(mySort) Example

We can test all "random", "sorted", "reversed", "partial", "Ksorted", "almost_equal" in a row
and it shows, best, average, worst time complexity

```py
from bigO import bigO

lib = bigO.bigO()

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
array: "random", "sorted", "partial", "reversed", "Ksorted" ,
        "hole", "equal", "almost_equal" or your custom array.

```py
from bigO import bigO
from bigO import algorithm

lib = bigO.bigO()

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

```py
lib = bigO.bigO()

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

## Test arrays sample (size = 20)
Results vary.

```py
random = [15, 15, -11, -16, -19, -16, -14, 14, 19, 2, 18, -10, 5, -17, -4, -2, 9, 12, 8, 12]
string = ['rwe55pi8hkwpjv5rhhoo', '5ecvybo5xi8p25wanh3t', '9qloe709sonjuun90p77', 'jqc06iabwk3v5utqo09d', 'shm2uweb4dsgbx14hts3', '07eivto20vmvp0nsa6b3', 'vyoqn5pt2swkuftv7g0p', 'pw06n5utxsd7j1u2kv82', 'k8trosl40h7qvozfjhex', 'r4zvaqnblc3uv6x95uvh', 'qsxliu3zm7z20gtjpo50', 'wg81sdzhc5wuanrk20n0', 'iioyowuktvbq71tsx30p', 'cazu363i51f55ccw3dol', '2hupx2egkcgpx6byeh3f', 'njodnkvuf12cfm5kp4f5', 'jm919g477ivcambii16t', 'wnlbj3hs33rilovbzhyq', '5in234a20dbz5zv69qx4', 'hysowkb230ts7fcwizmb']
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
Visit [here](https://github.com/Alfex4936/python-bigO-calculator/tree/master/benchmarks) to more results

<div align="center">
<p>
    <img src="https://raw.githubusercontent.com/Alfex4936/python-bigO-calculator/master/benchmarks/random.webp">
</p>
</div>
