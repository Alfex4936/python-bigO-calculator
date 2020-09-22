# Big-O Caculator

A big-O calculator to estimate time complexity of sorting functions.

inspired by : https://github.com/ismaelJimenez/cpp.leastsq

## Installation

```bash
pip install big-O-calculator
```

## What it does

You can test time complexity, calculate runtime, compare two sorting algorithms

(n : [10, 100, 1_000, 10_000, 100_000])

```py
Big-O calculator

Methods:
    test(function, array="random", limit=True, prtResult=True): It will run only specified array test, returns Tuple[str, estimatedTime]

    test_all(function): It will run all test cases, prints (best, average, worst cases), returns dict

    runtime(function, array="random", size, epoch=1): It will simply returns execution time to sort length of size of test array, returns Tuple[float, List[Any]]

    compare(function1, function2, array, size, epoch=3): It will compare two functions on {array} case, returns dict

test(**args):
    functionName [Callable]: a function to call.
    
    array [str]: "random", "sorted", "reversed", "partial", "Ksorted", "string".

    limit [bool] = True: To break before it takes "forever" to sort an array. (ex. selectionSort)

    prtResult [bool] = True: Whether to print result by itself
    
Returns:
    complexity [str] : ex) O(n)

    time [float] : Time took to sort all 5 different arrays in second (maxLen=100,000)

Info:
    To see the result of function, return the array.

    K in Ksorted will use testSize.bit_length().
  
```

## Usage

```py
from bigO import bigO
from bigO import algorithm

tester = bigO.bigO()

tester.test(bubbleSort, "sorted")
tester.test_all(bubbleSort)
tester.runtime(bubbleSort)
tester.runtime(algorithm.insertSort)
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

We can test all "random", "sorted", "reversed", "partial", "Ksorted" by one with test_all method

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

result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "random", 50000)
result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "reversed", 50000)
result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "sorted", 50000)
result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "partial", 50000)

print(result)

'''Result
quickSortHoare is faster than quickSort on random case
Time Difference: 0.05841s
quickSortHoare is faster than quickSort on reversed case
Time Difference: 0.09900s
quickSortHoare is faster than quickSort on sorted case
Time Difference: 0.11852s
quickSortHoare is faster than quickSort on partial case
Time Difference: 0.06365s
quickSortHoare is faster than quickSort on Ksorted case
Time Difference: 0.10311s

{'quickSort': 0.16498633333333354, 'quickSortHoare': 0.06188056666666656}
'''
```

## Built-in algorithms list
Visit [here](https://github.com/Alfex4936/python-bigO-calculator/blob/master/bigO/algorithm.py) to see codes

BinaryInsertSort, BubbleSort, CountSort, gnomeSort, heapSort, 
InsertSort, InsertSortOptimized, IntroSort,
mergeSort, quickSort(random pivot), quickSortHoare(Hoare+Tail recur), timSort(simplified)
