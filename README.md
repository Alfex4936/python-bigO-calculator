# Big-O Caculator

A big-O calculator to estimate time complexity of sorting functions.

inspired by : https://github.com/ismaelJimenez/cpp.leastsq

## Installation

```bash
pip install big-O-calculator
```

## Usage

You can call which array to test

(n : [10, 100, 1_000, 10_000, 100_000])

```py
Big-O calculator

Args:
    functionName [Callable]: a function to call
    
    array [str]: "random", "sorted", "reversed", "partial", "Ksorted"
    
Returns:
    complexity (str) : ex) O(n) |
    time (float) : Time took to sort all 5 different arrays in second (max=100,000)

Info:
    To see the result of function, return the array.

    K in Ksorted will use testSize//2
  
```

```py
from bigO import bigO

tester = bigO.bigO()

tester.test(bubbleSort, "sorted")
```

## Example

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
