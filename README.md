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
    functionName ([Callable]): a function to call
    
    array ([string]): "random", "sorted", "reversed", "partial", "Ksorted"

Warn:
    To see the result of function, return the array.

    K in Ksorted will use testSize//2
  
```

```py
from bigO import bigO

def countSort(arr):  # stable
    # Time Complexity : O(n) | Space Complexity : O(n)
    minValue = min(arr)
    maxValue = max(arr) - minValue

    buckets = [0 for x in range(maxValue + 1)]

    for i in arr:
        buckets[i - minValue] += 1

    index = 0
    for i in range(len(buckets)):
        while buckets[i] > 0:
            arr[index] = i + minValue
            index += 1
            buckets[i] -= 1

    return arr


tester = bigO.bigO()
complexity, _, res = tester.test(countSort, "random")
complexity, _, res = tester.test(countSort, "sorted")
complexity, _, res = tester.test(countSort, "reversed")
complexity, _, res = tester.test(countSort, "partial")
complexity, _, res = tester.test(countSort, "Ksorted")

''' Result
Running countSort(random array)...
Completed countSort(random array): O(N)
Running countSort(sorted array)...
Completed countSort(sorted array): O(N)
Running countSort(reversed array)...
Completed countSort(reversed array): O(N)
Running countSort(partial array)...
Completed countSort(partial array): O(N)
Running countSort(Ksorted array)...
Completed countSort(Ksorted array): O(N)
'''
```
