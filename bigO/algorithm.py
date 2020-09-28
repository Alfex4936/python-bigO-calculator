from random import randrange


def bubbleSort(array):
    isSorted = False
    counter = 0

    while not isSorted:
        isSorted = True
        for i in range(len(array) - 1 - counter):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False

        counter += 1

    return array


def binarySearch(array, item, start, end):
    if start == end:
        if array[start] > item:
            return start
        else:
            return start + 1
    if start > end:
        return start

    mid = (start + end) // 2

    if array[mid] < item:
        return binarySearch(array, item, mid + 1, end)

    elif array[mid] > item:
        return binarySearch(array, item, start, mid - 1)
    else:
        return mid


def binaryInsertSort(array):
    for index in range(1, len(array)):
        value = array[index]
        pos = binarySearch(array, value, 0, index - 1)
        array = array[:pos] + [value] + array[pos:index] + array[index + 1 :]
    return array


def countSort(arr):  # stable
    # Time Complexity : O(n) | Space Complexity : O(n)
    minValue = min(arr)
    maxValue = max(arr) - minValue

    buckets = [0 for _ in range(maxValue + 1)]

    for i in arr:
        buckets[i - minValue] += 1

    index = 0
    for i in range(len(buckets)):
        while buckets[i] > 0:
            arr[index] = i + minValue
            index += 1
            buckets[i] -= 1

    return arr


def selectionSort(array):
    for currentIdx in range(len(array) - 1):
        smallestIdx = currentIdx
        for i in range(currentIdx + 1, len(array)):
            if array[smallestIdx] > array[i]:
                smallestIdx = i
        array[currentIdx], array[smallestIdx] = array[smallestIdx], array[currentIdx]

    return array


def introSort(array):  # in-place | not-stable
    # Time Complexity O(nlogn) | Space Complexity O(logn)
    maxDepth = 2 * (len(array).bit_length() - 1)
    sizeThreshold = 16
    return introSortHelper(array, 0, len(array), sizeThreshold, maxDepth)


def introSortHelper(array, start, end, sizeThreshold, depthLimit):
    def medianOf3(array, lowIdx, midIdx, highIdx):
        if (array[lowIdx] - array[midIdx]) * (array[highIdx] - array[lowIdx]) >= 0:
            return array[lowIdx]

        elif (array[midIdx] - array[lowIdx]) * (array[highIdx] - array[midIdx]) >= 0:
            return array[midIdx]

        else:
            return array[highIdx]

    def getPartition(array, low, high, pivot):
        i = low
        j = high
        while True:

            while array[i] < pivot:
                i += 1
            j -= 1
            while pivot < array[j]:
                j -= 1
            if i >= j:
                return i
            array[i], array[j] = array[j], array[i]
            i += 1

    while end - start > sizeThreshold:
        if depthLimit == 0:
            return heapSort(array)
        depthLimit -= 1

        median = medianOf3(array, start, start + ((end - start) // 2) + 1, end - 1)
        p = getPartition(array, start, end, median)
        introSortHelper(array, p, end, sizeThreshold, depthLimit)
        end = p

    return insertSort(array, start, end)


def insertSort(array, begin=0, end=None):  # in-place | stable
    """
    Best O(n) Time | O(1) Space
    Average O(n^2) Time | O(1) Space
    Worst (On^2) Time | O(1) Space
    """
    if end == None:
        end = len(array)

    for i in range(begin, end):
        j = i
        while j > begin and array[j - 1] > array[j]:
            array[j], array[j - 1] = array[j - 1], array[j]
            j -= 1

    return array


def insertSortOptimized(array):
    for i in range(1, len(array)):
        if array[i] >= array[i - 1]:
            continue
        for j in range(i):
            if array[i] < array[j]:
                array[j], array[j + 1 : i + 1] = array[i], array[j:i]
                break

    return array


def heapSort(arr):  # in-place | not-stable
    # Time Complexity O(nlogn) | Space Complexity O(1)

    def heapify(arr, n, i):  # Max Heap
        largest = i  # 트리에서 가장 큰 값 찾기
        l = 2 * i + 1  # Left Node
        r = 2 * i + 2  # Right Node

        if l < n and arr[i] < arr[l]:
            largest = l

        if r < n and arr[largest] < arr[r]:
            largest = r

        # root가 최대가 아니면
        # 최대 값과 바꾸고, 계속 heapify
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)

    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        # Heapify root element
        heapify(arr, i, 0)

    return arr


def timSort(arr):  # in-place | stable
    """
    Best : O(n) Time | O(n) Space
    Average : O(nlogn) Time | O(n) Space
    Worst : O(nlogn) Time | O(n) Space
    """

    def calcMinRun(n):
        """Returns the minimum length of a run from 23 - 64 so that
        the len(array)/minrun is less than or equal to a power of 2.

        e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33, ..., 127=>64, 128=>32, ...
        """
        r = 0
        while n >= 64:
            r |= n & 1
            n >>= 1
        return n + r

    def insertSort(array, left, right):
        for i in range(left + 1, right + 1):
            temp = array[i]
            j = i - 1
            while j >= left and array[j] > temp:
                array[j + 1] = array[j]
                j -= 1
            array[j + 1] = temp
        return array

    def fastmerge(array1, array2):
        merged_array = []
        while array1 or array2:
            if not array1:
                merged_array.append(array2.pop())
            elif (not array2) or array1[-1] > array2[-1]:
                merged_array.append(array1.pop())
            else:
                merged_array.append(array2.pop())
        merged_array.reverse()
        return merged_array

    n = len(arr)
    minRun = calcMinRun(n)

    # 32만큼 건너뛰면서 삽입 정렬 실행
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        arr = insertSort(arr, start, end)
    currentSize = minRun

    # minRun이 배열 길이보다 작을 때까지만 minRun * 2 를 한다.
    while currentSize < n:
        for start in range(0, n, currentSize * 2):
            mid = min(n - 1, start + currentSize - 1)
            right = min(start + 2 * currentSize - 1, n - 1)
            merged = fastmerge(
                array1=arr[start : mid + 1], array2=arr[mid + 1 : right + 1]
            )
            arr[start : start + len(merged)] = merged

        currentSize *= 2

    return arr


def quickSort(array):  # in-place | not-stable
    """
    Best : O(nlogn) Time | O(logn) Space
    Average : O(nlogn) Time | O(logn) Space
    Worst : O(nlogn) Time | O(logn) Space

    """
    if len(array) <= 1:
        return array
    smaller, equal, larger = [], [], []
    pivot = array[randrange(0, len(array))]
    for x in array:
        if x < pivot:
            smaller.append(x)
        elif x == pivot:
            equal.append(x)
        else:
            larger.append(x)
    return quickSort(smaller) + equal + quickSort(larger)


def quickSortHoare(array, low=0, high=None):  # in-place | not-stable
    """
    QuickSort using tail recursive + insertion sort + hoare
    
    Best : O(nlogn) Time | O(1) Space
    Average : O(nlogn) Time | O(1) Space
    Worst : O(nlogn) Time | O(1) Space
    """

    def insertSort(array, low=0, high=None):
        if high is None:
            high = len(array) - 1

        for i in range(low + 1, high + 1):
            j = i
            while j > 0 and array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]
                j -= 1

        return array

    if high is None:
        high = len(array) - 1

    while low < high and high - low > 16:
        q = partition(array, low, high)
        quickSortHoare(array, low, q)
        low = q + 1

    return insertSort(array, low, high)


def quickSortHeap(array, low=0, high=None, depth=None):
    """
    QuickSort using tail recursive + insertion sort + heap sort + hoare + median of three killer
    """

    def medianOf3(array, lowIdx, midIdx, highIdx):
        if (array[lowIdx] - array[midIdx]) * (array[highIdx] - array[lowIdx]) >= 0:
            return array[lowIdx]

        elif (array[midIdx] - array[lowIdx]) * (array[highIdx] - array[midIdx]) >= 0:
            return array[midIdx]

        else:
            return array[highIdx]

    def partition(array, low, high):
        pivot = medianOf3(array, low, (low + high) // 2, high)
        i = low - 1
        j = high + 1
        while True:
            i += 1
            while array[i] < pivot:
                i += 1
            j -= 1
            while array[j] > pivot:
                j -= 1

            if i >= j:
                return j

            array[i], array[j] = array[j], array[i]

    def insertSort(array, low=0, high=None):
        if high is None:
            high = len(array) - 1

        for i in range(low + 1, high + 1):
            j = i
            while j > 0 and array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]
                j -= 1

        return array

    if high is None:
        high = len(array) - 1
    if depth is None:
        depth = 2 * (len(array).bit_length() - 1)

    if depth == 0:
        return heapSort(array)
    else:
        while low < high and high - low > 16:
            q = partition(array, low, high)
            quickSortHoare(array, low, q)
            low = q + 1

    return insertSort(array, low, high)


def partition(array, low, high):
    pivot = array[(high + low) // 2]
    # pivot = array[randint(low, high)]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while array[i] < pivot:
            i += 1
        j -= 1
        while array[j] > pivot:
            j -= 1

        if i >= j:
            return j

        array[i], array[j] = array[j], array[i]


def gnomeSort(array):  # in-place | stable
    """
    Best : O(n) Time | O(1) Space
    Average : O(n^2) Time | O(1) Space
    Worst : O(n^2) Time | O(1) Space
    """
    index = 0
    while index < len(array):
        if index == 0:
            index = index + 1
        if array[index] >= array[index - 1]:
            index = index + 1
        else:
            array[index], array[index - 1] = array[index - 1], array[index]
            index = index - 1

    return array


def mergeSort(arr):  # stable
    """
    Best : O(nlogn) Time | O(n) Space
    Average : O(nlogn) Time | O(n) Space
    Worst : O(nlongn) Time | O(n) Space
    """
    if len(arr) < 2:
        return arr

    mid = len(arr) // 2
    low_arr = mergeSort(arr[:mid])
    high_arr = mergeSort(arr[mid:])

    merged_arr = []
    l = h = 0
    while l < len(low_arr) and h < len(high_arr):
        if low_arr[l] < high_arr[h]:
            merged_arr.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            h += 1
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    return merged_arr


def goSort(array):
    def insertSort(array, begin=0, end=None):
        if end == None:
            end = len(array)
        for i in range(begin, end):
            j = i
            toChange = array[i]
            while j != begin and array[j - 1] > toChange:
                array[j] = array[j - 1]
                j -= 1
            array[j] = toChange
        return array

    def siftDown(array, lo, hi, first):
        root = lo
        while True:
            child = 2 * root + 1
            if child >= hi:
                break

            if child + 1 < hi and array[first + child] < array[first + child + 1]:
                child += 1

            if array[first + root] >= array[first + child]:
                return

            array[first + root], array[first + child] = (
                array[first + child],
                array[first + root],
            )
            root = child

    def heapSort(array, a, b):
        first = a
        lo = 0
        hi = b - a

        # Build heap with greatest element at top.
        for i in range((hi - 1) // 2, -1, -1):
            siftDown(array, i, hi, first)

        # Pop elements, largest first, into end of data.
        for i in range(hi - 1, -1, -1):
            array[first], array[first + i] = array[first + i], array[first]
            siftDown(array, lo, i, first)

    def unsigned(n):
        return n & 0xFFFFFFFF

    def medianOfThree(array, m1, m0, m2):
        # sort 3 elements
        if array[m1] < array[m0]:
            array[m1], array[m0] = array[m0], array[m1]

        # data[m0] <= data[m1]
        if array[m2] < array[m1]:
            array[m2], array[m1] = array[m1], array[m2]

            # data[m0] <= data[m2] && data[m1] < data[m2]
            if array[m1] < array[m0]:
                array[m1], array[m0] = array[m0], array[m1]

        # now data[m0] <= data[m1] <= data[m2]

    def doPivot(array, lo, hi):
        m = int(unsigned(lo + hi) >> 1)

        if hi - lo > 40:
            # Tukey's ``Ninther,'' median of three medians of three.
            s = (hi - lo) // 8
            medianOfThree(array, lo, lo + s, lo + 2 * s)
            medianOfThree(array, m, m - s, m + s)
            medianOfThree(array, hi - 1, hi - 1 - s, hi - 1 - 2 * s)

        medianOfThree(array, lo, m, hi - 1)

        pivot = lo
        a, c = lo + 1, hi - 1

        while a < c and array[a] < array[pivot]:
            a += 1
        b = a
        while True:
            while b < c and array[pivot] >= array[b]:
                b += 1
            while b < c and array[pivot] < array[c - 1]:
                c -= 1
            if b >= c:
                break

            array[b], array[c - 1] = array[c - 1], array[b]
            b += 1
            c -= 1

        protect = hi - c < 5
        if not protect and hi - c < (hi - lo) // 4:
            dups = 0
            if array[pivot] >= array[hi - 1]:
                array[c], array[hi - 1] = array[hi - 1], array[c]
                c += 1
                dups += 1
            if array[b - 1] >= array[pivot]:
                b -= 1
                dups += 1

            if array[m] >= array[pivot]:
                array[m], array[b - 1] = array[b - 1], array[m]
                b -= 1
                dups += 1

            protect = dups > 1

        if protect:
            while True:
                while a < b and array[b - 1] >= array[pivot]:
                    b -= 1
                while a < b and array[a] < array[pivot]:
                    a += 1
                if a >= b:
                    break

                array[a], array[b - 1] = array[b - 1], array[a]
                a += 1
                b -= 1

        array[pivot], array[b - 1] = array[b - 1], array[pivot]
        return b - 1, c

    def quickSort(array, a, b, maxDepth):
        while b - a > 12:  # Use ShellSort for slices <= 12 elements
            if maxDepth == 0:
                return heapSort(array, a, b)

            maxDepth -= 1
            mlo, mhi = doPivot(array, a, b)
            # Avoiding recursion on the larger subproblem guarantees
            # a stack depth of at most lg(b-a).
            if mlo - a < b - mhi:
                quickSort(array, a, mlo, maxDepth)
                a = mhi  # i.e., quickSort(array, mhi, b)
            else:
                quickSort(array, mhi, b, maxDepth)
                b = mlo  # i.e., quickSort(array, a, mlo)

        if b - a > 1:
            # Do ShellSort pass with gap 6
            # It could be written in this simplified form cause b-a <= 12
            for i in range(a + 6, b):
                if array[i] < array[i - 6]:
                    array[i], array[i - 6] = array[i - 6], array[i]

            return insertSort(array, a, b)

    maxDepth = 2 * (len(array).bit_length() - 1)
    quickSort(array, 0, len(array), maxDepth)
    return array
