from bigO import bigO
from random import randint


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
        toChange = array[i]
        while j != begin and array[j - 1] > toChange:
            array[j] = array[j - 1]
            j -= 1
        array[j] = toChange
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


def test_none():
    tester = bigO.bigO()

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

    _, _, _ = tester.test(countSort, "random")  # will return a warning


def test_count():
    tester = bigO.bigO()

    # Results may vary
    complexity, _, _ = tester.test(countSort, "random")
    assert complexity == "O(N)"
    complexity, _, _ = tester.test(countSort, "sorted")
    assert complexity == "O(N)"
    complexity, _, _ = tester.test(countSort, "reversed")
    assert complexity == "O(N)"
    complexity, _, _ = tester.test(countSort, "partial")
    assert complexity == "O(N)"


def test_intro():
    tester = bigO.bigO()

    # Results may vary, O(n) possible
    complexity, _, _ = tester.test(introSort, "random")
    assert complexity == "O(Nlg(N))"
    complexity, _, _ = tester.test(introSort, "sorted")
    assert complexity == "O(Nlg(N))"
    complexity, _, _ = tester.test(introSort, "reversed")
    assert complexity == "O(Nlg(N))"
    complexity, _, _ = tester.test(introSort, "partial")
    assert complexity == "O(Nlg(N))"


def test_introKsorted():
    tester = bigO.bigO()

    # Results may vary, O(n) possible
    complexity, _, _ = tester.test(introSort, "Ksorted")
    assert complexity == "O(Nlg(N))"


def test_timsort():
    tester = bigO.bigO()

    # Results may vary
    complexity, _, _ = tester.test(timSort, "random")
    complexity, _, _ = tester.test(timSort, "sorted")
    complexity, _, _ = tester.test(timSort, "reversed")
    complexity, _, _ = tester.test(timSort, "partial")
    complexity, _, _ = tester.test(timSort, "Ksorted")


def test_quickSort():
    tester = bigO.bigO()

    complexity, _, _ = tester.test(quickSort, "random")
    complexity, _, _ = tester.test(quickSort, "sorted")
    complexity, _, _ = tester.test(quickSort, "reversed")
    complexity, _, _ = tester.test(quickSort, "partial")
    complexity, _, _ = tester.test(quickSort, "Ksorted")
