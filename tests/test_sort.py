from bigO import bigO


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
