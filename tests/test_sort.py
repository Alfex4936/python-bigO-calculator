from bigO import algorithm
from bigO import bigO
import pytest


def empty(array):
    return array


def brokenBubble(array):
    isSorted = False
    counter = 0

    while not isSorted:
        isSorted = True
        for i in range(1, len(array) - 1 - counter):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False

        counter += 1

    return array


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

    _, _ = tester.test(countSort, "random")  # will return a warning


def test_Ksorted():
    tester = bigO()

    # Results may vary, O(n) possible
    complexity, _ = tester.test(algorithm.introSort, "Ksorted")
    complexity, _ = tester.test(algorithm.quickSort, "Ksorted")
    complexity, _ = tester.test(algorithm.timSort, "Ksorted")
    complexity, _ = tester.test(algorithm.countSort, "Ksorted")


def test_empty():
    big = bigO()
    cplx, _ = big.test(empty, "random")


@pytest.mark.timeout(600)
def test_bubble():
    tester = bigO()

    complexity, _ = tester.test(algorithm.bubbleSort, "random")
    complexity, _ = tester.test(algorithm.bubbleSort, "sorted")
    complexity, _ = tester.test(algorithm.bubbleSort, "reversed")
    complexity, _ = tester.test(algorithm.bubbleSort, "partial")
    complexity, _ = tester.test(algorithm.bubbleSort, "Ksorted")
    complexity, _ = tester.test(algorithm.bubbleSort, "almost_equal")
    complexity, _ = tester.test(algorithm.bubbleSort, "equal")
    complexity, _ = tester.test(algorithm.bubbleSort, "hole")


def test_brokenBubble():
    tester = bigO()
    _, result = tester.test(brokenBubble, "random")
    # will assert at index 0


def test_count():
    tester = bigO()

    # Results may vary
    complexity, _ = tester.test(algorithm.countSort, "random")
    assert complexity == "O(n)"
    complexity, _ = tester.test(algorithm.countSort, "sorted")
    assert complexity == "O(n)"
    complexity, _ = tester.test(algorithm.countSort, "reversed")
    assert complexity == "O(n)"
    complexity, _ = tester.test(algorithm.countSort, "partial")
    assert complexity == "O(n)"
    complexity, _ = tester.test(algorithm.countSort, "Ksorted")
    assert complexity == "O(n)"


@pytest.mark.timeout(600)
def test_insertion():
    tester = bigO()

    complexity, _ = tester.test(algorithm.insertSort, "random")
    complexity, _ = tester.test(algorithm.insertSort, "sorted")
    complexity, _ = tester.test(algorithm.insertSort, "reversed")
    complexity, _ = tester.test(algorithm.insertSort, "partial")
    complexity, _ = tester.test(algorithm.insertSort, "Ksorted")
    complexity, _ = tester.test(algorithm.insertSort, "string")


def test_intro():
    tester = bigO()

    # Results may vary, O(n) possible
    complexity, _ = tester.test(algorithm.introSort, "random")
    complexity, _ = tester.test(algorithm.introSort, "sorted")
    complexity, _ = tester.test(algorithm.introSort, "reversed")
    complexity, _ = tester.test(algorithm.introSort, "partial")
    complexity, _ = tester.test(algorithm.introSort, "Ksorted")
    # median of three won't work on string array


@pytest.mark.timeout(600)
def test_selection():
    tester = bigO()

    tester.test(algorithm.selectionSort, "random")
    tester.test(algorithm.selectionSort, "reversed")
    tester.test(algorithm.selectionSort, "sorted")
    tester.test(algorithm.selectionSort, "partial")
    tester.test(algorithm.selectionSort, "Ksorted")
    tester.test(algorithm.selectionSort, "string")


def test_timsort():
    tester = bigO()

    # Results may vary
    complexity, _ = tester.test(algorithm.timSort, "random")
    complexity, _ = tester.test(algorithm.timSort, "sorted")
    complexity, _ = tester.test(algorithm.timSort, "reversed")
    complexity, _ = tester.test(algorithm.timSort, "partial")
    complexity, _ = tester.test(algorithm.timSort, "Ksorted")
    complexity, _ = tester.test(algorithm.timSort, "hole")


def test_heap():
    tester = bigO()
    complexity, _ = tester.test(algorithm.heapSort2, "random")
    complexity, _ = tester.test(algorithm.heapSort2, "sorted")
    complexity, _ = tester.test(algorithm.heapSort2, "reversed")
    complexity, _ = tester.test(algorithm.heapSort2, "partial")
    complexity, _ = tester.test(algorithm.heapSort2, "Ksorted")


def test_quickSort():
    tester = bigO()

    complexity, time = tester.test(algorithm.quickSort, "random")
    complexity, time = tester.test(algorithm.quickSort, "sorted")
    complexity, time = tester.test(algorithm.quickSort, "reversed")
    complexity, time = tester.test(algorithm.quickSort, "partial")
    complexity, time = tester.test(algorithm.quickSort, "Ksorted")
    complexity, time = tester.test(algorithm.quickSort, "string")


def test_quickSort():
    tester = bigO()

    complexity, time = tester.test(algorithm.quickSortHoare, "random")
    complexity, time = tester.test(algorithm.quickSortHoare, "sorted")
    complexity, time = tester.test(algorithm.quickSortHoare, "reversed")
    complexity, time = tester.test(algorithm.quickSortHoare, "partial")
    complexity, time = tester.test(algorithm.quickSortHoare, "Ksorted")
    complexity, time = tester.test(algorithm.quickSortHoare, "hole")
    complexity, time = tester.test(algorithm.quickSortHoare, "equal")
    complexity, time = tester.test(algorithm.quickSortHoare, "almost_equal")
    complexity, time = tester.test(algorithm.quickSortHoare, "string")


def test_sort():
    lib = bigO()

    lib.test(sorted, "random")
    lib.test(sorted, "sorted")
    lib.test(sorted, "reversed")
    lib.test(sorted, "partial")
    lib.test(sorted, "Ksorted")
    lib.test(sorted, "string")
    lib.test(sorted, "hole")
    lib.test(sorted, "euqal")
    lib.test(sorted, "almost_equal")


def test_all_cases():
    lib = bigO()

    lib.test_all(sorted)
    lib.test_all(algorithm.bubbleSort)
    lib.test_all(algorithm.insertSort)
    lib.test_all(algorithm.selectionSort)
    lib.test_all(algorithm.doubleSelectionSort)
    lib.test_all(algorithm.timSort)
    lib.test_all(algorithm.heapSort2)
    lib.test_all(algorithm.quickSort)
