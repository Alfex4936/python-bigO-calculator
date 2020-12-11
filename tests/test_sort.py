from bigO import algorithm
from bigO import BigO
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
    tester = BigO()

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
    tester = BigO()

    # Results may vary, O(n) possible
    complexity = tester.test(algorithm.introSort, "Ksorted")
    complexity = tester.test(algorithm.quickSort, "Ksorted")
    complexity = tester.test(algorithm.timSort, "Ksorted")
    complexity = tester.test(algorithm.countSort, "Ksorted")


def test_empty():
    big = BigO()
    cplx = big.test(empty, "random")


@pytest.mark.timeout(600)
def test_bubble():
    tester = BigO()

    complexity = tester.test(algorithm.bubbleSort, "random")
    # complexity = tester.test(algorithm.bubbleSort, "sorted")
    # complexity = tester.test(algorithm.bubbleSort, "reversed")
    # complexity = tester.test(algorithm.bubbleSort, "partial")
    # complexity = tester.test(algorithm.bubbleSort, "Ksorted")
    # complexity = tester.test(algorithm.bubbleSort, "almost_equal")
    # complexity = tester.test(algorithm.bubbleSort, "equal")
    # complexity = tester.test(algorithm.bubbleSort, "hole")


def test_brokenBubble():
    tester = BigO()
    _ = tester.test(brokenBubble, "random")
    # will assert at index 0


def test_count():
    tester = BigO()

    # Results may vary
    complexity = tester.test(algorithm.countSort, "random")
    assert complexity == "O(n)"
    complexity = tester.test(algorithm.countSort, "sorted")
    assert complexity == "O(n)"
    complexity = tester.test(algorithm.countSort, "reversed")
    assert complexity == "O(n)"
    complexity = tester.test(algorithm.countSort, "partial")
    assert complexity == "O(n)"
    complexity = tester.test(algorithm.countSort, "Ksorted")
    assert complexity == "O(n)"


@pytest.mark.timeout(600)
def test_insertion():
    tester = BigO()

    complexity = tester.test(algorithm.insertSort, "random")
    complexity = tester.test(algorithm.insertSort, "sorted")
    complexity = tester.test(algorithm.insertSort, "reversed")
    complexity = tester.test(algorithm.insertSort, "partial")
    complexity = tester.test(algorithm.insertSort, "Ksorted")
    complexity = tester.test(algorithm.insertSort, "string")


def test_intro():
    tester = BigO()

    # Results may vary, O(n) possible
    complexity = tester.test(algorithm.introSort, "random")
    complexity = tester.test(algorithm.introSort, "sorted")
    complexity = tester.test(algorithm.introSort, "reversed")
    complexity = tester.test(algorithm.introSort, "partial")
    complexity = tester.test(algorithm.introSort, "Ksorted")
    # median of three won't work on string array


@pytest.mark.timeout(600)
def test_selection():
    tester = BigO()

    tester.test(algorithm.selectionSort, "random")
    tester.test(algorithm.selectionSort, "reversed")
    tester.test(algorithm.selectionSort, "sorted")
    tester.test(algorithm.selectionSort, "partial")
    tester.test(algorithm.selectionSort, "Ksorted")
    tester.test(algorithm.selectionSort, "string")


def test_timsort():
    tester = BigO()

    # Results may vary
    complexity = tester.test(algorithm.timSort, "random")
    complexity = tester.test(algorithm.timSort, "sorted")
    complexity = tester.test(algorithm.timSort, "reversed")
    complexity = tester.test(algorithm.timSort, "partial")
    complexity = tester.test(algorithm.timSort, "Ksorted")
    complexity = tester.test(algorithm.timSort, "hole")


def test_heap():
    tester = BigO()
    complexity = tester.test(algorithm.heapSort2, "random")
    complexity = tester.test(algorithm.heapSort2, "sorted")
    complexity = tester.test(algorithm.heapSort2, "reversed")
    complexity = tester.test(algorithm.heapSort2, "partial")
    complexity = tester.test(algorithm.heapSort2, "Ksorted")


def test_quickSort():
    tester = BigO()

    complexity = tester.test(algorithm.quickSort, "random")
    complexity = tester.test(algorithm.quickSort, "sorted")
    complexity = tester.test(algorithm.quickSort, "reversed")
    complexity = tester.test(algorithm.quickSort, "partial")
    complexity = tester.test(algorithm.quickSort, "Ksorted")
    complexity = tester.test(algorithm.quickSort, "string")


def test_quickSort():
    tester = BigO()

    complexity = tester.test(algorithm.quickSortHoare, "random")
    complexity = tester.test(algorithm.quickSortHoare, "sorted")
    complexity = tester.test(algorithm.quickSortHoare, "reversed")
    complexity = tester.test(algorithm.quickSortHoare, "partial")
    complexity = tester.test(algorithm.quickSortHoare, "Ksorted")
    complexity = tester.test(algorithm.quickSortHoare, "hole")
    complexity = tester.test(algorithm.quickSortHoare, "equal")
    complexity = tester.test(algorithm.quickSortHoare, "almost_equal")
    complexity = tester.test(algorithm.quickSortHoare, "string")


def test_sort():
    lib = BigO()

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
    lib = BigO()

    lib.test_all(sorted)
    lib.test_all(algorithm.bubbleSort)
    lib.test_all(algorithm.insertSort)
    lib.test_all(algorithm.selectionSort)
    lib.test_all(algorithm.doubleSelectionSort)
    lib.test_all(algorithm.timSort)
    lib.test_all(algorithm.heapSort2)
    lib.test_all(algorithm.quickSort)
