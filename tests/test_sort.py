import pytest

from bigO import algorithm


@pytest.fixture
def BigO():
    from bigO import BigO

    lib = BigO()
    return lib


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


def test_none(BigO):
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

    _, _ = BigO.test(countSort, "random")  # will return a warning


def test_Ksorted(BigO):
    # Results may vary, O(n) possible
    complexity = BigO.test(algorithm.introSort, "Ksorted")
    complexity = BigO.test(algorithm.quickSort, "Ksorted")
    complexity = BigO.test(algorithm.timSort, "Ksorted")
    complexity = BigO.test(algorithm.countSort, "Ksorted")


def test_empty(BigO):
    cplx = BigO.test(empty, "random")


@pytest.mark.timeout(600)
def test_bubble(BigO):
    complexity = BigO.test(algorithm.bubbleSort, "random")
    complexity = BigO.test(algorithm.bubbleSort, "sorted")
    complexity = BigO.test(algorithm.bubbleSort, "reversed")
    complexity = BigO.test(algorithm.bubbleSort, "partial")
    complexity = BigO.test(algorithm.bubbleSort, "Ksorted")
    complexity = BigO.test(algorithm.bubbleSort, "almost_equal")
    complexity = BigO.test(algorithm.bubbleSort, "equal")
    complexity = BigO.test(algorithm.bubbleSort, "hole")


def test_brokenBubble(BigO):
    _ = BigO.test(brokenBubble, "random")
    # will assert at index 0


def test_count(BigO):
    # Results may vary
    complexity = BigO.test(algorithm.countSort, "random")
    assert complexity == "O(n)"
    complexity = BigO.test(algorithm.countSort, "sorted")
    assert complexity == "O(n)"
    complexity = BigO.test(algorithm.countSort, "reversed")
    assert complexity == "O(n)"
    complexity = BigO.test(algorithm.countSort, "partial")
    assert complexity == "O(n)"
    complexity = BigO.test(algorithm.countSort, "Ksorted")
    assert complexity == "O(n)"


@pytest.mark.timeout(600)
def test_insertion(BigO):
    complexity = BigO.test(algorithm.insertSort, "random")
    complexity = BigO.test(algorithm.insertSort, "sorted")
    complexity = BigO.test(algorithm.insertSort, "reversed")
    complexity = BigO.test(algorithm.insertSort, "partial")
    complexity = BigO.test(algorithm.insertSort, "Ksorted")
    complexity = BigO.test(algorithm.insertSort, "string")


def test_intro(BigO):
    # Results may vary, O(n) possible
    complexity = BigO.test(algorithm.introSort, "random")
    complexity = BigO.test(algorithm.introSort, "sorted")
    complexity = BigO.test(algorithm.introSort, "reversed")
    complexity = BigO.test(algorithm.introSort, "partial")
    complexity = BigO.test(algorithm.introSort, "Ksorted")
    # median of three won't work on string array


@pytest.mark.timeout(600)
def test_selection(BigO):
    BigO.test(algorithm.selectionSort, "random")
    BigO.test(algorithm.selectionSort, "reversed")
    BigO.test(algorithm.selectionSort, "sorted")
    BigO.test(algorithm.selectionSort, "partial")
    BigO.test(algorithm.selectionSort, "Ksorted")
    BigO.test(algorithm.selectionSort, "string")


def test_timsort(BigO):
    # Results may vary
    complexity = BigO.test(algorithm.timSort, "random")
    complexity = BigO.test(algorithm.timSort, "sorted")
    complexity = BigO.test(algorithm.timSort, "reversed")
    complexity = BigO.test(algorithm.timSort, "partial")
    complexity = BigO.test(algorithm.timSort, "Ksorted")
    complexity = BigO.test(algorithm.timSort, "hole")


def test_heap(BigO):
    complexity = BigO.test(algorithm.heapSort2, "random")
    complexity = BigO.test(algorithm.heapSort2, "sorted")
    complexity = BigO.test(algorithm.heapSort2, "reversed")
    complexity = BigO.test(algorithm.heapSort2, "partial")
    complexity = BigO.test(algorithm.heapSort2, "Ksorted")


def test_quickSort(BigO):
    complexity = BigO.test(algorithm.quickSort, "random")
    complexity = BigO.test(algorithm.quickSort, "sorted")
    complexity = BigO.test(algorithm.quickSort, "reversed")
    complexity = BigO.test(algorithm.quickSort, "partial")
    complexity = BigO.test(algorithm.quickSort, "Ksorted")
    complexity = BigO.test(algorithm.quickSort, "string")


def test_quickSort(BigO):
    complexity = BigO.test(algorithm.quickSortHoare, "random")
    complexity = BigO.test(algorithm.quickSortHoare, "sorted")
    complexity = BigO.test(algorithm.quickSortHoare, "reversed")
    complexity = BigO.test(algorithm.quickSortHoare, "partial")
    complexity = BigO.test(algorithm.quickSortHoare, "Ksorted")
    complexity = BigO.test(algorithm.quickSortHoare, "hole")
    complexity = BigO.test(algorithm.quickSortHoare, "equal")
    complexity = BigO.test(algorithm.quickSortHoare, "almost_equal")
    complexity = BigO.test(algorithm.quickSortHoare, "string")


def test_sort(BigO):
    BigO.test(sorted, "random")
    BigO.test(sorted, "sorted")
    BigO.test(sorted, "reversed")
    BigO.test(sorted, "partial")
    BigO.test(sorted, "Ksorted")
    BigO.test(sorted, "string")
    BigO.test(sorted, "hole")
    BigO.test(sorted, "euqal")
    BigO.test(sorted, "almost_equal")


def test_all_cases(BigO):
    BigO.test_all(sorted)
    # BigO.test_all(algorithm.bubbleSort)
    # BigO.test_all(algorithm.insertSort)
    # BigO.test_all(algorithm.selectionSort)
    # BigO.test_all(algorithm.doubleSelectionSort)
    # BigO.test_all(algorithm.timSort)
    # BigO.test_all(algorithm.heapSort2)
    # BigO.test_all(algorithm.quickSort)
