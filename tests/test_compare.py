import pytest
from bigO import algorithm


@pytest.fixture
def BigO():
    from bigO import BigO

    lib = BigO()
    return lib


def test_run(BigO):
    result = BigO.compare(algorithm.bubbleSort, algorithm.insertSort, "reversed", 5000)
    result = BigO.compare(
        algorithm.insertSort, algorithm.insertSortOptimized, "reversed", 5000
    )
    result = BigO.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "reversed", 50000
    )
    result = BigO.compare(algorithm.timSort, algorithm.introSort, "reversed", 50000)
    result = BigO.compare(sorted, algorithm.introSort, "reversed", 50000)

    result = BigO.compare(algorithm.heapSort, algorithm.heapSort2, "all", 50000)
    result = BigO.compare(algorithm.introSort, algorithm.quickSortHeap, "all", 50000)


def test_quickcomp(BigO):
    # Hoare + Tail recur should be faster than random pivot choosing recursive one
    result = BigO.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "random", 50000
    )
    result = BigO.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "random", 50000
    )

    result = BigO.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "reversed", 50000
    )
    result = BigO.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "reversed", 50000
    )

    result = BigO.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "sorted", 50000
    )
    result = BigO.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "sorted", 50000
    )

    result = BigO.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "partial", 50000
    )
    result = BigO.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "partial", 50000
    )

    result = BigO.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "Ksorted", 50000
    )
    result = BigO.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "Ksorted", 50000
    )

    print(result)


def test_go(BigO):
    BigO.compare(algorithm.goSort, algorithm.introSort, "random", 100000)
    BigO.compare(algorithm.goSort, algorithm.quickSortHoare, "random", 10000)

    BigO.compare(algorithm.goSort, algorithm.heapSort, "random", 10000)
    BigO.compare(algorithm.goSort, algorithm.timSort, "random", 10000)

    BigO.compare(algorithm.goSort, algorithm.quickSortHoare, "Ksorted", 10000)
    BigO.compare(algorithm.goSort, algorithm.quickSortHoare, "Ksorted", 10000)


def test_mini(BigO):
    BigO.compare(algorithm.insertSortOptimized, algorithm.insertSort, "random", 16)

    BigO.compare(algorithm.bubbleSort, algorithm.insertSort, "random", 16)
    BigO.compare(algorithm.insertSort, algorithm.selectionSort, "random", 16)
    BigO.compare(algorithm.bubbleSort, algorithm.selectionSort, "random", 16)

    BigO.compare(algorithm.bubbleSort, algorithm.insertSort, "reversed", 16)
    BigO.compare(algorithm.insertSort, algorithm.selectionSort, "reversed", 16)
    BigO.compare(algorithm.bubbleSort, algorithm.selectionSort, "reversed", 16)


def test_all(BigO):
    result = BigO.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "all", 50000
    )

    result = BigO.compare(algorithm.insertSort, algorithm.bubbleSort, "all", 5000)
    print(result)

    result = BigO.compare(algorithm.quickSortHoare, algorithm.insertSort, "all", 5000)


def test_custom(BigO):
    BigO.compare(algorithm.doubleSelectionSort, algorithm.selectionSort, "all", 5000)
