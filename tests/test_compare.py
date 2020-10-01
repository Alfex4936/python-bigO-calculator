from bigO import bigO
from bigO import algorithm


def test_run():
    lib = bigO.bigO()

    result = lib.compare(algorithm.bubbleSort, algorithm.insertSort, "reversed", 5000)
    result = lib.compare(
        algorithm.insertSort, algorithm.insertSortOptimized, "reversed", 5000
    )
    result = lib.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "reversed", 50000
    )
    result = lib.compare(algorithm.timSort, algorithm.introSort, "reversed", 50000)
    result = lib.compare(sorted, algorithm.introSort, "reversed", 50000)


def test_quickcomp():
    lib = bigO.bigO()

    # Hoare + Tail recur should be faster than random pivot choosing recursive one
    result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "random", 50000)
    result = lib.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "random", 50000
    )

    result = lib.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "reversed", 50000
    )
    result = lib.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "reversed", 50000
    )

    result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "sorted", 50000)
    result = lib.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "sorted", 50000
    )

    result = lib.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "partial", 50000
    )
    result = lib.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "partial", 50000
    )

    result = lib.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "Ksorted", 50000
    )
    result = lib.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "Ksorted", 50000
    )

    print(result)


def test_go():
    lib = bigO.bigO()

    lib.compare(algorithm.goSort, algorithm.introSort, "random", 100000)
    lib.compare(algorithm.goSort, algorithm.quickSortHoare, "random", 10000)

    lib.compare(algorithm.goSort, algorithm.heapSort, "random", 10000)
    lib.compare(algorithm.goSort, algorithm.timSort, "random", 10000)

    lib.compare(algorithm.goSort, algorithm.quickSortHoare, "Ksorted", 10000)
    lib.compare(algorithm.goSort, algorithm.quickSortHoare, "Ksorted", 10000)


def test_mini():
    lib = bigO.bigO()

    lib.compare(algorithm.insertSortOptimized, algorithm.insertSort, "random", 16)

    lib.compare(algorithm.bubbleSort, algorithm.insertSort, "random", 16)
    lib.compare(algorithm.insertSort, algorithm.selectionSort, "random", 16)
    lib.compare(algorithm.bubbleSort, algorithm.selectionSort, "random", 16)

    lib.compare(algorithm.bubbleSort, algorithm.insertSort, "reversed", 16)
    lib.compare(algorithm.insertSort, algorithm.selectionSort, "reversed", 16)
    lib.compare(algorithm.bubbleSort, algorithm.selectionSort, "reversed", 16)


def test_all():
    lib = bigO.bigO()
    result = lib.compare(
        algorithm.quickSortHoare, algorithm.quickSortHeap, "all", 50000
    )

    result = lib.compare(algorithm.insertSort, algorithm.bubbleSort, "all", 5000)
    print(result)

    result = lib.compare(algorithm.quickSortHoare, algorithm.insertSort, "all", 5000)

