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
        algorithm.quickSort, algorithm.quickSortHoare, "reversed", 50000
    )
    result = lib.compare(algorithm.quickSort, algorithm.quickSortHoare, "sorted", 50000)
    result = lib.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "partial", 50000
    )
    result = lib.compare(
        algorithm.quickSort, algorithm.quickSortHoare, "Ksorted", 50000
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
