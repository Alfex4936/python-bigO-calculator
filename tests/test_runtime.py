from bigO import bigO
from bigO import algorithm


def test_run():
    lib = bigO.bigO()

    lib.runtime(algorithm.bubbleSort, "random", 5000)
    lib.runtime(algorithm.countSort, "random", 5000)
    lib.runtime(algorithm.binaryInsertSort, "random", 5000)
    lib.runtime(algorithm.gnomeSort, "random", 5000)
    lib.runtime(algorithm.heapSort, "random", 5000)
    lib.runtime(algorithm.insertSort, "random", 5000)
    lib.runtime(algorithm.insertSortOptimized, "random", 5000)
    lib.runtime(algorithm.introSort, "random", 5000)
    lib.runtime(algorithm.mergeSort, "random", 5000)
    lib.runtime(algorithm.timSort, "random", 5000)
    lib.runtime(algorithm.selectionSort, "random", 5000)


def test_custom():
    lib = bigO.bigO()

    arr = [0, 1, 2, -1, 1]
    time, result = lib.runtime(algorithm.bubbleSort, arr)

    print(time)
    print(result)

    assert result == sorted(arr)
