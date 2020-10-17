from bigO import bigO
from bigO import algorithm


def test_run():
    lib = bigO.bigO()

    lib.runtime(sorted, "random", 5000)
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


def test_custom():
    lib = bigO.bigO()

    arr = ["dbc", "bbc", "ccd", "ef", "az"]
    time, result = lib.runtime(brokenBubble, arr)

    print(time)
    print(result)


def test_str_array():
    lib = bigO.bigO()

    time, result = lib.runtime(algorithm.bubbleSort, "string", 10)
    print(result)

    time, result = lib.runtime(algorithm.introSort, "string", 10)
    print(result)

    time, result = lib.runtime(algorithm.quickSort, "string", 10)
    print(result)


def test_big():
    lib = bigO.bigO()
    lib.runtime(algorithm.bubbleSort, "random", 5000)
    lib.runtime(algorithm.bubbleSort, "big", 5000)


def test_epoch():
    lib = bigO.bigO()
    lib.runtime(algorithm.quickSort, "random", 5000, 3, True)
    lib.runtime(algorithm.quickSortHoare, "random", 5000, 3, True)


def test_heap():
    lib = bigO.bigO()
    lib.runtime(algorithm.heapSortMax, "random", 500)