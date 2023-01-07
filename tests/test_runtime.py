import pytest

from bigO import algorithm


@pytest.fixture
def BigO():
    from bigO import BigO

    lib = BigO()
    return lib


def test_run(BigO):
    BigO.runtime(sorted, "random", 5000)
    BigO.runtime(algorithm.bubbleSort, "random", 5000)
    BigO.runtime(algorithm.countSort, "random", 5000)
    BigO.runtime(algorithm.binaryInsertSort, "random", 5000)
    BigO.runtime(algorithm.gnomeSort, "random", 5000)
    BigO.runtime(algorithm.heapSort, "random", 5000)
    BigO.runtime(algorithm.insertSort, "random", 5000)
    BigO.runtime(algorithm.insertSortOptimized, "random", 5000)
    BigO.runtime(algorithm.introSort, "random", 5000)
    BigO.runtime(algorithm.mergeSort, "random", 5000)
    BigO.runtime(algorithm.timSort, "random", 5000)
    BigO.runtime(algorithm.selectionSort, "random", 5000)


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


def test_custom(BigO):
    arr = ["dbc", "bbc", "ccd", "ef", "az"]
    time, result = BigO.runtime(brokenBubble, arr)

    print(time)
    print(result)


def test_str_array(BigO):
    time, result = BigO.runtime(algorithm.bubbleSort, "string", 10)
    print(result)

    time, result = BigO.runtime(algorithm.introSort, "string", 10)
    print(result)

    time, result = BigO.runtime(algorithm.quickSort, "string", 10)
    print(result)


def test_big(BigO):
    BigO.runtime(algorithm.bubbleSort, "random", 5000)
    BigO.runtime(algorithm.bubbleSort, "big", 5000)


def test_epoch(BigO):
    BigO.runtime(algorithm.quickSort, "random", 5000, 3, True)
    BigO.runtime(algorithm.quickSortHoare, "random", 5000, 3, True)


def test_heap(BigO):
    BigO.runtime(algorithm.heapSort2, "random", 500)
