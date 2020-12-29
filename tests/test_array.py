import pytest


@pytest.fixture
def BigO():
    from bigO import BigO

    lib = BigO()
    return lib


def test_BigO(BigO):
    print(BigO.genRandomArray(20))
    print(BigO.genRandomString(5, 20))
    print(BigO.genSortedArray(20))
    print(BigO.genReversedArray(20))
    print(BigO.genPartialArray(20))
    print(BigO.genKsortedArray(20, 6))

    for i in range(21):
        arr = BigO.genKsortedArray(20, i)
        assert isKSortedArray(arr, i) == True

    print(BigO.genAlmostEqualArray(9))
    print(BigO.genAlmostEqualArray(20))
    print(BigO.genEqualArray(20))
    print(BigO.genHoleArray(20))
    print(BigO.genRandomBigArray(20))


def test_Ksorted(BigO):
    arr = BigO.genKsortedArray(9, 1)
    assert isKSortedArray(arr, 1) == True

    arr = BigO.genKsortedArray(100, 9)
    assert isKSortedArray(arr, 9) == True

    arr = BigO.genKsortedArray(9, 9)
    assert isKSortedArray(arr, 9) == True

    arr = BigO.genKsortedArray(10, 5)
    assert isKSortedArray(arr, 5) == True

    arr = BigO.genKsortedArray(1000, 100)
    assert isKSortedArray(arr, 100) == True


def binarySearch(arr, low, high, x):
    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            high = mid - 1
        else:
            low = mid + 1


# function to check whether the given
# array is a 'k' sorted array or not
def isKSortedArray(arr, k):
    n = len(arr)
    aux = sorted(arr)

    # for every element of 'arr' at
    # index 'i', find its index 'j' in 'aux'
    for i in range(0, n):

        # index of arr[i] in sorted
        # array 'aux'
        j = binarySearch(aux, 0, n - 1, arr[i])

        # if abs(i-j) > k, then that element is
        # not at-most k distance away from its
        # target position. Thus, 'arr' is not a
        # k sorted array
        if abs(i - j) > k:
            return False

    # 'arr' is a k sorted array
    return True
