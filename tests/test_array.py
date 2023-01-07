import pytest


@pytest.fixture
def BigO():
    from bigO import BigO

    lib = BigO()
    return lib


def test_BigO(BigO):
    print(BigO.gen_random_positive_ints(20))
    print(BigO.gen_random_strings(5, 20))
    print(BigO.gen_sorted_ints(20))
    print(BigO.gen_reversed_ints(20))
    print(BigO.gen_partial_ints(20))
    print(BigO.gen_ksorted_ints(20, 6))

    for i in range(21):
        arr = BigO.gen_ksorted_ints(20, i)
        assert isKSortedArray(arr, i) == True

    print(BigO.gen_almost_equal_ints(9))
    print(BigO.gen_almost_equal_ints(20))
    print(BigO.gen_equal_ints(20))
    print(BigO.gen_hole_ints(20))
    print(BigO.gen_random_big_ints(20))


def test_Ksorted(BigO):
    arr = BigO.gen_ksorted_ints(9, 1)
    assert isKSortedArray(arr, 1) == True

    arr = BigO.gen_ksorted_ints(100, 9)
    assert isKSortedArray(arr, 9) == True

    arr = BigO.gen_ksorted_ints(9, 9)
    assert isKSortedArray(arr, 9) == True

    arr = BigO.gen_ksorted_ints(10, 5)
    assert isKSortedArray(arr, 5) == True

    arr = BigO.gen_ksorted_ints(1000, 100)
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
