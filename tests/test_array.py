from bigO import bigO


def test_gen():
    gen = bigO.bigO()

    print(gen.genRandomArray(20))
    print(gen.genRandomString(5, 20))
    print(gen.genSortedArray(20))
    print(gen.genReversedArray(20))
    print(gen.genPartialArray(20))
    print(gen.genKsortedArray(20, 6))

    for i in range(21):
        arr = gen.genKsortedArray(20, i)
        assert isKSortedArray(arr, i) == True

    print(gen.genAlmostEqualArray(9))
    print(gen.genAlmostEqualArray(20))
    print(gen.genEqualArray(20))
    print(gen.genHoleArray(20))


def test_Ksorted():
    gen = bigO.bigO()

    arr = gen.genKsortedArray(9, 1)
    assert isKSortedArray(arr, 1) == True

    arr = gen.genKsortedArray(100, 9)
    assert isKSortedArray(arr, 9) == True

    arr = gen.genKsortedArray(9, 9)
    assert isKSortedArray(arr, 9) == True

    arr = gen.genKsortedArray(10, 5)
    assert isKSortedArray(arr, 5) == True

    arr = gen.genKsortedArray(1000, 100)
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
