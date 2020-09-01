from random import randrange


def test_gen():
    def genRandomArray(size):
        array = []
        for _ in range(size):
            array.append(randrange(-size, size))
        return array

    def genSortedArray(size):
        array = []
        for value in range(size):
            array.append(value)
        return array

    def genReversedArray(size):
        array = []
        for value in reversed(range(size)):
            array.append(value)
        return array

    def genPartialArray(size):
        array = genRandomArray(size)
        sorted_array = genSortedArray(size)

        array[size // 4 : size // 2] = sorted_array[size // 4 : size // 2]
        return array

    print(genReversedArray(100))
    print(genPartialArray(100))
    print(genKsortedArray(10, 4))


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
    aux = arr[0:n]
    aux.sort()

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


def genKsortedArray(size, k):
    def reverseRange(array, a, b):
        i = a
        j = b - 1
        while i < j:
            array[i], array[j] = array[j], array[i]
            i += 1
            j -= 1

        return array

    assert size >= k, "K must be smaller than the size."

    array = []
    for value in range(-size // 2, size // 2):
        array.append(value)

    right = randrange(k)
    while right > size - k - 1:  # Don't reverse again
        right -= 1

    if size != k:
        k += 1

    reverseRange(array, 0, k)
    reverseRange(array, len(array) - right, len(array))

    return array


def test_Ksorted():
    arr = genKsortedArray(9, 1)
    assert isKSortedArray(arr, 1) == True

    arr = genKsortedArray(100, 9)
    assert isKSortedArray(arr, 9) == True

    arr = genKsortedArray(9, 9)
    assert isKSortedArray(arr, 9) == True

    arr = genKsortedArray(10, 5)
    assert isKSortedArray(arr, 5) == True

    arr = genKsortedArray(1000, 100)
    assert isKSortedArray(arr, 100) == True
