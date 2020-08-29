from bigO import bigO


def countSort(arr):  # stable
    # Time Complexity : O(n) | Space Complexity : O(n)
    minValue = min(arr)
    maxValue = max(arr) - minValue

    buckets = [0 for _ in range(maxValue + 1)]

    for i in arr:
        buckets[i - minValue] += 1

    index = 0
    for i in range(len(buckets)):
        while buckets[i] > 0:
            arr[index] = i + minValue
            index += 1
            buckets[i] -= 1

    return arr


def test_count():
    tester = bigO.bigO()

    complexity, _, res = tester.test(countSort, "random")
    assert complexity == "O(N)"
    complexity, _, res = tester.test(countSort, "sorted")
    assert complexity == "O(N)"
    complexity, _, res = tester.test(countSort, "reversed")
    assert complexity == "O(N)"
    complexity, _, res = tester.test(countSort, "partial")
    assert complexity == "O(N)"
