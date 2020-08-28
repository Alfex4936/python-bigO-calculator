from bigO import bigO


def countSort(arr):  # stable
    # Time Complexity : O(n) | Space Complexity : O(n)
    sortedArr = arr[:]
    minValue = min(arr)
    maxValue = max(arr) - minValue

    buckets = [0 for x in range(maxValue + 1)]

    for i in sortedArr:
        buckets[i - minValue] += 1

    index = 0
    for i in range(len(buckets)):
        while buckets[i] > 0:
            sortedArr[index] = i + minValue
            index += 1
            buckets[i] -= 1

    return sortedArr


tester = bigO.bigO()
complexity, _, res = tester.test(countSort, "random")
complexity, _, res = tester.test(countSort, "sorted")
complexity, _, res = tester.test(countSort, "reversed")
complexity, _, res = tester.test(countSort, "partial")

