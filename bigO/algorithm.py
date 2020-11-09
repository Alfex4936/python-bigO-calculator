import bisect
from random import randrange


def bubbleSort(array: list):
    isSorted: bool = False
    counter: int = 0
    n: int = len(array) - 1

    while not isSorted:
        isSorted = True
        for i in range(n - counter):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False

        counter += 1

    return array


def brickSort(array: list):
    """ Odd-Even Sort"""
    isSorted: bool = False
    n: int = len(array) - 1

    while not isSorted:
        isSorted = True
        for i in range(1, n, 2):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False

        for i in range(0, n, 2):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False

    return array


def binarySearch(array: list, item, start, end):
    if start == end:
        if array[start] > item:
            return start
        else:
            return start + 1
    if start > end:
        return start

    mid = (start + end) // 2

    if array[mid] < item:
        return binarySearch(array, item, mid + 1, end)
    elif array[mid] > item:
        return binarySearch(array, item, start, mid - 1)
    else:
        return mid


def binaryInsertSort(array: list):
    for index in range(1, len(array)):
        value = array[index]
        pos = binarySearch(array, value, 0, index - 1)
        array = array[:pos] + [value] + array[pos:index] + array[index + 1 :]
    return array


def countSort(arr: list):  # stable
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


def combSort(array: list):
    gap = n = len(array)
    swapped = True

    while gap > 1 or swapped:
        swapped = False

        gap = int(gap * 10 / 13)

        if gap < 1:
            swapped = False
            gap = 1

        for i in range(n - gap):
            if array[i] > array[i + gap]:
                array[i], array[i + gap] = array[i + gap], array[i]
                swapped = True

    return array


def selectionSort(array: list):
    """
    Best : O(n^2) Time | O(1) Space
    Average : O(n^2) Time | O(1) Space
    Worst : O(n^2) Time | O(1) Space
    """
    n: int = len(array)
    for currentIdx in range(n - 1):
        smallestIdx = currentIdx
        for i in range(currentIdx + 1, n):
            if array[smallestIdx] > array[i]:
                smallestIdx = i
        array[currentIdx], array[smallestIdx] = array[smallestIdx], array[currentIdx]

    return array


def introSort(array: list):  # in-place | not-stable
    # Time Complexity O(nlogn) | Space Complexity O(logn)
    maxDepth = 2 * (len(array).bit_length() - 1)
    sizeThreshold = 16
    return introSortHelper(array, 0, len(array), sizeThreshold, maxDepth)


def introSortHelper(array, start, end, sizeThreshold, depthLimit):
    def medianOf3(array, lowIdx, midIdx, highIdx):
        if (array[lowIdx] - array[midIdx]) * (array[highIdx] - array[lowIdx]) >= 0:
            return array[lowIdx]

        elif (array[midIdx] - array[lowIdx]) * (array[highIdx] - array[midIdx]) >= 0:
            return array[midIdx]

        else:
            return array[highIdx]

    def getPartition(array, low, high, pivot):
        i = low
        j = high
        while True:

            while array[i] < pivot:
                i += 1
            j -= 1
            while pivot < array[j]:
                j -= 1
            if i >= j:
                return i
            array[i], array[j] = array[j], array[i]
            i += 1

    while end - start > sizeThreshold:
        if depthLimit == 0:
            return heapSort2(array)
        depthLimit -= 1

        median = medianOf3(array, start, start + ((end - start) // 2) + 1, end - 1)
        p = getPartition(array, start, end, median)
        introSortHelper(array, p, end, sizeThreshold, depthLimit)
        end = p

    return insertSort(array, start, end)


def insertSort(array, begin=0, end=None):  # in-place | stable
    """
    Best O(n) Time | O(1) Space
    Average O(n^2) Time | O(1) Space
    Worst (On^2) Time | O(1) Space
    """
    if end == None:
        end = len(array)

    for i in range(begin, end):
        j = i
        while j > begin and array[j - 1] > array[j]:
            array[j], array[j - 1] = array[j - 1], array[j]
            j -= 1

    return array


def insertSortOptimized(array):
    for i in range(1, len(array)):
        if array[i] >= array[i - 1]:
            continue
        for j in range(i):
            if array[i] < array[j]:
                array[j], array[j + 1 : i + 1] = array[i], array[j:i]
                break

    return array


def heapSort(arr):  # in-place | not-stable
    # Time Complexity O(nlogn) | Space Complexity O(1)

    def heapify(arr, n, i):  # Max Heap
        largest = i  # 트리에서 가장 큰 값 찾기
        l = 2 * i + 1  # Left Node
        r = 2 * i + 2  # Right Node

        if l < n and arr[largest] < arr[l]:
            largest = l

        if r < n and arr[largest] < arr[r]:
            largest = r

        # root가 최대가 아니면
        # 최대 값과 바꾸고, 계속 heapify
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)

    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        # Heapify root element
        heapify(arr, i, 0)

    return arr


def heapSort2(iterable):  # in-place | not-stable
    # Time Complexity O(nlogn)
    # Using Python source at
    # https://github.com/python/cpython/blob/975d10a4f8f5d99b01d02fc5f99305a86827f28e/Lib/heapq.py

    def heappop(heap):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            _siftup(heap, 0)
            return returnitem
        return lastelt

    def _siftup(heap, pos):
        endpos = len(heap)
        startpos = pos
        newitem = heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not heap[childpos] < heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            heap[pos] = heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap[pos] = newitem
        _siftdown(heap, startpos, pos)

    def _siftdown(heap, startpos, pos):
        newitem = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem < parent:
                heap[pos] = parent
                pos = parentpos
                continue
            break
        heap[pos] = newitem

    def heapify(x):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(x)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(n // 2)):
            _siftup(x, i)

        return x

    h = []
    # for value in iterable:
    #     heappush(h, value)
    h = heapify(iterable)

    return [heappop(h) for _ in range(len(h))]


def timSort(lst):
    """
    Python sort(), sorted() implementation
    from https://github.com/hu-ng/timsort/blob/master/timsort.py
    
    Best : O(n) Time | O(n) Space
    Average : O(nlogn) Time | O(n) Space
    Worst : O(nlogn) Time | O(n) Space
    """

    def reverse(lst, s, e):
        """Reverse the order of a list in place
        Input: s = starting index, e = ending index"""
        while s < e and s != e:
            lst[s], lst[e] = lst[e], lst[s]
            s += 1
            e -= 1

    def make_temp_array(lst, s, e):
        """From the lst given, make a copy from index s to index e"""
        array = []
        while s <= e:
            array.append(lst[s])
            s += 1
        return array

    def merge_compute_minrun(n):
        """Returns the minimum length of a run from 23 - 64 so that
        the len(array)/minrun is less than or equal to a power of 2."""
        r = 0
        while n >= 32:
            r |= n & 1
            n >>= 1
        return n + r

    def count_run(lst, s_run):
        """Count the length of one run, returns starting/ending indices,
        a boolean value to present increasing/decreasing run,
        and the length of the run"""
        increasing = True

        # If count_run started at the final position of the array
        if s_run == len(lst) - 1:
            return [s_run, s_run, increasing, 1]
        else:
            e_run = s_run
            # Decreasing run (strictly decreasing):
            if lst[s_run] > lst[s_run + 1]:
                while lst[e_run] > lst[e_run + 1]:
                    e_run += 1
                    if e_run == len(lst) - 1:
                        break
                increasing = False
                return [s_run, e_run, increasing, e_run - s_run + 1]

            # Increasing run (non-decreasing):
            else:
                while lst[e_run] <= lst[e_run + 1]:
                    e_run += 1
                    if e_run == len(lst) - 1:
                        break
                return [s_run, e_run, increasing, e_run - s_run + 1]

    def bin_sort(lst, s, e, extend):
        """Binary insertion sort, assumed that lst[s:e + 1] is sorted.
        Extend the run by the number indicated by 'extend'"""

        for i in range(1, extend + 1):
            pos = 0
            start = s
            end = e + i

            # Value to be inserted
            value = lst[end]

            # If the value is already bigger than the last element from start -> end:
            # Don't do the following steps
            if value >= lst[end - 1]:
                continue

            # While-loop does the binary search
            while start <= end:
                if start == end:
                    if lst[start] > value:
                        pos = start
                        break
                    else:
                        pos = start + 1
                        break
                mid = (start + end) // 2
                if value >= lst[mid]:
                    start = mid + 1
                else:
                    end = mid - 1

            if start > end:
                pos = start

            # 'Push' the elements to the right by 1 element
            # Copy the value back the right position.
            for x in range(e + i, pos, -1):
                lst[x] = lst[x - 1]
            lst[pos] = value

    def gallop(lst, val, low, high, ltr):
        """Find the index of val in the slice[low:high]"""

        if ltr == True:
            # Used for merging from left to right
            # The index found will be so that every element prior
            # to that index is strictly smaller than val
            pos = bisect.bisect_left(lst, val, low, high)
            return pos

        else:
            # Used for merging from right to left
            # The index found will be so that every element from
            # that index onwards is strictly larger than val
            pos = bisect.bisect_right(lst, val, low, high)
            return pos

    def merge(lst, stack, run_num):
        """Merge the two runs and update the remaining runs in the stack
        Only consequent runs are merged, one lower, one upper."""

        # Make references to the to-be-merged runs
        run_a = stack[run_num]
        run_b = stack[run_num + 1]

        # Make a reference to where the new combined run would be.
        new_run = [run_a[0], run_b[1], True, run_b[1] - run_a[0] + 1]

        # Put this new reference in the correct position in the stack
        stack[run_num] = new_run

        # Delete the upper run of the two runs from the stack
        del stack[run_num + 1]

        # If the length of run_a is smaller than or equal to length of run_b
        if run_a[3] <= run_b[3]:
            merge_low(lst, run_a, run_b, 7)

        # If the length of run_a is bigger than length of run_b
        else:
            merge_high(lst, run_a, run_b, 7)

    def merge_low(lst, a, b, min_gallop):
        """Merges the two runs quasi in-place if a is the smaller run
        - a and b are lists that store data of runs
        - min_gallop: threshold needed to switch to galloping mode
        - galloping mode: uses gallop() to 'skip' elements instead of linear merge"""

        # Make a copy of the run a, the smaller run
        temp_array = make_temp_array(lst, a[0], a[1])
        # The first index of the merging area
        k = a[0]
        # Counter for the temp array of a
        i = 0
        # Counter for b, starts at the beginning
        j = b[0]

        gallop_thresh = min_gallop
        while True:
            a_count = 0  # number of times a win in a row
            b_count = 0  # number of times b win in a row

            # Linear merge mode, taking note of how many times a and b wins in a row.
            # If a_count or b_count > threshold, switch to gallop
            while i <= len(temp_array) - 1 and j <= b[1]:

                # if elem in a is smaller, a wins
                if temp_array[i] <= lst[j]:
                    lst[k] = temp_array[i]
                    k += 1
                    i += 1

                    a_count += 1
                    b_count = 0

                    # If a runs out during linear merge
                    # Copy the rest of b
                    if i > len(temp_array) - 1:
                        while j <= b[1]:
                            lst[k] = lst[j]
                            k += 1
                            j += 1
                        return

                    # threshold reached, switch to gallop
                    if a_count >= gallop_thresh:
                        break

                # if elem in b is smaller, b wins
                else:
                    lst[k] = lst[j]
                    k += 1
                    j += 1

                    a_count = 0
                    b_count += 1

                    # If b runs out during linear merge
                    # copy the rest of a
                    if j > b[1]:
                        while i <= len(temp_array) - 1:
                            lst[k] = temp_array[i]
                            k += 1
                            i += 1
                        return

                    # threshold reached, switch to gallop
                    if b_count >= gallop_thresh:
                        break

            # If one run is winning consistently, switch to galloping mode.
            # i, j, and k are incremented accordingly
            while True:
                # Look for the position of b[j] in a
                # bisect_left() -> a_adv = index in the slice [i: len(temp_array)]
                # so that every elem before temp_array[a_adv] is strictly smaller than lst[j]
                a_adv = gallop(temp_array, lst[j], i, len(temp_array), True)

                # Copy the elements prior to a_adv to the merge area, increment k
                for x in range(i, a_adv):
                    lst[k] = temp_array[x]
                    k += 1

                # Update the a_count to check successfulness of galloping
                a_count = a_adv - i

                # Advance i to a_adv
                i = a_adv

                # If run a runs out
                if i > len(temp_array) - 1:
                    # Copy all of b over, if there is any left
                    while j <= b[1]:
                        lst[k] = lst[j]
                        k += 1
                        j += 1
                    return

                # Copy b[j] over
                lst[k] = lst[j]
                k += 1
                j += 1

                # If b runs out
                if j > b[1]:
                    # Copy all of a over, if there is any left
                    while i < len(temp_array):
                        lst[k] = temp_array[i]
                        k += 1
                        i += 1
                    return

                # ------------------------------------------------------

                # Look for the position of a[i] in b
                # b_adv is analogous to a_adv
                b_adv = gallop(lst, temp_array[i], j, b[1] + 1, True)
                for y in range(j, b_adv):
                    lst[k] = lst[y]
                    k += 1

                # Update the counters and check the conditions
                b_count = b_adv - j
                j = b_adv

                # If b runs out
                if j > b[1]:
                    # copy the rest of a over
                    while i <= len(temp_array) - 1:
                        lst[k] = temp_array[i]
                        k += 1
                        i += 1
                    return

                # copy a[i] over to the merge area
                lst[k] = temp_array[i]
                i += 1
                k += 1

                # If a runs out
                if i > len(temp_array) - 1:
                    # copy the rest of b over
                    while j <= b[1]:
                        lst[k] = lst[j]
                        k += 1
                        j += 1
                    return

                # if galloping proves to be unsuccessful, return to linear
                if a_count < gallop_thresh and b_count < gallop_thresh:
                    break

            # punishment for leaving galloping
            # makes it harder to enter galloping next time
            gallop_thresh += 1

    def merge_high(lst, a, b, min_gallop):
        """Merges the two runs quasi in-place if b is the smaller run
        - Analogous to merge_low, but starts from the end
        - a and b are lists that store data of runs
        - min_gallop: threshold needed to switch to galloping mode
        - galloping mode: uses gallop() to 'skip' elements instead of linear merge"""

        # Make a copy of b, the smaller run
        temp_array = make_temp_array(lst, b[0], b[1])

        # Counter for the merge area, starts at the last index of array b
        k = b[1]
        # Counter for the temp array

        i = len(temp_array) - 1  # Lower bound is 0

        # Counter for a, starts at the end this time
        j = a[1]

        gallop_thresh = min_gallop
        while True:
            a_count = 0  # number of times a win in a row
            b_count = 0  # number of times b win in a row

            # Linear merge, taking note of how many times a and b wins in a row.
            # If a_count or b_count > threshold, switch to gallop
            while i >= 0 and j >= a[0]:
                if temp_array[i] > lst[j]:
                    lst[k] = temp_array[i]
                    k -= 1
                    i -= 1

                    a_count = 0
                    b_count += 1

                    # If b runs out during linear merge
                    if i < 0:
                        while j >= a[0]:
                            lst[k] = lst[j]
                            k -= 1
                            j -= 1
                        return

                    if b_count >= gallop_thresh:
                        break

                else:
                    lst[k] = lst[j]
                    k -= 1
                    j -= 1

                    a_count += 1
                    b_count = 0

                    # If a runs out during linear merge
                    if j < a[0]:
                        while i >= 0:
                            lst[k] = temp_array[i]
                            k -= 1
                            i -= 1
                        return

                    if a_count >= gallop_thresh:
                        break

            # i, j, k are DECREMENTED in this case
            while True:
                # Look for the position of b[i] in a[0, j + 1]
                # ltr = False -> uses bisect_right()
                a_adv = gallop(lst, temp_array[i], a[0], j + 1, False)

                # Copy the elements from a_adv -> j to merge area
                # Go backwards to the index a_adv
                for x in range(j, a_adv - 1, -1):
                    lst[k] = lst[x]
                    k -= 1

                # # Update the a_count to check successfulness of galloping
                a_count = j - a_adv + 1

                # Decrement index j
                j = a_adv - 1

                # If run a runs out:
                if j < a[0]:
                    while i >= 0:
                        lst[k] = temp_array[i]
                        k -= 1
                        i -= 1
                    return

                # Copy the b[i] into the merge area
                lst[k] = temp_array[i]
                k -= 1
                i -= 1

                # If a runs out:
                if i < 0:
                    while j >= a[0]:
                        lst[k] = lst[j]
                        k -= 1
                        j -= 1
                    return

                # -------------------------------------------------

                # Look for the position of A[j] in B:
                b_adv = gallop(temp_array, lst[j], 0, i + 1, False)
                for y in range(i, b_adv - 1, -1):
                    lst[k] = temp_array[y]
                    k -= 1

                b_count = i - b_adv + 1
                i = b_adv - 1

                # If b runs out:
                if i < 0:
                    while j >= a[0]:
                        lst[k] = lst[j]
                        k -= 1
                        j -= 1
                    return

                # Copy the a[j] back to the merge area
                lst[k] = lst[j]
                k -= 1
                j -= 1

                # If a runs out:
                if j < a[0]:
                    while i >= 0:
                        lst[k] = temp_array[i]
                        k -= 1
                        i -= 1
                    return

                # if galloping proves to be unsuccessful, return to linear
                if a_count < gallop_thresh and b_count < gallop_thresh:
                    break

            # punishment for leaving galloping
            gallop_thresh += 1

    def merge_collapse(lst, stack):
        """The last three runs in the stack is A, B, C.
        Maintains invariants so that their lengths: A > B + C, B > C
        Translated to stack positions:
        stack[-3] > stack[-2] + stack[-1]
        stack[-2] > stack[-1]
        Takes a stack that holds many lists of type [s, e, bool, length]"""

        # This loops keeps running until stack has one element
        # or the invariant holds.
        while len(stack) > 1:
            if len(stack) >= 3 and stack[-3][3] <= stack[-2][3] + stack[-1][3]:
                if stack[-3][3] < stack[-1][3]:
                    # merge -3 and -2, merge at -3
                    merge(lst, stack, -3)
                else:
                    # merge -2 and -1, merge at -2
                    merge(lst, stack, -2)
            elif stack[-2][3] <= stack[-1][3]:
                # merge -2 and -1, merge at -2
                merge(lst, stack, -2)
            else:
                break

    def merge_force_collapse(lst, stack):
        """When the invariant holds and there are > 1 run
        in the stack, this function finishes the merging"""
        while len(stack) > 1:
            # Only merges at -2, because when the invariant holds,
            # merging would be balanced
            merge(lst, stack, -2)

    # Starting index
    s = 0

    # Ending index
    e = len(lst) - 1

    # The stack
    stack = []

    # Compute min_run using size of lst
    min_run = merge_compute_minrun(len(lst))

    while s <= e:

        # Find a run, return [start, end, bool, length]
        run = count_run(lst, s)

        # If decreasing, reverse
        if run[2] == False:
            reverse(lst, run[0], run[1])
            # Change bool to True
            run[2] = True

        # If length of the run is less than min_run
        if run[3] < min_run:
            # The number of indices by which we want to extend the run
            # either by the distance to the end of the lst
            # or by the length difference between run and minrun
            extend = min(min_run - run[3], e - run[1])

            # Extend the run using binary insertion sort
            bin_sort(lst, run[0], run[1], extend)

            # Update last index of the run
            run[1] = run[1] + extend

            # Update the run length
            run[3] = run[3] + extend

        # Push the run into the stack
        stack.append(run)

        # Start merging to maintain the invariant
        merge_collapse(lst, stack)

        # Update starting position to find the next run
        # If run[1] == end of the lst, s > e, loop exits
        s = run[1] + 1

    # Some runs might be left in the stack, complete the merging.
    merge_force_collapse(lst, stack)

    return lst


def quickSort(array):  # in-place | not-stable
    """
    Best : O(nlogn) Time | O(logn) Space
    Average : O(nlogn) Time | O(logn) Space
    Worst : O(nlogn) Time | O(logn) Space

    """
    if len(array) <= 1:
        return array
    smaller, equal, larger = [], [], []
    pivot = array[randrange(0, len(array))]
    for x in array:
        if x < pivot:
            smaller.append(x)
        elif x == pivot:
            equal.append(x)
        else:
            larger.append(x)
    return quickSort(smaller) + equal + quickSort(larger)


def quickSortHoare(array, low=0, high=None):  # in-place | not-stable
    """
    QuickSort using tail recursive + insertion sort + hoare
    
    Best : O(nlogn) Time | O(1) Space
    Average : O(nlogn) Time | O(1) Space
    Worst : O(nlogn) Time | O(1) Space
    """

    def insertSort(array, low=0, high=None):
        if high is None:
            high = len(array) - 1

        for i in range(low + 1, high + 1):
            j = i
            while j > 0 and array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]
                j -= 1

        return array

    if high is None:
        high = len(array) - 1

    while low < high and high - low > 16:
        q = partition(array, low, high)
        quickSortHoare(array, low, q)
        low = q + 1

    return insertSort(array, low, high)


def quickSortHeap(array, low=0, high=None, depth=None):
    """
    QuickSort using tail recursive + insertion sort + heap sort + hoare + median of three killer
    """

    def medianOf3(array, lowIdx, midIdx, highIdx):
        if (array[lowIdx] - array[midIdx]) * (array[highIdx] - array[lowIdx]) >= 0:
            return array[lowIdx]

        elif (array[midIdx] - array[lowIdx]) * (array[highIdx] - array[midIdx]) >= 0:
            return array[midIdx]

        else:
            return array[highIdx]

    def partition(array, low, high):
        pivot = medianOf3(array, low, (low + high) // 2, high)
        i = low - 1
        j = high + 1
        while True:
            i += 1
            while array[i] < pivot:
                i += 1
            j -= 1
            while array[j] > pivot:
                j -= 1

            if i >= j:
                return j

            array[i], array[j] = array[j], array[i]

    def insertSort(array, low=0, high=None):
        if high is None:
            high = len(array) - 1

        for i in range(low + 1, high + 1):
            j = i
            while j > 0 and array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]
                j -= 1

        return array

    if high is None:
        high = len(array) - 1
    if depth is None:
        depth = 2 * (len(array).bit_length() - 1)

    if depth == 0:
        return heapSort2(array)
    else:
        while high - low > 16:
            q = partition(array, low, high)
            quickSortHeap(array, low, q)
            low = q + 1

    return insertSort(array, low, high)


def partition(array, low, high):
    pivot = array[(high + low) // 2]
    # pivot = array[randint(low, high)]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while array[i] < pivot:
            i += 1
        j -= 1
        while array[j] > pivot:
            j -= 1

        if i >= j:
            return j

        array[i], array[j] = array[j], array[i]


def gnomeSort(array):  # in-place | stable
    """
    Best : O(n) Time | O(1) Space
    Average : O(n^2) Time | O(1) Space
    Worst : O(n^2) Time | O(1) Space
    """
    index = 0
    while index < len(array):
        if index == 0:
            index = index + 1
        if array[index] >= array[index - 1]:
            index = index + 1
        else:
            array[index], array[index - 1] = array[index - 1], array[index]
            index = index - 1

    return array


def mergeSort(array):
    """
    Best : O(nlogn) Time | O(n) Space
    Average : O(nlogn) Time | O(n) Space
    Worst : O(nlongn) Time | O(n) Space
    """

    def mergeHelper(main, startIdx, endIdx, aux):
        if startIdx == endIdx:
            return
        middleIdx = (startIdx + endIdx) // 2
        mergeHelper(aux, startIdx, middleIdx, main)
        mergeHelper(aux, middleIdx + 1, endIdx, main)
        merge(main, startIdx, middleIdx, endIdx, aux)

    def merge(main, startIdx, middleIdx, endIdx, aux):
        k = startIdx
        i = startIdx
        j = middleIdx + 1
        while i <= middleIdx and j <= endIdx:
            if aux[i] <= aux[j]:
                main[k] = aux[i]
                i += 1
            else:
                main[k] = aux[j]
                j += 1
            k += 1

        while i <= middleIdx:
            main[k] = aux[i]
            i += 1
            k += 1
        while j <= endIdx:
            main[k] = aux[j]
            j += 1
            k += 1

    if len(array) <= 1:
        return array

    aux = array[:]
    mergeHelper(array, 0, len(array) - 1, aux)
    return array


def goSort(array):
    def insertSort(array, begin=0, end=None):
        if end == None:
            end = len(array)
        for i in range(begin, end):
            j = i
            toChange = array[i]
            while j != begin and array[j - 1] > toChange:
                array[j] = array[j - 1]
                j -= 1
            array[j] = toChange
        return array

    def siftDown(array, lo, hi, first):
        root = lo
        while True:
            child = 2 * root + 1
            if child >= hi:
                break

            if child + 1 < hi and array[first + child] < array[first + child + 1]:
                child += 1

            if array[first + root] >= array[first + child]:
                return

            array[first + root], array[first + child] = (
                array[first + child],
                array[first + root],
            )
            root = child

    def heapSort(array, a, b):
        first = a
        lo = 0
        hi = b - a

        # Build heap with greatest element at top.
        for i in range((hi - 1) // 2, -1, -1):
            siftDown(array, i, hi, first)

        # Pop elements, largest first, into end of data.
        for i in range(hi - 1, -1, -1):
            array[first], array[first + i] = array[first + i], array[first]
            siftDown(array, lo, i, first)

    def unsigned(n):
        return n & 0xFFFFFFFF

    def medianOfThree(array, m1, m0, m2):
        # sort 3 elements
        if array[m1] < array[m0]:
            array[m1], array[m0] = array[m0], array[m1]

        # data[m0] <= data[m1]
        if array[m2] < array[m1]:
            array[m2], array[m1] = array[m1], array[m2]

            # data[m0] <= data[m2] && data[m1] < data[m2]
            if array[m1] < array[m0]:
                array[m1], array[m0] = array[m0], array[m1]

        # now data[m0] <= data[m1] <= data[m2]

    def doPivot(array, lo, hi):
        m = int(unsigned(lo + hi) >> 1)

        if hi - lo > 40:
            # Tukey's ``Ninther,'' median of three medians of three.
            s = (hi - lo) // 8
            medianOfThree(array, lo, lo + s, lo + 2 * s)
            medianOfThree(array, m, m - s, m + s)
            medianOfThree(array, hi - 1, hi - 1 - s, hi - 1 - 2 * s)

        medianOfThree(array, lo, m, hi - 1)

        pivot = lo
        a, c = lo + 1, hi - 1

        while a < c and array[a] < array[pivot]:
            a += 1
        b = a
        while True:
            while b < c and array[pivot] >= array[b]:
                b += 1
            while b < c and array[pivot] < array[c - 1]:
                c -= 1
            if b >= c:
                break

            array[b], array[c - 1] = array[c - 1], array[b]
            b += 1
            c -= 1

        protect = hi - c < 5
        if not protect and hi - c < (hi - lo) // 4:
            dups = 0
            if array[pivot] >= array[hi - 1]:
                array[c], array[hi - 1] = array[hi - 1], array[c]
                c += 1
                dups += 1
            if array[b - 1] >= array[pivot]:
                b -= 1
                dups += 1

            if array[m] >= array[pivot]:
                array[m], array[b - 1] = array[b - 1], array[m]
                b -= 1
                dups += 1

            protect = dups > 1

        if protect:
            while True:
                while a < b and array[b - 1] >= array[pivot]:
                    b -= 1
                while a < b and array[a] < array[pivot]:
                    a += 1
                if a >= b:
                    break

                array[a], array[b - 1] = array[b - 1], array[a]
                a += 1
                b -= 1

        array[pivot], array[b - 1] = array[b - 1], array[pivot]
        return b - 1, c

    def quickSort(array, a, b, maxDepth):
        while b - a > 12:  # Use ShellSort for slices <= 12 elements
            if maxDepth == 0:
                return heapSort(array, a, b)

            maxDepth -= 1
            mlo, mhi = doPivot(array, a, b)
            # Avoiding recursion on the larger subproblem guarantees
            # a stack depth of at most lg(b-a).
            if mlo - a < b - mhi:
                quickSort(array, a, mlo, maxDepth)
                a = mhi  # i.e., quickSort(array, mhi, b)
            else:
                quickSort(array, mhi, b, maxDepth)
                b = mlo  # i.e., quickSort(array, a, mlo)

        if b - a > 1:
            # Do ShellSort pass with gap 6
            # It could be written in this simplified form cause b-a <= 12
            for i in range(a + 6, b):
                if array[i] < array[i - 6]:
                    array[i], array[i - 6] = array[i - 6], array[i]

            return insertSort(array, a, b)

    maxDepth = 2 * (len(array).bit_length() - 1)
    quickSort(array, 0, len(array), maxDepth)
    return array


def gravitySort(array):  # Bead Sort
    """Gravity Sort (Bead Sort)
    
    Anti-gravity sort (works with negative integers well)

    """
    minimum, maximum = min(array), max(array)
    n = len(array)

    temp = [minimum] * n
    for i in range(maximum - 1, minimum - 1, -1):
        k = 0
        for j in range(n):
            if array[j] > i:
                temp[k] += 1
                k += 1

    for i in range(n):
        array[i] = temp[n - i - 1]

    return array


def doubleSelectionSort(array):
    """Double Selection Sort
    
    Best : O(n^2) Time | O(1) Space
    Average : O(n^2) Time | O(1) Space
    Worst : O(n^2) Time | O(1) Space

    """
    N = len(array)

    for currentIdx in range(N // 2):
        smallest = biggest = currentIdx

        for i in range(currentIdx + 1, N - currentIdx):
            if array[i] >= array[biggest]:
                biggest = i
            if array[i] < array[smallest]:
                smallest = i

        array[currentIdx], array[smallest] = array[smallest], array[currentIdx]
        if biggest == currentIdx:
            biggest = smallest
        array[N - currentIdx - 1], array[biggest] = (
            array[biggest],
            array[N - currentIdx - 1],
        )
    return array


def radixSort(array):
    # Time Complexity : O(n) | Space Complexity : O(n)
    def countSort(arr, exp):

        buckets = [0] * 10
        output = [None] * len(arr)

        for i in arr:
            buckets[(i // exp) % 10] += 1

        for i in range(1, 10):
            buckets[i] += buckets[i - 1]

        for i in reversed(range(0, len(arr))):
            current = (arr[i] // exp) % 10
            # arr[i] = buckets[current] - 1
            output[buckets[current] - 1] = arr[i]
            buckets[current] -= 1

        for i in range(len(arr)):
            arr[i] = output[i]

        return arr

    m = max(array)

    exp = 1
    while m / exp > 0:
        array = countSort(array, exp)

        exp *= 10

    return array
