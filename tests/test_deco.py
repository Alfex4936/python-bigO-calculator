from bigO import BigO, utils


@utils.is_sorted
def bubbleSort(array):  # in-place | stable
    isSorted = False
    counter = 1  # not correct
    while not isSorted:
        isSorted = True
        for i in range(len(array) - 1 - counter):
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                isSorted = False

        counter += 1

    return array


if __name__ == "__main__":
    bubbleSort(BigO.gen_random_ints(100))
