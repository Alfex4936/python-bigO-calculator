from collections import Counter
import math
from random import randint
from timeit import default_timer
from typing import Any, Callable, List, Tuple

from win10toast import ToastNotifier


class bigO:
    """
    Big-O calculator

    Methods
    -------
    test(function, str):
        Returns time complexity and the execution time to sort arrays with your function

    Usage
    -----
        from bigO import bigO
        bigO = bigO.bigO()
        bigO.test(mySort, "random")
    """

    def __init__(self):
        self.coef = 0.0
        self.rms = 0.0
        self.cplx = 0
        self.O1 = 1
        self.ON = 2
        self.OLogN = 3
        self.ONLogN = 4
        self.ON2 = 5
        self.ON3 = 6

        self.OLambda = 7
        self.fitCurves = [self.O1, self.ON, self.OLogN, self.ONLogN, self.ON2, self.ON3]

    def str(self):
        return self.complexity2str(self.cplx)

    def complexity2str(self, cplx: int) -> str:
        return {
            self.ON: "O(n)",
            self.ON2: "O(n^2)",
            self.ON3: "O(n^3)",
            self.OLogN: "O(log(n)",
            self.ONLogN: "O(nlog(n))",
            self.O1: "O(1)",
        }.get(cplx, "f(n)")

    def complexity2int(self, cplx: str) -> int:
        return {
            "O(1)": self.O1,
            "O(n)": self.ON,
            "O(log(n)": self.OLogN,
            "O(nlog(n))": self.ONLogN,
            "O(n^2)": self.ON2,
            "O(n^3)": self.ON3,
        }.get(cplx, self.ON)

    def fittingCurve(self, cplx: int) -> Callable:
        def bigO_ON(n):
            return n

        def bigO_ON2(n):
            return n * n

        def bigO_ON3(n):
            return n * n * n

        def bigO_OLogN(n):
            return math.log2(n)

        def bigO_ONLogN(n):
            return n * math.log2(n)

        def bigO_O1(_):
            return 1.0

        return {
            self.O1: bigO_O1,
            self.ON: bigO_ON,
            self.ON2: bigO_ON2,
            self.ON3: bigO_ON3,
            self.OLogN: bigO_OLogN,
            self.ONLogN: bigO_ONLogN,
        }.get(cplx, bigO_O1)

    def minimalLeastSq(self, arr: List[Any], times: float, function: Callable):
        # sigmaGn = 0.0
        sigmaGnSquared = 0.0
        sigmaTime = 0.0
        sigmaTimeGn = 0.0

        floatN = float(len(arr))

        for i in range(len(arr)):
            gnI = function(arr[i])
            # sigmaGn = gnI
            sigmaGnSquared += gnI * gnI
            sigmaTime += times[i]
            sigmaTimeGn += times[i] * gnI

        result = bigO()
        result.cplx = self.OLambda

        result.coef = sigmaTimeGn / sigmaGnSquared

        rms = 0.0
        for i in range(len(arr)):
            fit = result.coef * function(arr[i])
            rms += math.pow(times[i] - fit, 2)

        mean = sigmaTime / floatN
        result.rms = math.sqrt(rms / floatN) / mean

        return result

    def estimate(self, n: List[int], times: List[float]):
        assert len(n) == len(
            times
        ), f"ERROR: Length mismatch between N:{len(n)} and TIMES:{len(times)}."
        assert len(n) >= 2, "ERROR: Need at least 2 runs."

        bestFit = bigO()

        # assume that O1 is the best case
        bestFit = self.minimalLeastSq(n, times, self.fittingCurve(self.O1))
        bestFit.cplx = self.O1

        for fit in self.fitCurves:
            currentFit = self.minimalLeastSq(n, times, self.fittingCurve(fit))
            if currentFit.rms < bestFit.rms:
                bestFit = currentFit
                bestFit.cplx = fit

        return bestFit

    def genRandomArray(_, size: int):
        array = [randint(-size, size) for i in range(size)]
        return array

    def genSortedArray(_, size: int):
        array = [i for i in range(size)]
        return array

    def genReversedArray(_, size: int):
        array = [i for i in reversed(range(size))]
        return array

    def genPartialArray(self, size: int):
        array = self.genRandomArray(size)
        sorted_array = self.genSortedArray(size)

        array[size // 4 : size // 2] = sorted_array[size // 4 : size // 2]
        return array

    def genKsortedArray(self, size: int, k: int = None):
        def reverseRange(array, a, b):
            i = a
            j = b - 1
            while i < j:
                array[i], array[j] = array[j], array[i]
                i += 1
                j -= 1

            return array

        if k is None:
            k = size.bit_length()

        assert size >= k, "K must be smaller than the size."
        if k == 0:
            return self.genSortedArray(size)
        elif size == k:
            return self.genReversedArray(size)

        array = [value for value in range(-size // 2, size // 2)]

        right = randint(0, k - 1)
        while right >= size - k:
            right -= 1

        reverseRange(array, 0, k + 1)
        reverseRange(array, size - right, size)

        return array

    def test(
        self,
        functionName: Callable,
        array: str,
        limit: bool = True,
        prtResult: bool = True,
    ) -> Tuple[str, float]:
        """
        ex) test(bubbleSort, "random")

        Args:
            functionName (Callable): a function to call |
            array (str): "random", "sorted", "reversed", "partial", "Ksorted" |
            limit (bool): To terminate before it takes forever to sort (usually 10,000) |
            prtResult (bool): Whether to print the result by itself (default = True)

        Returns:
            complexity (str) : ex) O(n) |
            time (float) : Time took to sort all 5 different arrays in second (max=100,000)

        """
        # TODO : internal sorting algorithms, test all option
        sizes = [10, 100, 1000, 10000, 100000]
        maxIter = 5
        times = []
        isSlow = False  # To see if sorting algorithm takes forever

        toaster = ToastNotifier()
        if prtResult:
            print(f"Running {functionName.__name__}({array} array)...")
        toaster.show_toast(
            "Big-O Caculator",
            f"Running {functionName.__name__}({array} array)...",
            duration=2,
        )

        for size in sizes:

            if isSlow:
                sizes = sizes[: len(times)]
                break

            timeTaken = 0.0
            nums = []

            array = array.lower()
            if array == "random":
                nums = self.genRandomArray(size)
            elif array == "sorted":
                nums = self.genSortedArray(size)
            elif array == "partial":
                nums = self.genPartialArray(size)
            elif array == "reversed":
                nums = self.genReversedArray(size)
            elif array == "ksorted":
                nums = self.genKsortedArray(size, size.bit_length())
            # elif array == "custom":
            #    nums = custom
            #    assert len(nums) != 0, "Please, pass the custom array you want.
            else:  # default = random array
                nums = self.genRandomArray(size)

            currentIter = 0

            while currentIter < maxIter:
                start = default_timer()
                result = functionName(nums)
                end = default_timer()
                timeTaken += end - start
                currentIter += 1

                if result != None:
                    assert result == sorted(
                        nums
                    ), "This function doesn't sort correctly."

            if (
                timeTaken >= 5 and limit
            ):  # if it takes more than 5 seconds to sort one array, break
                isSlow = True

            timeTaken /= maxIter
            times.append(timeTaken)

        complexity = self.estimate(sizes, times)
        estimatedTime = sum(times)

        if prtResult:
            print(
                f"Completed {functionName.__name__}({array} array): {complexity.str()}"
            )
            print(f"Time took: {estimatedTime:.5f}s")
        toaster.show_toast(
            "Big-O Caculator",
            f"Completed {functionName.__name__}({array} array): {complexity.str()}",
            duration=3,
        )

        return complexity.str(), estimatedTime

    def test_all(self, function):
        result = {"random": 0, "sorted": 0, "reversed": 0, "partial": 0, "Ksorted": 0}

        bestCase = self.complexity2int("O(n^3)")
        worstCase = self.complexity2int("O(1)")

        print(f"Running {function.__name__}(tests)")
        for test in result:
            cplx, _ = self.test(function, test, prtResult=False)
            result[test] = cplx
            cplxInt = self.complexity2int(cplx)

            if cplxInt < bestCase:
                bestCase = cplxInt
            if cplxInt > worstCase:
                worstCase = cplxInt

        averageCase, _ = Counter(result.values()).most_common(1)[0]

        print(f"Best : {self.complexity2str(bestCase)} Time")
        print(f"Average : {averageCase} Time")
        print(f"Worst : {self.complexity2str(worstCase)} Time")

        return result
