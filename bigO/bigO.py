import math
import string
from collections import Counter
from random import choice, getrandbits, randint, random
from timeit import default_timer
from typing import Any, Callable, Dict, List, Tuple

from win10toast import ToastNotifier


class bigO:
    """
    Big-O calculator

    Methods
    -------
    test(function, array, limit=True, prtResult=True) -> Tuple[complexity, executionTime]:
        Returns time complexity and the execution time to sort arrays with your function

    test_all(function) -> Dict[str, int]:
        Returns dictionary with all cases timecomplexity

    runtime(function, array, size) -> Tuple[executionTime, sorted result]:
        Returns executionTime and the result
        
    compare(function1, function2, array, size) -> Dict{functionName: executionTime}
        Returns dictionary with execution time on each function

    Usage
    -----
        from bigO import bigO
        from bigO import algorithm

        lib = bigO.bigO()
        
        lib.test(mySort, "random")
        lib.test_all(mySort)
        lib.runtime(algorithm.bubbleSort, "random", 5000)
        lib.compare(algorithm.bubbleSort, algorithm.insertSort, "random", 5000)
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

    def genRandomPositive(_, size: int = 10):
        return [randint(1, size) for _ in range(size)]

    def genRandomArray(_, size: int = 10):
        return [randint(-size, size) for _ in range(size)]
    
    def genRandomBigArray(_, size: int = 10):
        array = []
        for _ in range(size):
            isPositive = True if random() >= 0.5 else False
            nextValue = getrandbits(50)  # More than 100 trillion
            if not isPositive:
                nextValue = -nextValue
            array.append(nextValue)
        return array
        

    def genRandomString(_, stringLen: int = None, size: int = 10):
        if stringLen == None:
            stringLen = size // 2

        letters = string.ascii_lowercase + string.digits
        array = [
            "".join(choice(letters) for _ in range(randint(1, stringLen)))
            for _ in range(size)
        ]  # secrets.choice?
        return array

    def genSortedArray(_, size: int = 10):
        return [i for i in range(size)]

    def genReversedArray(_, size: int = 10):
        return [i for i in reversed(range(size))]

    def genPartialArray(self, size: int = 10):
        array = self.genRandomArray(size)
        sorted_array = self.genSortedArray(size)

        array[size // 4 : size // 2] = sorted_array[size // 4 : size // 2]
        return array

    def genKsortedArray(self, size: int = 10, k: int = None):
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

    def genEqualArray(_, size: int = 10):
        n = randint(-size, size)
        return [n for _ in range(size)]

    def genAlmostEqualArray(_, size: int = 10):
        return [randint(-1, 1) + size for _ in range(size)]

    def genHoleArray(
        self, size: int = 10
    ):  # returns equal array with only one different element
        arr = self.genEqualArray(size)
        arr[randint(-size, size)] = -9999
        return arr

    def isAscendingSorted(_, array: List[Any]):
        """Is correctly ascending sorted? Time: O(n)

        Args:
            array [List]: Array to check if it is sorted correctly

        Returns:
            isSorted, index [bool, int]: returns True/False with unsorted index
        """
        # Ascending order
        for i in range(len(array) - 1):
            if array[i] > array[i + 1]:
                return False, i + 1

        return True, None

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
            array (str): "random", "big", "sorted", "reversed", "partial", "Ksorted", "string",
            "hole", "equal", "almost_equal" |
            limit (bool): To terminate before it takes forever to sort (usually 10,000) |
            prtResult (bool): Whether to print the result by itself (default = True)

        Returns:
            complexity (str) : ex) O(n) |
            time (float) : Time took to sort all 5 different arrays in second (max=100,000)

        """
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
            elif array == "big":
                nums = self.genRandomBigArray(size)
            elif array == "sorted":
                nums = self.genSortedArray(size)
            elif array == "partial":
                nums = self.genPartialArray(size)
            elif array == "reversed":
                nums = self.genReversedArray(size)
            elif array == "ksorted":
                nums = self.genKsortedArray(size, size.bit_length())
            elif array == "string":
                nums = self.genRandomString(stringLen=100, size=size)
            elif array == "hole":
                nums = self.genHoleArray(size)
            elif array == "equal":
                nums = self.genEqualArray(size)
            elif array == "almost_equal":
                nums = self.genAlmostEqualArray(size)
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
                    isSorted, index = self.isAscendingSorted(result)
                    if index == 1:
                        msg = f"{result[index - 1]}, {result[index]}..."
                    elif index == len(result) - 1:
                        msg = f"...{result[index - 1]}, {result[index]}"
                    elif isinstance(index, int):
                        msg = f"...{result[index - 1]}, {result[index]}, {result[index + 1]}..."
                    assert (
                        isSorted
                    ), f"{functionName.__name__} doesn't sort correctly.\nAt {index} index: [{msg}]"

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

    def test_all(self, function: Callable) -> Dict[str, int]:
        """
        ex) test_all(bubbleSort)

        Args:
            function [Callable]: a function to call

        Returns:
            Dict[str, int]: ex) {"random": "O(n)" ...}
        """
        result = {
            "random": 0,
            "sorted": 0,
            "reversed": 0,
            "partial": 0,
            "Ksorted": 0,
            "almost_equal": 0,
        }

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

    def runtime(
        self,
        function: Callable,
        array,
        size: int = None,
        epoch: int = 1,
        prtResult: bool = True,
    ) -> Tuple[float, List[Any]]:
        """
        ex) runtime(bubbleSort, "random", 5000)

        Args:
            function [Callable]: a function to call |
            array: "random", "big", "sorted", "partial", "reversed", "Ksorted" ,
            "hole", "equal", "almost_equal" or your custom array |
            size [int]: How big test array should be |
            epoch [int]: How many tests to run and calculte average |
            prtResult (bool): Whether to print the result by itself (default = True) |

        Returns:
            Tuple[float, List[Any]]: An execution time and sorted result
        """
        if epoch < 1:
            epoch = 1

        if isinstance(array, list):
            nums = array
            array = "custom"
            size = len(nums)
        else:
            array = array.lower()
            if array == "random":
                nums = self.genRandomArray(size)
            elif array == "big":
                nums = self.genRandomBigArray(size)
            elif array == "sorted":
                nums = self.genSortedArray(size)
            elif array == "partial":
                nums = self.genPartialArray(size)
            elif array == "reversed":
                nums = self.genReversedArray(size)
            elif array == "ksorted":
                nums = self.genKsortedArray(size, size.bit_length())
            elif array == "string":
                nums = self.genRandomString(size)
            elif array == "hole":
                nums = self.genHoleArray(size)
            elif array == "equal":
                nums = self.genEqualArray(size)
            elif array == "almost_equal":
                nums = self.genAlmostEqualArray(size)
            else:  # default = random array
                nums = self.genRandomArray(size)

        if prtResult:
            print(f"Running {function.__name__}(len {size} {array} array)")

        timeTaken = 0.0

        for _ in range(epoch):
            timeStart = default_timer()
            result = function(nums)
            timeEnd = default_timer()

            timeTaken += timeEnd - timeStart

            if result != None:
                isSorted, index = self.isAscendingSorted(result)
                if index == 1:
                    msg = f"{result[index - 1]}, {result[index]}..."
                elif index == len(result) - 1:
                    msg = f"...{result[index - 1]}, {result[index]}"
                elif isinstance(index, int):
                    msg = f"...{result[index - 1]}, {result[index]}, {result[index + 1]}..."

                if not isSorted:
                    # Just see the result if it doesn't sort correctly
                    print(
                        f"{function.__name__} doesn't sort correctly.\nAt {index} index: [{msg}]"
                    )

        finalTime = timeTaken / epoch

        if prtResult:
            print(f"Took {finalTime:.5f}s to sort {function.__name__}({array})")

        return finalTime, result

    def compare(
        self, function1: Callable, function2: Callable, array, size: int = None
    ) -> Dict:
        """
        ex) compare(bubbleSort, insertSort, "random", 5000)

        Args:
            function1 [Callable]: a function to compare |
            function2 [Callable]: a function to compare |
            array [str]|[List]: "random", "big", "sorted", "partial", "reversed", "Ksorted", 
            "hole", "equal", "almost_equal", "all" or your custom array |
            size [int]: How big test array should be |

        Returns:
            Dict: function1 execution time and function2 execution time
        """
        s = ""

        if array == "all":
            test = [
                "random",
                "big",
                "sorted",
                "reversed",
                "partial",
                "Ksorted",
                "hole",
                "equal",
                "almost_equal",
            ]
            func1_sum = 0.0
            func2_sum = 0.0
            wins = 0

            print(f"Running {function1.__name__}(tests) vs {function2.__name__}(tests)")
            for arr in test:
                function1_time, _ = self.runtime(
                    function1, arr, size, epoch=3, prtResult=False
                )
                func1_sum += function1_time

                function2_time, _ = self.runtime(
                    function2, arr, size, epoch=3, prtResult=False
                )
                func2_sum += function2_time

                if function1_time > function2_time:
                    wins += 1

            func1_sum /= len(test)
            func2_sum /= len(test)
            function1_time = func1_sum
            function2_time = func2_sum

            wins = wins if function1_time > function2_time else len(test) - wins
            array = f"{wins} of {len(test)}"
            s = "s"
        else:
            if isinstance(array, list):
                nums = array
                array = "custom"
                size = len(nums)

            function1_time, _ = self.runtime(
                function1, array, size, epoch=3, prtResult=False
            )
            function2_time, _ = self.runtime(
                function2, array, size, epoch=3, prtResult=False
            )

        timeDiff = abs(function1_time - function2_time)

        if function1_time < function2_time:
            percentage = function2_time / function1_time * 100.0 - 100.0
            print(
                f"{function1.__name__} is {percentage:.1f}% faster than {function2.__name__} on {array} case{s}"
            )
            print(f"Time Difference: {timeDiff:.5f}s")
        else:
            percentage = function1_time / function2_time * 100.0 - 100.0
            print(
                f"{function2.__name__} is {percentage:.1f}% faster than {function1.__name__} on {array} case{s}"
            )
            print(f"Time Difference: {timeDiff:.5f}s")

        return {function1.__name__: function1_time, function2.__name__: function2_time}
