import math
import os
import string
import sys
from collections import Counter
from random import choice, getrandbits, randint, random
from timeit import default_timer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class BigOException(Exception):
    __module__ = Exception.__module__

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class BigO:
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
        from bigO import BigO
        from bigO import algorithm

        lib = BigO()

        lib.test(mySort, "random")
        lib.test_all(mySort)
        lib.runtime(algorithm.bubbleSort, "random", 5000)
        lib.compare(algorithm.bubbleSort, algorithm.insertSort, "random", 5000)
    """

    __slots__ = (
        "_is_window",
        "_coef",
        "_rms",
        "_cplx",
        "_O1",
        "_ON",
        "_OLogN",
        "_ONLogN",
        "_ON2",
        "_ON3",
        "_OLambda",
        "_fitCurves",
    )

    def __init__(self):
        self._is_window: bool = os.name == "nt"
        self._coef: float = 0.0
        self._rms: float = 0.0
        self._cplx: int = 0
        self._O1: int = 1
        self._ON: int = 2
        self._OLogN: int = 3
        self._ONLogN: int = 4
        self._ON2: int = 5
        self._ON3: int = 6

        self._OLambda: int = 7
        self._fitCurves = (
            self._O1,
            self._ON,
            self._OLogN,
            self._ONLogN,
            self._ON2,
            self._ON3,
        )

    def __repr__(self):
        return f"I'm using {'window' if self._is_window else 'posix'}"

    def _to_str(self) -> str:
        return self._complexity2str(self._cplx)

    def _complexity2str(self, cplx: int) -> str:
        return {
            self._ON: "O(n)",
            self._ON2: "O(n^2)",
            self._ON3: "O(n^3)",
            self._OLogN: "O(log(n)",
            self._ONLogN: "O(nlog(n))",
            self._O1: "O(1)",
        }.get(cplx, "f(n)")

    def _complexity2int(self, cplx: str) -> int:
        return {
            "O(1)": self._O1,
            "O(n)": self._ON,
            "O(log(n)": self._OLogN,
            "O(nlog(n))": self._ONLogN,
            "O(n^2)": self._ON2,
            "O(n^3)": self._ON3,
        }.get(cplx, self._ON)

    def _fittingCurve(self, cplx: int) -> Callable:
        def _bigO_ON(n):
            return n

        def _bigO_ON2(n):
            return n * n

        def _bigO_ON3(n):
            return n * n * n

        def _bigO_OLogN(n):
            return math.log2(n)

        def _bigO_ONLogN(n):
            return n * math.log2(n)

        def _bigO_O1(_):
            return 1.0

        return {
            self._O1: _bigO_O1,
            self._ON: _bigO_ON,
            self._ON2: _bigO_ON2,
            self._ON3: _bigO_ON3,
            self._OLogN: _bigO_OLogN,
            self._ONLogN: _bigO_ONLogN,
        }.get(cplx, _bigO_O1)

    def _minimalLeastSq(self, arr: List[Any], times: List[float], function: Callable):
        sigmaGnSquared = 0.0
        sigmaTime = 0.0
        sigmaTimeGn = 0.0

        floatN = float(len(arr))

        for i in range(len(arr)):
            gnI = function(arr[i])
            sigmaGnSquared += gnI * gnI
            sigmaTime += times[i]
            sigmaTimeGn += times[i] * gnI

        result = BigO()
        result._cplx = self._OLambda

        result._coef = sigmaTimeGn / sigmaGnSquared

        rms = 0.0
        for i in range(len(arr)):
            fit = result._coef * function(arr[i])
            rms += math.pow(times[i] - fit, 2)

        mean = sigmaTime / floatN
        result._rms = math.sqrt(rms / floatN) / mean

        return result

    def _estimate(self, n: List[int], times: List[float]):
        assert len(n) == len(
            times
        ), f"ERROR: Length mismatch between N:{len(n)} and TIMES:{len(times)}."
        assert len(n) >= 2, "ERROR: Need at least 2 runs."

        # assume that O1 is the best case
        bestFit = self._minimalLeastSq(n, times, self._fittingCurve(self._O1))
        bestFit._cplx = self._O1

        for fit in self._fitCurves:
            currentFit = self._minimalLeastSq(n, times, self._fittingCurve(fit))
            if currentFit._rms < bestFit._rms:
                bestFit = currentFit
                bestFit._cplx = fit

        return bestFit

    @staticmethod
    def genRandomPositive(size: int = 10):
        return [randint(1, size) for _ in range(size)]

    @staticmethod
    def genRandomArray(size: int = 10):
        return [randint(-size, size) for _ in range(size)]

    @staticmethod
    def genRandomBigArray(size: int = 10):
        array = []
        append = array.append
        for _ in range(size):
            isPositive = random() < 0.5
            nextValue = getrandbits(50)  # More than 100 trillion
            if not isPositive:
                nextValue = -nextValue
            append(nextValue)
        return array

    @staticmethod
    def genRandomString(size: int = 10, stringLen: int = 0):
        stringLen = stringLen or size // 2

        letters = string.ascii_lowercase + string.digits
        array = [
            "".join(choice(letters) for _ in range(randint(1, stringLen)))
            for _ in range(size)
        ]  # secrets.choice?
        return array

    @staticmethod
    def genSortedArray(size: int = 10):
        return [i for i in range(size)]

    @staticmethod
    def genReversedArray(size: int = 10):
        return [i for i in reversed(range(size))]

    def genPartialArray(self, size: int = 10):
        array = self.genRandomArray(size)
        sorted_array = self.genSortedArray(size)

        array[size // 4 : size // 2] = sorted_array[size // 4 : size // 2]
        return array

    def genKsortedArray(self, size: int = 10, k: Optional[int] = None):
        def _reverseRange(array, a, b):
            i = a
            j = b - 1
            while i < j:
                array[i], array[j] = array[j], array[i]
                i += 1
                j -= 1

            return array

        if k is None:
            k = size.bit_length()

        if size < k:
            raise BigOException("K must be smaller than the size.")

        if k == 0:
            return self.genSortedArray(size)
        elif size == k:
            return self.genReversedArray(size)

        array = [value for value in range(-size // 2, size // 2)]

        right = randint(0, k - 1)
        while right >= size - k:
            right -= 1

        _reverseRange(array, 0, k + 1)
        _reverseRange(array, size - right, size)

        return array

    @staticmethod
    def genEqualArray(size: int = 10):
        n = randint(-size, size)
        return [n for _ in range(size)]

    @staticmethod
    def genAlmostEqualArray(size: int = 10):
        return [randint(-1, 1) + size for _ in range(size)]

    def genHoleArray(
        self, size: int = 10
    ):  # returns equal array with only one different element
        arr = self.genEqualArray(size)
        arr[randint(-size, size)] = -sys.maxsize
        return arr

    @staticmethod
    def isAscendingSorted(array: List[Any]):
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

    @staticmethod
    def isDescendingSorted(array: List[Any]) -> Tuple[bool, Optional[int]]:
        """Is correctly descending sorted? Time: O(n)

        Args:
            array [List]: Array to check if it is sorted correctly

        Returns:
            isSorted, index [bool, int]: returns True/False with unsorted index
        """
        # Descending order
        for i in range(len(array) - 1, 0, -1):
            if array[i] > array[i - 1]:
                return False, i + 1

        return True, None

    def test(
        self,
        functionName: Callable,
        array: str,
        limit: bool = True,
        prtResult: bool = True,
    ) -> str:
        """
        ex) test(bubbleSort, "random")

        Args:
            functionName (Callable): a function to call |
            array (str): "random", "big", "sorted", "reversed", "partial", "Ksorted", "string",
            "hole", "equal", "almost_equal" |
            limit (bool) = True: To terminate before it takes forever to sort (usually 10,000) |
            prtResult (bool) = True: Whether to print the result by itself

        Returns:
            complexity (str) : ex) O(n)

        """
        # if functionName.__code__.co_argcount - 1 != len(args):
        #     raise BigOException(
        #         f"{functionName.__name__} takes {functionName.__code__.co_argcount - 1} but got more {args}."
        #     )
        if self._is_window:
            from win10toast import ToastNotifier
        else:
            toaster = None
            ToastNotifier = None

        sizes: List[int] = [10, 100, 1000, 10000, 100000]
        maxIter: int = 5
        times: List[float] = []
        isSlow: bool = False  # To see if sorting algorithm takes forever

        if prtResult:
            print(f"Running {functionName.__name__}({array} array)...")
        if self._is_window:
            toaster = ToastNotifier()
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
                nums = self.genRandomString(size=size, stringLen=100)
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
                    else:
                        msg = ""
                    if not isSorted:
                        raise BigOException(
                            f"{functionName.__name__} doesn't sort correctly.\nAt {index} index: [{msg}]"
                        )

            if (
                timeTaken >= 4 and limit
            ):  # if it takes more than 4 seconds to sort an array, then break it
                isSlow = True

            timeTaken /= maxIter
            times.append(timeTaken)

        complexity = self._estimate(sizes, times)
        cplx = complexity._to_str()
        # estimatedTime = sum(times)

        if prtResult:
            print(f"Completed {functionName.__name__}({array} array): {cplx}")
            # print(f"Time took: {estimatedTime:.5f}s")

        if self._is_window:
            toaster.show_toast(
                "Big-O Caculator",
                f"Completed {functionName.__name__}({array} array): {cplx}",
                duration=3,
            )

        return cplx

    def test_all(self, function: Callable) -> Dict[str, str]:
        """
        ex) test_all(bubbleSort)

        Args:
            function [Callable]: a function to call

        Returns:
            Dict[str, str]: ex) {"random": "O(n)" ...}
        """
        result = {
            "random": "0",
            "sorted": "0",
            "reversed": "0",
            "partial": "0",
            "Ksorted": "0",
            "almost_equal": "0",
        }

        bestCase = self._complexity2int("O(n^3)")
        worstCase = self._complexity2int("O(1)")

        print(f"Running {function.__name__}(tests)")
        for test in result:
            cplx = self.test(function, test, prtResult=False)
            result[test] = cplx
            cplxInt = self._complexity2int(cplx)

            if cplxInt < bestCase:
                bestCase = cplxInt
            if cplxInt > worstCase:
                worstCase = cplxInt

        averageCase, _ = Counter(result.values()).most_common(1)[0]

        print(f"Best : {self._complexity2str(bestCase)} Time")
        print(f"Average : {averageCase} Time")
        print(f"Worst : {self._complexity2str(worstCase)} Time")

        return result

    def runtime(
        self,
        function: Callable,
        array: Union[str, List[Any]],
        size: int = 0,
        epoch: int = 1,
        prtResult: bool = True,
    ) -> Tuple[float, Optional[List[Any]]]:
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

        # TODO - runtime(lambda array, start, end: sort(array, start, end), "random", 5000)
        # In case when we pass a function like
        # lambda array, start, end: quickSort(array, start, end)
        # isLambda = (
        #     callable(function)
        #     and function.__name__ == "<lambda>"
        #     and len(signature(function).parameters) > 1
        # )

        if isinstance(array, list):
            nums = array
            array = "custom"
            size = len(nums)
            if size == 0:
                raise BigOException("Length of array must be greater than 0.")
        else:
            if size == 0:
                raise BigOException("Length of array must be greater than 0.")
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
        result = None
        # args = None
        # if isLambda:
        #     result = function(nums, 0, len(nums) - 1)

        for _ in range(epoch):
            # args = function.__code__.co_varnames
            # args = (nums, 0, len(nums) - 1)
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
                else:
                    msg = ""

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
        self,
        function1: Callable,
        function2: Callable,
        array: Union[str, List[Any]],
        size: int = 50,
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
        s: str = ""

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
            func1_sum: float = 0.0
            func2_sum: float = 0.0
            wins: int = 0

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
