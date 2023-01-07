import math
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
    test(function, array, limit=True, show_result=True) -> Tuple[complexity, executionTime]:
        Returns time complexity and the execution time to sort arrays with your function

    test_all(function) -> Dict[str, int]:
        Returns dictionary with all cases timecomplexity

    runtime(function, array, size) -> Tuple[executionTime, sorted result]:
        Returns executionTime and the result

    compare(function1, function2, array, size) -> Dict{function: executionTime}
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
        # "_is_window",
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
        # self._is_window: bool = os.name == "nt"
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
    def gen_random_positive_ints(size: int = 10) -> List[int]:
        """
        Generate random positive ints

        ex) [7, 5, 8, 4, 8, 10, 2, 6, 10, 6]
        """
        return [randint(1, size) for _ in range(size)]

    @staticmethod
    def gen_random_ints(size: int = 10) -> List[int]:
        """
        Generate random ints

        ex) [-1, 10, -6, -1, -8, -9, 5, -2, 10, -9]
        """
        return [randint(-size, size) for _ in range(size)]

    @staticmethod
    def gen_random_big_ints(size: int = 10) -> List[int]:
        """
        Generate random big ints (More than 100 trillion)

        ex) [-135088285944124, 783720798870257, -34720574126312, -718786035797451, 309948566813132, -300094477426098, 279527848265212, 483464802488144, -1002326358321167, -1030593724928167]
        """
        array: List[int] = []
        append = array.append
        for _ in range(size):
            isPositive = random() < 0.5
            nextValue = getrandbits(50)  # More than 100 trillion
            if not isPositive:
                nextValue = -nextValue
            append(nextValue)
        return array

    @staticmethod
    def gen_random_strings(size: int = 10, stringLen: int = 0) -> List[str]:
        """
        Generate random strings

        ex) ['ys4', 'u19iq', '93r6q', '9uao', 'i960', 'jw', 'tt', 'l', '2t99', 'f4l']
        """
        stringLen = stringLen or size // 2

        letters = string.ascii_lowercase + string.digits
        array: List[str] = [
            "".join(choice(letters) for _ in range(randint(1, stringLen)))
            for _ in range(size)
        ]  # secrets.choice?
        return array

    @staticmethod
    def gen_sorted_ints(size: int = 10) -> List[int]:
        """
        Generate sorted ints from 0 to size -1

        ex) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        return [i for i in range(size)]

    @staticmethod
    def gen_reversed_ints(size: int = 10) -> List[int]:
        """
        Generate sorted ints from size - 1 to 0

        ex) [-1, -2, -3, 1, 2, 3, 0, ... ]
        """
        return [i for i in reversed(range(size))]

    def gen_partial_ints(self, size: int = 10) -> List[int]:
        """
        Generate partially sorted ints from size - 1 to 0

        ex) [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        """
        array = self.gen_random_ints(size)
        sorted_array = self.gen_sorted_ints(size)

        array[size // 4 : size // 2] = sorted_array[size // 4 : size // 2]
        return array

    def gen_ksorted_ints(self, size: int = 10, k: Optional[int] = None) -> List[int]:
        """
        Generate K sorted ints

        ex) [-1, -2, -3, -4, -5, 0, 1, 2, 4, 3]
        """

        def _reverse_range(array, a, b):
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
            return self.gen_sorted_ints(size)
        elif size == k:
            return self.gen_reversed_ints(size)

        array = [value for value in range(-size // 2, size // 2)]

        right = randint(0, k - 1)
        while right >= size - k:
            right -= 1

        _reverse_range(array, 0, k + 1)
        _reverse_range(array, size - right, size)

        return array

    @staticmethod
    def gen_equal_ints(size: int = 10) -> List[int]:
        """
        Generate a list of one integer

        ex) [-8, -8, -8, -8, -8, -8, -8, -8, -8, -8]
        """
        n = randint(-size, size)
        return [n for _ in range(size)]

    @staticmethod
    def gen_almost_equal_ints(size: int = 10) -> List[int]:
        """
        Generate a list of ints with small difference

        ex) [9, 11, 9, 9, 11, 9, 10, 9, 10, 9]
        """
        return [randint(-1, 1) + size for _ in range(size)]

    @staticmethod
    def gen_hole_ints(
        size: int = 10,
    ) -> List[int]:  # returns equal array with only one different element
        """
        Generate

        ex) [-1, -2, -3, -4, -5, 0, 1, 2, 4, 3]
        """
        n = randint(-size, size)
        arr = [n for _ in range(size)]  # gen_equal_ints

        arr[randint(-size, size)] = -sys.maxsize
        return arr

    @staticmethod
    def is_ascending_sorted(array: List[Any]):
        """Is correctly ascending sorted? Time: O(n)

        Args:
            array [List]: Array to check if it is sorted correctly

        Returns:
            is_sorted, index [bool, int]: returns True/False with unsorted index
        """
        # Ascending order
        for i in range(len(array) - 1):
            if array[i] > array[i + 1]:
                return False, i + 1

        return True, None

    @staticmethod
    def is_descending_sorted(array: List[Any]) -> Tuple[bool, Optional[int]]:
        """Is correctly descending sorted? Time: O(n)

        Args:
            array [List]: Array to check if it is sorted correctly

        Returns:
            is_sorted, index [bool, int]: returns True/False with unsorted index
        """
        # Descending order
        for i in range(len(array) - 1, 0, -1):
            if array[i] > array[i - 1]:
                return False, i + 1

        return True, None

    def test(
        self,
        function: Callable,
        array: str,
        limit: bool = True,
        show_result: bool = True,
    ) -> str:
        """
        ex) test(bubbleSort, "random")

        Args:
            function (Callable): a function to call |
            array (str): "random", "big", "sorted", "reversed", "partial", "Ksorted", "string",
            "hole", "equal", "almost_equal" |
            limit (bool) = True: To terminate before it takes forever to sort (usually 10,000) |
            show_result (bool) = True: Whether to print the result by itself

        Returns:
            complexity (str) : ex) O(n)

        """
        # if function.__code__.co_argcount - 1 != len(args):
        #     raise BigOException(
        #         f"{function.__name__} takes {function.__code__.co_argcount - 1} but got more {args}."
        #     )

        # if self._is_window:
        #     from win10toast import ToastNotifier
        # else:
        #     toaster = None
        #     ToastNotifier = None

        sizes: List[int] = [10, 100, 1000, 10000, 100000]
        maxIter: int = 5
        times: List[float] = []
        is_slow: bool = False  # To see if sorting algorithm takes forever

        if show_result:
            print(f"Running {function.__name__}({array} array)...")
        # if self._is_window:
        #     toaster = ToastNotifier()
        #     toaster.show_toast(
        #         "Big-O Caculator",
        #         f"Running {function.__name__}({array} array)...",
        #         duration=2,
        #     )

        for size in sizes:
            if is_slow:
                sizes = sizes[: len(times)]
                break

            sum_time = 0.0

            array = array.lower()
            if array == "random":
                nums: List[int] = self.gen_random_ints(size)
            elif array == "big":
                nums = self.gen_random_big_ints(size)
            elif array == "sorted":
                nums = self.gen_sorted_ints(size)
            elif array == "partial":
                nums = self.gen_partial_ints(size)
            elif array == "reversed":
                nums = self.gen_reversed_ints(size)
            elif array == "ksorted":
                nums = self.gen_ksorted_ints(size, size.bit_length())
            elif array == "string":
                nums = self.gen_random_strings(size=size, stringLen=100)  # type: ignore
            elif array == "hole":
                nums = self.gen_hole_ints(size)
            elif array == "equal":
                nums = self.gen_equal_ints(size)
            elif array == "almost_equal":
                nums = self.gen_almost_equal_ints(size)
            # elif array == "custom":
            #    nums = custom
            #    assert len(nums) != 0, "Please, pass the custom array you want.
            else:  # default = random array
                nums = self.gen_random_ints(size)

            currentIter = 0

            while currentIter < maxIter:
                start = default_timer()
                result = function(nums)
                end = default_timer()
                sum_time += end - start
                currentIter += 1

                if result != None:
                    is_sorted, index = self.is_ascending_sorted(result)
                    if index == 1:
                        msg = f"{result[index - 1]}, {result[index]}..."
                    elif index == len(result) - 1:
                        msg = f"...{result[index - 1]}, {result[index]}"
                    elif isinstance(index, int):
                        msg = f"...{result[index - 1]}, {result[index]}, {result[index + 1]}..."
                    else:
                        msg = ""
                    if not is_sorted:
                        raise BigOException(
                            f"{function.__name__} doesn't sort correctly.\nAt {index} index: [{msg}]"
                        )

            if (
                sum_time >= 4 and limit
            ):  # if it takes more than 4 seconds to sort an array, then break it
                is_slow = True

            sum_time /= maxIter
            times.append(sum_time)

        complexity = self._estimate(sizes, times)
        cplx = complexity._to_str()
        # estimatedTime = sum(times)

        if show_result:
            print(f"Completed {function.__name__}({array} array): {cplx}")

        # if self._is_window:
        #     toaster.show_toast(
        #         "Big-O Caculator",
        #         f"Completed {function.__name__}({array} array): {cplx}",
        #         duration=3,
        #     )

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
            cplx = self.test(function, test, show_result=False)
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
        show_result: bool = True,
    ) -> Tuple[float, Optional[List[Any]]]:
        """
        ex) runtime(bubbleSort, "random", 5000)

        Args:
            function [Callable]: a function to call |
            array: "random", "big", "sorted", "partial", "reversed", "Ksorted" ,
            "hole", "equal", "almost_equal" or your custom array |
            size [int]: How big test array should be |
            epoch [int]: How many tests to run and calculte average |
            show_result (bool): Whether to print the result by itself (default = True) |

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
                nums = self.gen_random_ints(size)
            elif array == "big":
                nums = self.gen_random_big_ints(size)
            elif array == "sorted":
                nums = self.gen_sorted_ints(size)
            elif array == "partial":
                nums = self.gen_partial_ints(size)
            elif array == "reversed":
                nums = self.gen_reversed_ints(size)
            elif array == "ksorted":
                nums = self.gen_ksorted_ints(size, size.bit_length())
            elif array == "string":
                nums = self.gen_random_strings(size)
            elif array == "hole":
                nums = self.gen_hole_ints(size)
            elif array == "equal":
                nums = self.gen_equal_ints(size)
            elif array == "almost_equal":
                nums = self.gen_almost_equal_ints(size)
            else:  # default = random array
                nums = self.gen_random_ints(size)

        if show_result:
            print(f"Running {function.__name__}(len {size} {array} array)")

        sum_time = 0.0
        result = None
        # args = None
        # if isLambda:
        #     result = function(nums, 0, len(nums) - 1)

        for _ in range(epoch):
            # args = function.__code__.co_varnames
            # args = (nums, 0, len(nums) - 1)
            start_time = default_timer()
            result = function(nums)
            end_time = default_timer()

            sum_time += end_time - start_time

            if result != None:
                is_sorted, index = self.is_ascending_sorted(result)
                if index == 1:
                    msg = f"{result[index - 1]}, {result[index]}..."
                elif index == len(result) - 1:
                    msg = f"...{result[index - 1]}, {result[index]}"
                elif isinstance(index, int):
                    msg = f"...{result[index - 1]}, {result[index]}, {result[index + 1]}..."
                else:
                    msg = ""

                if not is_sorted:
                    # Just see the result if it doesn't sort correctly
                    print(
                        f"{function.__name__} doesn't sort correctly.\nAt {index} index: [{msg}]"
                    )

        estimated_time = sum_time / epoch

        if show_result:
            print(f"Took {estimated_time:.5f}s to sort {function.__name__}({array})")

        return estimated_time, result

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
                    function1, arr, size, epoch=3, show_result=False
                )
                func1_sum += function1_time

                function2_time, _ = self.runtime(
                    function2, arr, size, epoch=3, show_result=False
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
                function1, array, size, epoch=3, show_result=False
            )
            function2_time, _ = self.runtime(
                function2, array, size, epoch=3, show_result=False
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
