from bigO import BigO


def is_sorted(func):
    def wrapper(array, *args):
        lib = BigO()

        result = func(array, *args)

        _sorted, index = lib.is_ascending_sorted(result)
        if index == 1:
            msg = f"{result[index - 1]}, {result[index]}..."
        elif index == len(result) - 1:
            msg = f"...{result[index - 1]}, {result[index]}"
        elif isinstance(index, int):
            msg = f"...{result[index - 1]}, {result[index]}, {result[index + 1]}..."
        else:
            msg = ""

        if not _sorted:
            # Just see the result if it doesn't sort correctly
            print(f"{func.__name__} doesn't sort correctly.\nAt {index} index: [{msg}]")
        else:
            print(f"{func.__name__} sorts correctly.")

    return wrapper
