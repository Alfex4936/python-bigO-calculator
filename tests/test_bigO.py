from bigO import BigO


def test_cplx():
    lib = BigO()
    assert lib._complexity2str(0) == "f(n)"
    assert lib._complexity2str(2) == "O(n)"
    assert lib._complexity2str(5) == "O(n^2)"


def test_fitting():
    lib = BigO()
    assert lib._fittingCurve(1)(100) == 1.0
    assert lib._fittingCurve(5)(2) == 4  # O(n^2)
    assert lib._fittingCurve(6)(2) == 8  # O(n^3)


def test_estimate():
    lib = BigO()
    n = [10, 100, 1000, 10000]
    times = [
        3.959999999736397e-06,
        0.00015611999999993743,
        0.015288159999999884,
        1.62602698,
    ]
    result = lib._estimate(n, times)
    assert result._to_str() == "O(n^2)", "Estimation is wrong."
