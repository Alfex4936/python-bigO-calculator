from bigO import bigO


def test_cplx():
    tester = bigO.bigO()
    assert tester.complexity2str(0) == "f(n)"
    assert tester.complexity2str(2) == "O(N)"
    assert tester.complexity2str(3) == "O(N^2)"


def test_fitting():
    tester = bigO.bigO()
    assert tester.fittingCurve(1)(100) == 1.0
    assert tester.fittingCurve(3)(2) == 4  # O(n^2)
    assert tester.fittingCurve(4)(2) == 8  # O(n^3)

