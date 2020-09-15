from bigO import bigO


def test_cplx():
    tester = bigO.bigO()
    assert tester.complexity2str(0) == "f(n)"
    assert tester.complexity2str(2) == "O(n)"
    assert tester.complexity2str(5) == "O(n^2)"


def test_fitting():
    tester = bigO.bigO()
    assert tester.fittingCurve(1)(100) == 1.0
    assert tester.fittingCurve(5)(2) == 4  # O(n^2)
    assert tester.fittingCurve(6)(2) == 8  # O(n^3)

