import { bigO } from './bigO';
import { bubbleSort, countSort, insertSort, quickSort } from './sorting';

test('basic', () => {
  const calc = new bigO();
  expect(calc.coef).toEqual(0.0);
  expect(calc.rms).toEqual(0.0);

  expect(calc.cplx2str(2)).toEqual("O(n)")
});

test('fittingCurve', () => {
  const calc = new bigO();
  const ON = calc.fittingCurve(2)
  const ON2 = calc.fittingCurve(5)
  expect(ON(3)).toEqual(3)
  expect(ON2(3)).toEqual(9)
});

// test('genArray', () => {
//   const calc = new bigO();
//   console.log(calc.genRandomArray(10))
//   console.log(calc.genReversedArray(10))
//   console.log(calc.genSortedArray(10))
//   console.log(calc.genPartialArray(10))
// });

test('test', () => {
  const calc = new bigO();
  expect(calc.test(bubbleSort, "sorted")).toEqual("O(n)")
  expect(calc.test(countSort, "sorted")).toEqual("O(n)")
  expect(calc.test(insertSort, "sorted")).toEqual("O(n)")
  expect(calc.test(insertSort, "random")).toEqual("O(n^2)")
  // expect(calc.test(quickSort, "random")).toEqual("O(nlog(n))")
  console.log(calc.test(quickSort, "sorted"))
  console.log(calc.test(quickSort, "random"))
  // console.log(calc.test(insertSort, "sorted"))
});


// test('test(countSort)', () => {
//   const calc = new bigO();
//   console.log(calc.test(countSort, "random"))
//   console.log(calc.test(countSort, "sorted"))
//   console.log(calc.test(countSort, "reversed"))
//   console.log(calc.test(countSort, "ksorted"))
//   console.log(calc.test(countSort, "partial"))
// });

test('estimate', () => {
  const calc = new bigO();
  const n = [10, 100, 1000, 10000, 100000];
  const times = [3.0399999999985995e-06, 1.115999999985462e-05, 0.00013217999999994844, 0.0008946599999999805, 0.00874170000000003]

  const result = calc.estimate(n, times).to_str();
  console.log(result);
})
