const { performance } = require('perf_hooks');

type bigOFunc = (n: number) => number;

type sortingFunc = (arr: number[], ...args: number[]) => number[];


export class bigO {
  coef?: number = 0.0;
  rms?: number = 0.0;
  cplx?: number = 0;
  O1?: number = 1;
  ON?: number = 2;
  OLogN?: number = 3;
  ONLogN?: number = 4;
  ON2?: number = 5;
  ON3?: number = 6;
  OLambda?: number = 7;
  fitCurves?: number[] = [this.O1, this.ON, this.OLogN, this.ONLogN, this.ON2, this.ON3];


  to_str(): string {
    return this.cplx2str(this.cplx);
  }

  cplx2str(cplx: number): string {
    switch (cplx) {
      case this.ON:
        return "O(n)";
      case this.ON2:
        return "O(n^2)";
      case this.ON3:
        return "O(n^3)";
      case this.OLogN:
        return "O(log(n))";
      case this.ONLogN:
        return "O(nlog(n))";
      case this.O1:
        return "O(1)";
      default:
        return "f(n)"
    }
  }

  cplx2int(cplx: string): number {
    switch (cplx) {
      case "O(n)":
        return this.ON;
      case "O(n^2)":
        return this.ON2;
      case "O(n^3)":
        return this.ON3;
      case "O(log(n))":
        return this.OLogN;
      case "O(nlog(n))":
        return this.ONLogN;
      case "O(1)":
        return this.O1;
      default:
        return this.ON
    }
  }

  fittingCurve(cplx: number): bigOFunc {
    switch (cplx) {
      case this.ON:
        return function (n: number) { return n };
      case this.ON2:
        return function (n: number) { return n * n };
      case this.ON3:
        return function (n: number) { return n * n * n };
      case this.OLogN:
        return function (n: number) { return Math.log2(n) };
      case this.ONLogN:
        return function (n: number) { return n * Math.log2(n) };
      case this.O1:
        return function (_: number) { return 1.0 };
      default:
        return function (_: number) { return 1.0 };
    }
  }



  genRandomArray(size: number = 10) {
    // const arr = Array.from(Range(size), ({ email }) => email)
    return Array.from({ length: size }, () => Math.floor(Math.random() * size * 2) - size);
  }

  genReversedArray(size: number = 10) {
    return Array.from(Array(size).keys()).reverse();
  }

  genSortedArray(size: number = 10) {
    return Array.from(Array(size).keys());
  }

  genPartialArray(size: number = 10) {
    let result = this.genRandomArray(size);
    const sorted_array = this.genSortedArray(size);

    for (let i = ~~(size / 4); i < ~~(size / 2); i++) {
      result[i] = sorted_array[i];
    }
    return result;
  }

  genKsortedArray(size: number = 10, k: number = undefined) {
    if (!k) {
      k = ((size).toString(2)).length;  // bit_length()
    }

    if (size < k) {
      throw new Error("K must be smaller than the size.");
    }

    if (k == 0) {
      return this.genSortedArray(size);
    } else if (size == k) {
      return this.genReversedArray(size);
    }

    let array: number[] = [];
    for (let i = ~~(-size / 2); i < ~~(size / 2); i++) {
      array.push(i);
    }

    let right: number = Math.floor(Math.random() * (k - 1));
    while (right >= size - k) {
      right--;
    }

    this._reverseRange(array, 0, k + 1);
    this._reverseRange(array, size - right, size);

    return array;
  }

  _reverseRange(array: number[], a: number, b: number): number[] {
    let i = a;
    let j = b - 1;
    while (i < j) {
      [array[i], array[j]] = [array[j], array[i]]
      i++;
      j--;
    }
    return array;
  }

  isAscendingSorted(array: number[]) {
    for (let i = 0; i < array.length - 1; i++) {
      if (array[i] > array[i + 1]) {
        return [false, i + 1];
      }
    }
    return [true, null]
  }

  minimalLeastSq(array: number[], times: number[], func: bigOFunc): bigO {
    let sigmaGnSquared: number = 0.0;
    let sigmaTime: number = 0.0
    let sigmaTimeGn: number = 0.0

    let floatN: number = parseFloat((array.length).toFixed(1));

    for (let i = 0; i < array.length; i++) {
      const gnI: number = func(array[i]);
      sigmaGnSquared += gnI * gnI;
      sigmaTime += times[i];
      sigmaTimeGn += times[i] * gnI;
    }

    const result: bigO = new bigO();
    result.cplx = this.OLambda;
    result.coef = sigmaTimeGn / sigmaGnSquared;

    let rms: number = 0.0;
    for (let i = 0; i < array.length; i++) {
      const fit = result.coef * func(array[i]);
      rms += Math.pow(times[i] - fit, 2);
    }

    const mean: number = sigmaTime / floatN;
    result.rms = Math.sqrt(rms / floatN) / mean;

    return result;
  }

  estimate(n: number[], times: number[]) {
    if (n.length != times.length) {
      throw new Error(`Length mismatch between N:${n.length} and TIMES:${times.length}.`);
    }
    if (n.length < 2) {
      throw new Error("Need at least 2 runs.");
    }
    times.sort();

    let bestFit: bigO = this.minimalLeastSq(n, times, this.fittingCurve(this.O1));
    bestFit.cplx = this.O1;

    for (const fit of this.fitCurves) {
      const currentFit = this.minimalLeastSq(n, times, this.fittingCurve(fit));
      if (currentFit.rms < bestFit.rms) {
        bestFit = currentFit;
        bestFit.cplx = fit;
      }
    }

    return bestFit;
  }

  // Main
  test(func: sortingFunc, array: string, limit: boolean = true, prtResult: boolean = true): string {
    let sizes: number[] = [10, 100, 1000, 10000, 100000];
    const maxIter: number = 5;
    let times: number[] = [];
    let isSlow: boolean = false;

    if (prtResult) {
      console.log(`Running ${func.name}(${array} array)...`)
    }

    for (const size of sizes) {
      if (isSlow) {
        sizes = sizes.slice(0, times.length - 1);
        break;
      }

      let timeTaken: number = 0.0
      let nums: number[] = [];

      array = array.toLocaleLowerCase();
      switch (array) {
        case "random":
          nums = this.genRandomArray(size);
          break;
        case "sorted":
          nums = this.genSortedArray(size);
          break;
        case "partial":
          nums = this.genPartialArray(size);
          break;
        case "ksorted":
          nums = this.genKsortedArray(size);
          break;
        case "reversed":
          nums = this.genReversedArray(size);
          break;
        default:
          nums = this.genRandomArray(size);
          break;
      }

      let currentIter: number = 0;

      while (currentIter < maxIter) {
        const start: number = performance.now();
        const result = func(nums);
        const end: number = performance.now();

        timeTaken += (end - start) * (0.001);
        currentIter++;

        // TODO if result null
      }

      if (timeTaken >= 4.0 && limit) {
        isSlow = true;
      }

      timeTaken /= maxIter;
      times.push(timeTaken);
    }
    console.log(`RUNTIME ${times}`)

    const complexity = this.estimate(sizes, times).to_str();
    if (prtResult) {
      console.log(`Completed ${func.name}(${array} array): ${complexity}`)
    }

    return complexity;
  }
}
