export function bubbleSort(array: number[]): number[] {
  let isSorted = false;
  let n = array.length - 1;
  let count = 0;

  while (!isSorted) {
    isSorted = true;
    for (let j = 0; j < n - count; j++) {
      if (array[j] > array[j + 1]) {
        [array[j], array[j + 1]] = [array[j + 1], array[j]];
        isSorted = false;
      }
    }
    count++;
  }
  return array;
}

export function countSort(arr: number[]): number[] {
  const min = Math.min(...arr);
  const max = Math.max(...arr) - min;
  const occurrences = [];

  for (let i = 0; i < max + 1; i++) {
    occurrences[i] = 0;
  }

  for (let j = 0; j < arr.length; j++) {
    occurrences[arr[j] - min]++;
  }

  let index = 0;
  for (let j = 0; j < occurrences.length; j++) {
    while (occurrences[j] > 0) {
      arr[index] = j + min;
      index++;
      occurrences[j]--;
    }
  }

  return arr;
};

export function insertSort(array: number[]): number[] {
  for (let i = 1; i < array.length; i++) {
    let j = i;
    while (j > 0 && array[j] < array[j - 1]) {
      const temp = array[j];
      array[j] = array[j - 1];
      array[j - 1] = temp;
      // [array[j], array[j - 1]] = [array[j - 1], array[j]];
      j--;
    }
  }
  return array;
}

/**
 * Sorts an array using quick sort
 */
export function quickSort(array: number[]): number[] {
  array = array.slice();
  partition(array, 0, array.length);
  return array;
}

/**
 * Partitions the [start, before) index of the array
 */
function partition(array: number[], start: number, before: number): void {
  const length = before - start;

  /** Terminate the recursion */
  if (length <= 1) return;

  /** Randomly select a pivot and move it to the head of the array */
  const pivotIndex = start + Math.floor(Math.random() * length);
  [array[start], array[pivotIndex]] = [array[pivotIndex], array[start]];

  /** The first element is our pivot */
  const pivot = array[start];
  let pivotRank = start;

  /** Loop through all the elements, partitioning around the pivot */
  for (let index = start + 1; index < before; index++) {
    if (array[index] < pivot) {
      pivotRank++;
      [array[index], array[pivotRank]] = [array[pivotRank], array[index]];
    }
  }

  /** Finally put the pivot at its rightfull place */
  if (pivotRank !== start) {
    [array[pivotRank], array[start]] = [array[start], array[pivotRank]];
  }

  /** Partition all the elements less than the pivot */
  partition(array, start, pivotRank);

  /** Partition all the elements more than the pivot */
  partition(array, pivotRank + 1, before);
}
