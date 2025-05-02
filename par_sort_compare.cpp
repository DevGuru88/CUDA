#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

vector<int> deepCopy(const vector<int> &source) {
    return vector<int>(source);
}

void BubbleSort(vector<int> &array) {
    int n = array.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (array[j] > array[j + 1]) {
                swap(array[j], array[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(vector<int> &arr) {
    int N = arr.size();
    for (int i = 0; i < N - 1; i++) {
#pragma omp parallel for
        for (int j = 0; j < N - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void merge(vector<int> &arr, int start, int mid, int end) {
    vector<int> L(arr.begin() + start, arr.begin() + mid + 1);
    vector<int> R(arr.begin() + mid + 1, arr.begin() + end + 1);
    int i = 0, j = 0, k = start;
    while (i < L.size() && j < R.size()) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < L.size()) arr[k++] = L[i++];
    while (j < R.size()) arr[k++] = R[j++];
}

void MergeSort(vector<int> &arr, int start, int end) {
    if (start < end) {
        int mid = (start + end) / 2;
        MergeSort(arr, start, mid);
        MergeSort(arr, mid + 1, end);
        merge(arr, start, mid, end);
    }
}

void MergeSortParallel(vector<int> &arr, int start, int end) {
    if (start < end) {
        int mid = (start + end) / 2;
#pragma omp parallel sections
        {
#pragma omp section
            MergeSortParallel(arr, start, mid);
#pragma omp section
            MergeSortParallel(arr, mid + 1, end);
        }
        merge(arr, start, mid, end);
    }
}

int main() {
    vector<int> array;
    const int SIZE = 50000;
    for (int i = 0; i < SIZE; i++) {
        array.push_back(rand() % 100000);
    }

    vector<int> arr1 = deepCopy(array);
    vector<int> arr2 = deepCopy(array);
    vector<int> arr3 = deepCopy(array);
    vector<int> arr4 = deepCopy(array);

    double start, end;

    start = omp_get_wtime();
    BubbleSort(arr1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << end - start << " seconds\n";

    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds\n";

    start = omp_get_wtime();
    MergeSort(arr3, 0, arr3.size() - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end - start << " seconds\n";

    start = omp_get_wtime();
    MergeSortParallel(arr4, 0, arr4.size() - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << " seconds\n";

    return 0;
}
