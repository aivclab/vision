package dk.aivclab.demo.usecases.classification;

import java.util.Arrays;

class Selection {

  static int[] topK(float[] a, final int num) {
    float[] values = new float[num];
    Arrays.fill(values, -Float.MAX_VALUE);
    int[] indices = new int[num];
    Arrays.fill(indices, -1);

    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < num; j++) {
        if (a[i] > values[j]) {
          for (int k = num - 1; k >= j + 1; k--) {
            values[k] = values[k - 1];
            indices[k] = indices[k - 1];
          }
          values[j] = a[i];
          indices[j] = i;
          break;
        }
      }
    }
    return indices;
  }
}
