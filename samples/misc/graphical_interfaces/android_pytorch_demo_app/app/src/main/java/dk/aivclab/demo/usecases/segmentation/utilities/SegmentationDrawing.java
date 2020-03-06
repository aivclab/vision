package dk.aivclab.demo.usecases.segmentation.utilities;

import android.graphics.Bitmap;
import android.graphics.Canvas;

public class SegmentationDrawing {

  public static void segmentResultToBitmap(float[] segmentedImage, int[] classColors, Bitmap targetBitmap) {
    int width = targetBitmap.getWidth();
    int height = targetBitmap.getHeight();
    int[] pixels = new int[width * height];

    for (int i1 = 0; i1 < segmentedImage.length; i1++) {
      pixels[i1] = classColors[(int) segmentedImage[i1]];
    }

    targetBitmap.setPixels(pixels, 0, width, 0, 0, width, height);
  }

  public static Bitmap resizeBitmap(Bitmap bitmap, int width, int height) {
    Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

    if (bitmap.getWidth() != bitmap.getHeight()) {
      throw new Error("Mask expected to be square but got something else");
    }
    // Expects image to be square shape
    Bitmap scaledBitmap;
    if (height > width) {
      scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, width, true);
    } else {
      scaledBitmap = Bitmap.createScaledBitmap(bitmap, height, height, true);
    }

    float pX = (width - scaledBitmap.getWidth()) / 2;
    float pY = (height - scaledBitmap.getHeight()) / 2;
    Canvas can = new Canvas(result);
    // can.drawARGB(0x00, 0xff, 0xff, 0xff)
    can.drawBitmap(scaledBitmap, pX, pY, null);
    // scaledBitmap.recycle()

    return result;
  }
}
