package dk.aivclab.demo.usecases.segmentation;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.UseCase;
import androidx.camera.view.PreviewView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Objects;
import java.util.Queue;

import dk.aivclab.demo.CameraXActivity;
import dk.aivclab.demo.R;
import dk.aivclab.demo.usecases.segmentation.utilities.SegmentationDrawing;
import dk.aivclab.demo.utilities.FileUtilities;

import static dk.aivclab.demo.usecases.segmentation.Constants.colors;


public class SegmentationActivity extends CameraXActivity {

  private boolean mAnalyzeImageErrorState;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private long mMovingAvgSum = 0;
  private Queue<Long> mMovingAvgQueue = new LinkedList<>();
  private long mLastAnalysisResultTime;
  PreviewView textureView;
  private ImageView segmentationOverlayView;
  Bitmap unscaled_result;

  static class AnalysisResult {

    private final long analysisDuration;
    private final long moduleForwardDuration;
    private final float[] output;

    AnalysisResult(long moduleForwardDuration, long analysisDuration, float[] output) {
      this.output = output;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
    }
  }

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_segmentation;
  }

  @Override
  protected UseCase[] getUseCases() {

    textureView = findViewById(R.id.activity_image_preview_view);
    segmentationOverlayView = findViewById(R.id.segmentation_overlay_image_view);

    unscaled_result = Bitmap.createBitmap(Constants.INPUT_TENSOR_WIDTH, // width
        Constants.INPUT_TENSOR_HEIGHT, // height
        Bitmap.Config.ARGB_8888);

    /*
    final ImageAnalysisConfig imageAnalysisConfig = new ImageAnalysisConfig.Builder().
        setTargetResolution(new Size(224, 224)).
        setBackgroundExecutor(mBackgroundExecutor).
        setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE).build();

    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(mBackgroundExecutor, (image, rotationDegrees) -> {

      if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
        return;
      }

      final AnalysisResult result = analyzeImage(image, rotationDegrees);
      if (result != null) {
        mLastAnalysisResultTime = SystemClock.elapsedRealtime();
        runOnUiThread(() -> drawResult(result));
      }
    });

    return new UseCase[]{imageAnalysis};
         */
    return new UseCase[]{};
  }

  @WorkerThread
  @Nullable
  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      if (mModule == null) {
        final String moduleFileAbsoluteFilePath = new File(FileUtilities.assetFilePath(this,
            getModuleAssetName())).getAbsolutePath();
        mModule = Module.load(moduleFileAbsoluteFilePath);

        mInputTensorBuffer = Tensor.allocateFloatBuffer(3
                                                        * Constants.INPUT_TENSOR_WIDTH
                                                        * Constants.INPUT_TENSOR_HEIGHT);
        mInputTensor = Tensor.fromBlob(mInputTensorBuffer,
            new long[]{1, 3, Constants.INPUT_TENSOR_HEIGHT, Constants.INPUT_TENSOR_WIDTH
            });
      }

      final long startTime = SystemClock.elapsedRealtime();
      TensorImageUtils.imageYUV420CenterCropToFloatBuffer(Objects.requireNonNull(image.getImage()),
          rotationDegrees,
          Constants.INPUT_TENSOR_WIDTH,
          Constants.INPUT_TENSOR_HEIGHT,
          TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
          TensorImageUtils.TORCHVISION_NORM_STD_RGB,
          mInputTensorBuffer,
          0);

      final long moduleForwardStartTime = SystemClock.elapsedRealtime();
      final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
      final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

      final float[] scores = outputTensor.getDataAsFloatArray();

      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult( moduleForwardDuration, analysisDuration,scores);
    } catch (Exception e) {
      Log.e(Constants.TAG, "Error during image analysis", e);
      mAnalyzeImageErrorState = true;
      runOnUiThread(() -> {
        if (!isFinishing()) {
          SegmentationActivity.this.finish();
        }
      });

    }
    return null;
  }

  protected String getModuleAssetName() {
    return Constants.MODEL_NAME;
  }

  protected void drawResult(AnalysisResult result) {
    SegmentationDrawing.segmentResultToBitmap(result.output, colors, unscaled_result);
    segmentationOverlayView.setImageBitmap(SegmentationDrawing.resizeBitmap(unscaled_result,
        textureView.getWidth(),
        textureView.getHeight()));

    mMovingAvgSum += result.moduleForwardDuration;
    mMovingAvgQueue.add(result.moduleForwardDuration);
    if (mMovingAvgQueue.size() > Constants.MOVING_AVG_PERIOD) {
      mMovingAvgSum -= mMovingAvgQueue.remove();
    }


    mMsText.setText(String.format(Locale.US, Constants.FORMAT_MS, result.moduleForwardDuration));
    if (mMsText.getVisibility() != View.VISIBLE) {
      mMsText.setVisibility(View.VISIBLE);
    }
    mFpsText.setText(String.format(Locale.US, Constants.FORMAT_FPS, (1000.f / result.analysisDuration)));
    if (mFpsText.getVisibility() != View.VISIBLE) {
      mFpsText.setVisibility(View.VISIBLE);
    }

    if (mMovingAvgQueue.size() == Constants.MOVING_AVG_PERIOD) {
      float avgMs = (float) mMovingAvgSum / Constants.MOVING_AVG_PERIOD;
      mMsAvgText.setText(String.format(Locale.US, Constants.FORMAT_AVG_MS, avgMs));
      if (mMsAvgText.getVisibility() != View.VISIBLE) {
        mMsAvgText.setVisibility(View.VISIBLE);
      }
    }
  }

}
