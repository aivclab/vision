package dk.aivclab.demo.usecases.classification;

import android.content.Context;
import android.hardware.display.DisplayManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
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
import dk.aivclab.demo.usecases.classification.views.ResultRowView;
import dk.aivclab.demo.utilities.DisplayHelperFunctions;
import dk.aivclab.demo.utilities.FileUtilities;
import dk.aivclab.demo.usecases.classification.utilities.Selection;


public class ClassificationActivity extends CameraXActivity {

  private boolean mAnalyzeImageErrorState;
  private ResultRowView[] mResultRowViews = new ResultRowView[Constants.TOP_K];
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private long mMovingAvgSum = 0;
  private Queue<Long> mMovingAvgQueue = new LinkedList<>();
  private long mLastAnalysisResultTime;
  DisplayManager displayManager;
  int rotation;

  static class AnalysisResult {

    private final String[] topNClassNames;
    private final float[] topNScores;
    private final long analysisDuration;
    private final long moduleForwardDuration;

    AnalysisResult(String[] topNClassNames,
                   float[] topNScores,
                   long moduleForwardDuration,
                   long analysisDuration) {
      this.topNClassNames = topNClassNames;
      this.topNScores = topNScores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
    }
  }

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_classification;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
/*
    final ResultHeaderView headerResultRowView = findViewById(R.id.image_classification_result_header_row);
    headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
    headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);
*/
    mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
    mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
    mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);
  }


  final protected UseCase[] getUseCases() {

    displayManager = (DisplayManager) getSystemService(Context.DISPLAY_SERVICE);
    PreviewView textureView = findViewById(R.id.activity_image_preview_view);
    int screenAspectRatio = DisplayHelperFunctions.getAspectRatio(displayManager.getDisplays()[0]);

    rotation = displayManager.getDisplays()[0].getRotation();
    final Preview preview = new Preview.Builder()
        .setTargetAspectRatio(screenAspectRatio)     // We request aspect ratio but no resolution
        .setTargetRotation(rotation)        // Set initial target rotation
        .build();

    preview.setSurfaceProvider(textureView.getPreviewSurfaceProvider());


    final ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().
        setTargetRotation(rotation).
        setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).
        setTargetResolution(new Size(Constants.INPUT_TENSOR_WIDTH, Constants.INPUT_TENSOR_HEIGHT)).
        setBackgroundExecutor(mBackgroundExecutor).build();

    imageAnalysis.setAnalyzer(mBackgroundExecutor, image -> analyze(image));

    return new UseCase[]{preview, imageAnalysis};
  }

  private void analyze(ImageProxy image) {

    if (!(SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500)) {



    final AnalysisResult result = analyzeImage(image, rotation);
    if (result != null) {
      mLastAnalysisResultTime = SystemClock.elapsedRealtime();
      runOnUiThread(() -> drawResult(result));
    }
    }

    image.close();
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
      final int[] ixs = Selection.topK(scores, Constants.TOP_K);

      final String[] topKClassNames = new String[Constants.TOP_K];
      final float[] topKScores = new float[Constants.TOP_K];
      for (int i = 0; i < Constants.TOP_K; i++) {
        final int ix = ixs[i];
        topKClassNames[i] = Constants.CATEGORIES[ix];
        topKScores[i] = scores[ix];
      }
      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult(topKClassNames, topKScores, moduleForwardDuration, analysisDuration);
    } catch (Exception e) {
      Log.e(Constants.TAG, "Error during image analysis", e);
      mAnalyzeImageErrorState = true;
      runOnUiThread(() -> {
        if (!isFinishing()) {
          ClassificationActivity.this.finish();
        }
      });
      return null;
    }
  }

  protected String getModuleAssetName() {
    return Constants.MODEL_NAME;
  }

  protected void drawResult(AnalysisResult result) {
    mMovingAvgSum += result.moduleForwardDuration;
    mMovingAvgQueue.add(result.moduleForwardDuration);
    if (mMovingAvgQueue.size() > Constants.MOVING_AVG_PERIOD) {
      mMovingAvgSum -= mMovingAvgQueue.remove();
    }

    float[] soft_maxed_res = Selection.SoftMax(result.topNScores);

    for (int i = 0; i < Constants.TOP_K; i++) {
      final ResultRowView rowView = mResultRowViews[i];
      rowView.nameTextView.setText(result.topNClassNames[i]);
      rowView.scoreTextView.setText(String.format(Locale.US, Constants.SCORES_FORMAT, soft_maxed_res[i]));
      rowView.setProgressState(false);

      rowView.scoreProgressBar.setProgress((int) (soft_maxed_res[i] * 100));
    }

    if (mMsText != null) {
      mMsText.setText(String.format(Locale.US, Constants.FORMAT_MS, result.moduleForwardDuration));
      if (mMsText.getVisibility() != View.VISIBLE) {
        mMsText.setVisibility(View.VISIBLE);
      }
    }
    if (mFpsText != null) {
      mFpsText.setText(String.format(Locale.US, Constants.FORMAT_FPS, (1000.f / result.analysisDuration)));
      if (mFpsText.getVisibility() != View.VISIBLE) {
        mFpsText.setVisibility(View.VISIBLE);
      }
    }
    if (mMsAvgText != null) {
      if (mMovingAvgQueue.size() == Constants.MOVING_AVG_PERIOD) {
        float avgMs = (float) mMovingAvgSum / Constants.MOVING_AVG_PERIOD;
        mMsAvgText.setText(String.format(Locale.US, Constants.FORMAT_AVG_MS, avgMs));
        if (mMsAvgText.getVisibility() != View.VISIBLE) {
          mMsAvgText.setVisibility(View.VISIBLE);
        }
      }
    }
  }

}
