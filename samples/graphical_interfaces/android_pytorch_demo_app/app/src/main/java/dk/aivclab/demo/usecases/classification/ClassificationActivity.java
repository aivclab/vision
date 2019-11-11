package dk.aivclab.demo.usecases.classification;

import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.UseCase;

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
import dk.aivclab.demo.usecases.classification.views.ResultHeaderView;
import dk.aivclab.demo.usecases.classification.views.ResultRowView;
import dk.aivclab.demo.utilities.Utils;


public class ClassificationActivity extends CameraXActivity<ClassificationActivity.AnalysisResult> {

  private boolean mAnalyzeImageErrorState;
  private ResultRowView[] mResultRowViews = new ResultRowView[Constants.TOP_K];
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private long mMovingAvgSum = 0;
  private Queue<Long> mMovingAvgQueue = new LinkedList<>();
  private long mLastAnalysisResultTime;

  static class AnalysisResult {

    private final String[] topNClassNames;
    private final float[] topNScores;
    private final long analysisDuration;
    private final long moduleForwardDuration;
    private final float accum_score;

    AnalysisResult(String[] topNClassNames,
                   float[] topNScores,
                   long moduleForwardDuration,
                   long analysisDuration,
                   float accum_score) {
      this.topNClassNames = topNClassNames;
      this.topNScores = topNScores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
      this.accum_score = accum_score;
    }
  }

  private void analyze(ImageProxy image, int rotationDegrees) {
    if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
      return;
    }

    final AnalysisResult result = analyzeImage(image, rotationDegrees);
    if (result != null) {
      mLastAnalysisResultTime = SystemClock.elapsedRealtime();
      runOnUiThread(() -> drawResult(result));
    }
  }


  protected String getModuleAssetName() {
    return Constants.MODEL_NAME;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    final ResultHeaderView headerResultRowView = findViewById(R.id.image_classification_result_header_row);
    headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
    headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);

    mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
    mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
    mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);
  }

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_classification;
  }

  final protected UseCase[] getUseCases() {
    final TextureView textureView = findViewById(R.id.image_classification_texture_view);
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    preview.setOnPreviewOutputUpdateListener(output -> {
      textureView.setSurfaceTexture(output.getSurfaceTexture());
    });

    final ImageAnalysisConfig imageAnalysisConfig = new ImageAnalysisConfig.Builder().
        setTargetResolution(new Size(Constants.INPUT_TENSOR_WIDTH, Constants.INPUT_TENSOR_HEIGHT)).
        setBackgroundExecutor(mBackgroundExecutor).
        setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE).build();

    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(mBackgroundExecutor,
        (image, rotationDegrees) -> analyze(image, rotationDegrees));

    return new UseCase[]{preview, imageAnalysis};
  }

  @WorkerThread
  @Nullable
  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      if (mModule == null) {
        final String moduleFileAbsoluteFilePath = new File(Utils.assetFilePath(this,
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
      float AccumScore = 0;
      for (int i = 0; i < Constants.TOP_K; i++) {
        final int ix = ixs[i];
        topKClassNames[i] = Constants.IMAGENET_CLASSES[ix];
        topKScores[i] = scores[ix];
        AccumScore += scores[ix];
      }
      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult(topKClassNames,
          topKScores,
          moduleForwardDuration,
          analysisDuration,
          AccumScore);
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

  protected void drawResult(AnalysisResult result) {
    mMovingAvgSum += result.moduleForwardDuration;
    mMovingAvgQueue.add(result.moduleForwardDuration);
    if (mMovingAvgQueue.size() > Constants.MOVING_AVG_PERIOD) {
      mMovingAvgSum -= mMovingAvgQueue.remove();
    }

    for (int i = 0; i < Constants.TOP_K; i++) {
      final ResultRowView rowView = mResultRowViews[i];
      rowView.nameTextView.setText(result.topNClassNames[i]);
      rowView.scoreTextView.setText(String.format(Locale.US, Constants.SCORES_FORMAT, result.topNScores[i]));
      rowView.setProgressState(false);
      rowView.scoreProgressBar.setProgress((int) ((result.topNScores[i] / result.accum_score) * 100));
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
