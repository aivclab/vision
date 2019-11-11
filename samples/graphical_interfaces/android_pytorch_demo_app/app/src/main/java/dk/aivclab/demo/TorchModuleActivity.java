package dk.aivclab.demo;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Menu;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import org.pytorch.Module;

import java.util.concurrent.Executor;

import dk.aivclab.demo.usecases.classification.Constants;
import dk.aivclab.demo.utilities.DirectExecutor;

public abstract class TorchModuleActivity extends AppCompatActivity {
  private static final int UNSET = 0;

  protected HandlerThread mBackgroundThread;
  protected Executor mBackgroundExecutor;
  protected Handler mUIHandler;

  protected TextView mFpsText;
  protected TextView mMsText;
  protected TextView mMsAvgText;
  protected Module mModule;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    setContentView(getContentViewLayoutId());

    mFpsText = findViewById(R.id.image_classification_fps_text);
    mMsText = findViewById(R.id.image_classification_ms_text);
    mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);

    mUIHandler = new Handler(getMainLooper());
  }



  @Override
  protected void onPostCreate(@Nullable Bundle savedInstanceState) {
    super.onPostCreate(savedInstanceState);
    final Toolbar toolbar = findViewById(R.id.toolbar);
    if (toolbar != null) {
      setSupportActionBar(toolbar);
    }
    startBackgroundThread();
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread("ModuleActivity");
    mBackgroundThread.start();
    mBackgroundExecutor = new DirectExecutor();
  }

  @Override
  protected void onDestroy() {
    stopBackgroundThread();
    super.onDestroy();
    if (mModule != null) {
      mModule.destroy();
    }
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundExecutor = null;
    } catch (InterruptedException e) {
      Log.e(Constants.TAG, "Error on stopping background thread", e);
    }
  }

  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    getMenuInflater().inflate(R.menu.menu_model, menu);
    menu.findItem(R.id.action_info).setVisible(getInfoViewCode() != UNSET);
    return true;
  }

  protected int getInfoViewCode() {
    return UNSET;
  }

  protected abstract int getContentViewLayoutId();
}
