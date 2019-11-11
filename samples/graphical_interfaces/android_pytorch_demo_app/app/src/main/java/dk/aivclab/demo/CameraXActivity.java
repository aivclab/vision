package dk.aivclab.demo;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Toast;

import androidx.camera.core.CameraX;
import androidx.camera.core.UseCase;
import androidx.core.app.ActivityCompat;

import dk.aivclab.demo.usecases.classification.StatusBarUtils;

public abstract class CameraXActivity<R> extends TorchModuleActivity {
  private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
  private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    StatusBarUtils.setStatusBarOverlay(getWindow(), true);


    startBackgroundThread();

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_CODE_CAMERA_PERMISSION);
    } else {
      setupCameraXUseCases();
    }
  }

  private void setupCameraXUseCases() {
    CameraX.bindToLifecycle(this, getUseCases());
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
      if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        Toast.makeText(this,
            "You can't use image classification example without granting CAMERA permission",
            Toast.LENGTH_LONG).show();
        finish();
      } else {
        setupCameraXUseCases();
      }
    }
  }

  protected abstract UseCase[] getUseCases();
}
