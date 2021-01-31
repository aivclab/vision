from pathlib import Path

import cv2
import torch
from apppath import ensure_existence

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.multitask import SkipHourglassFission
from neodroidvision.utilities import OutputActivationModule
from torchvision import transforms
from tqdm import tqdm

from draugr import sprint
from draugr.opencv_utilities import frame_generator
from draugr.torch_utilities import (
  TorchDeviceSession,
  TorchEvalSession,
  global_torch_device,
  )
from draugr.random_utilities import seed_stack


@torch.no_grad()
def export_detection_model(model_export_path: Path = ensure_existence(PROJECT_APP_PATH.user_data / 'penn_fudan_segmentation') / "seg_skip_fis", SEED=87539842) -> None:
  """

:param model_export_path:
:type model_export_path:
:return:
:rtype:
"""

  model = OutputActivationModule(
      SkipHourglassFission(input_channels=3, output_heads=(1,), encoding_depth=1)
      )

  with TorchDeviceSession(
      device=global_torch_device('cpu'), model=model
      ):
    with TorchEvalSession(model):

      seed_stack(SEED)

      # standard PyTorch mean-std input image normalization
      transform = transforms.Compose(
          [
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
              ]
          )

      frame_g = frame_generator(cv2.VideoCapture(0))

      for image in tqdm(frame_g):
        example_input = (
            transform(image).unsqueeze(0).to(global_torch_device()),
            )

        try:
          traced_script_module = torch.jit.trace(
              model,
              example_input,
              # strict=strict_jit,
              check_inputs=(
                  transform(next(frame_g))
                    .unsqueeze(0)
                    .to(global_torch_device()),
                  transform(next(frame_g))
                    .unsqueeze(0)
                    .to(global_torch_device()),
                  ),
              )
          exp_path = model_export_path.with_suffix(".traced")
          traced_script_module.save(str(exp_path))
          print(
              f"Traced Ops used {torch.jit.export_opnames(traced_script_module)}"
              )
          sprint(
              f"Successfully exported JIT Traced model at {exp_path}",
              color="green",
              )
        except Exception as e_i:
          sprint(f"Torch JIT Trace export does not work!, {e_i}", color="red")

        break


if __name__ == "__main__":
  export_detection_model()
