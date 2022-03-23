from enum import Enum
from pathlib import Path
from typing import Iterable, Union, Tuple

import cv2
from apppath import ensure_existence
from draugr.opencv_utilities import cv2_resize, InterpolationEnum
from draugr.tqdm_utilities import progress_bar
from sorcery import assigned_names
from warg import Number

__all__ = ["ResizeMethodEnum", "resize", "resize_children"]


class ResizeMethodEnum(Enum):
    crop, scale, scale_crop = assigned_names()


def resize(image, width=None, height=None, inter=InterpolationEnum.area):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2_resize(image, dim, interpolation=inter)


def resize_children(
    src_path: Union[Path, str],
    size: Union[Tuple[Number, Number], Number],
    dst_path: Union[Path, str] = "resized",
    *,
    from_extensions: Iterable[str] = ("jpg", "png"),
    to_extension: "str" = "jpg",
    resize_method: ResizeMethodEnum = ResizeMethodEnum.scale_crop,
) -> None:
    target_size = (size, size)
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not dst_path.root:
        dst_path = src_path.parent / dst_path
    for ext in progress_bar(from_extensions):
        for c in progress_bar(src_path.rglob(f'*.{ext.rstrip("*").rstrip(".")}')):
            image = cv2.imread(str(c))
            if resize_method == resize_method.scale:
                resized = cv2_resize(image, target_size, InterpolationEnum.area)
            elif resize_method == resize_method.crop:
                center = (image.shape[0] / 2, image.shape[1] / 2)
                x = int(center[1] - target_size[0] / 2)
                y = int(center[0] - target_size[1] / 2)
                resized = image[y : y + target_size[1], x : x + target_size[0]]
            elif resize_method == resize_method.scale_crop:
                resized = resize(image, width=target_size[0])
                center = (resized.shape[0] / 2, resized.shape[1] / 2)
                x = int(center[1] - target_size[0] / 2)
                y = int(center[0] - target_size[1] / 2)
                resized = resized[y : y + target_size[1], x : x + target_size[0]]
            else:
                raise NotImplementedError

            target_folder = ensure_existence(
                dst_path.joinpath(*(c.relative_to(src_path).parent.parts))
            )
            cv2.imwrite(
                str(
                    (target_folder / c.name).with_suffix(
                        f'.{to_extension.rstrip("*").rstrip(".")}'
                    )
                ),
                resized,
            )


if __name__ == "__main__":

    def aush():
        src_path = (
            Path.home()
            / "ProjectsWin"
            / "Github"
            / "Aivclab"
            / "eyetest"
            / "images2"
            / "faces"
            / "raw"
        )
        resize_children(src_path, 512)

    aush()
