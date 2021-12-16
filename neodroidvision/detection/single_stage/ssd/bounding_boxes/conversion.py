import torch

__all__ = [
    "locations_to_boxes",
    "convert_boxes_to_locations",
    "center_to_corner_form",
    "corner_form_to_center_form",
]


def locations_to_boxes(
    *,
    locations: torch.Tensor,
    priors: torch.Tensor,
    center_variance: float,
    size_variance: float
) -> torch.tensor:
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
    $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
    $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
    locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as
    well.
    priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
    center_variance: a float used to change the scale of center.
    size_variance: a float used to change of scale of size.
    Returns:
    boxes:  priors: [[center_x, center_y, w, h]]. All the values
        are relative to the image size."""
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat(
        [
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:],
        ],
        dim=locations.dim() - 1,
    )


def convert_boxes_to_locations(
    *,
    center_form_boxes: torch.Tensor,
    center_form_priors: torch.Tensor,
    center_variance: float,
    size_variance: float
) -> torch.tensor:
    """

    :param center_form_boxes:
    :param center_form_priors:
    :param center_variance:
    :param size_variance:
    :return:
    """
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat(
        [
            (center_form_boxes[..., :2] - center_form_priors[..., :2])
            / center_form_priors[..., 2:]
            / center_variance,
            torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:])
            / size_variance,
        ],
        dim=center_form_boxes.dim() - 1,
    )


def center_to_corner_form(locations: torch.Tensor) -> torch.Tensor:
    """

    Args:
      locations:

    Returns:

    """
    return torch.cat(
        [
            locations[..., :2] - locations[..., 2:] / 2,
            locations[..., :2] + locations[..., 2:] / 2,
        ],
        locations.dim() - 1,
    )


def corner_form_to_center_form(boxes: torch.Tensor) -> torch.Tensor:
    """

    Args:
      boxes:

    Returns:

    """
    return torch.cat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2, boxes[..., 2:] - boxes[..., :2]],
        boxes.dim() - 1,
    )


if __name__ == "__main__":
    pass
