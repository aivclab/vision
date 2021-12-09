import numpy
import os
import pickle

import numpy

from draugr.visualisation import denormalise_minusoneone, matplotlib_bounding_box
from matplotlib import animation, pyplot
from pathlib import Path

from samples.classification.ram.ram_params import get_ram_config


def main(plot_dir: Path, epoch=None):
    """

    Args:
      plot_dir:
      epoch:
    """
    if epoch is None:
        list_of_files = list(plot_dir.rglob("*.p"))
        lastest_model_path = max(list_of_files, key=os.path.getctime)
        epoch = int(str(lastest_model_path).split("_")[-1].split(".")[0])

    print(f"Load epoch model {epoch}")

    glimpses = pickle.load(open(str(plot_dir / f"g_{epoch}.p"), "rb"))
    locations = pickle.load(open(str(plot_dir / f"l_{epoch}.p"), "rb"))

    glimpses = numpy.concatenate(glimpses)

    size = int(str(plot_dir).split("_")[2][0])
    num_anims = len(locations)
    num_cols = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    coords = [
        denormalise_minusoneone(img_shape, l) for l in locations
    ]  # denormalize coordinates

    fig, axs = pyplot.subplots(nrows=1, ncols=num_cols)

    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def update_data(i):
        """

        Args:
          i:
        """
        color = "r"
        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = matplotlib_bounding_box(c[0], c[1], size, color)
            ax.add_patch(rect)

    anim = animation.FuncAnimation(
        fig, update_data, frames=num_anims, interval=500, repeat=True
    )
    anim.save(
        str(plot_dir / f"epoch_{epoch}.mp4"),
        extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"],
    )  # save as mp4
    pyplot.show()


if __name__ == "__main__":
    config = get_ram_config()
    plot_dir = list(config.plot_dir.iterdir())[-1]
    print(plot_dir)
    main(plot_dir)
