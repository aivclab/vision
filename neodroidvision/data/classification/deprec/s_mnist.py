from pathlib import Path
from typing import Tuple, Any

import torch
from draugr.numpy_utilities import SplitEnum, SplitIndexer
from draugr.torch_utilities import SupervisedDataset
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataset2(SupervisedDataset):
    """ """

    @property
    def response_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return (len(self.categories),)

    @property
    def predictor_shape(self) -> Tuple[int, ...]:
        """

        :return:
        :rtype:"""
        return self._resize_shape

    def __init__(
        self,
        dataset_path: Path,
        split: SplitEnum = SplitEnum.training,
        validation: float = 0.3,
        resize_s: int = 28,
        seed: int = 42,
        download: bool = True,
    ):
        """
        :param dataset_path: dataset directory
        :param split: train, valid, test"""
        super().__init__()

        if not download:
            assert dataset_path.exists(), f"root: {dataset_path} not found."

        self._resize_shape = (1, resize_s, resize_s)

        train_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(resize_s),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        val_trans = transforms.Compose(
            [
                transforms.Resize(resize_s),
                # transforms.CenterCrop(resize_s),
                transforms.ToTensor(),
            ]
        )

        if split == SplitEnum.training:
            mnist_data = MNIST(
                str(dataset_path), train=True, download=download, transform=train_trans
            )
        elif split == SplitEnum.validation:
            mnist_data = MNIST(
                str(dataset_path), train=True, download=download, transform=val_trans
            )
        else:
            mnist_data = MNIST(
                str(dataset_path), train=False, download=download, transform=val_trans
            )

        if split != SplitEnum.testing:
            torch.manual_seed(seed)
            train_ind, val_ind, test_ind = (
                SplitIndexer(len(mnist_data), validation=validation, testing=0.0)
                .shuffled_indices()
                .values()
            )
            if split == SplitEnum.validation:
                self.mnist_data_split = Subset(mnist_data, val_ind)
            else:
                self.mnist_data_split = Subset(mnist_data, train_ind)
        else:
            self.mnist_data_split = mnist_data

        self.categories = mnist_data.classes

    def __len__(self) -> int:
        return len(self.mnist_data_split)

    def __getitem__(self, index: int) -> Any:
        return self.mnist_data_split.__getitem__(index)


if __name__ == "__main__":

    def siuadyh():
        """ """
        import tqdm

        batch_size = 32

        dt_t = MNISTDataset2(
            Path(Path.home() / "Data" / "mnist"), split=SplitEnum.training
        )

        print(len(dt_t))

        dt_v = MNISTDataset2(
            Path(Path.home() / "Data" / "mnist"), split=SplitEnum.validation
        )

        print(len(dt_v))

        dt = MNISTDataset2(
            Path(Path.home() / "Data" / "mnist"), split=SplitEnum.testing
        )

        print(len(dt))

        data_loader = torch.utils.data.DataLoader(
            dt, batch_size=batch_size, shuffle=False
        )

        for batch_idx, (imgs, label) in tqdm.tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Bro",
            ncols=80,
            leave=False,
        ):
            # pyplot.imshow(dt.inverse_transform(imgs[0]))
            # pyplot.imshow(imgs)
            # pyplot.show()
            print(imgs.shape)
            break

    siuadyh()
