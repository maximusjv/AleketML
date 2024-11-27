import json
import os
import shutil

import PIL.Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors

from utils.consts import collate_fn


def download_dataset(save_dir, patched_dataset_gdrive_id=""):
    """Downloads and extracts the dataset if it doesn't exist locally.

    Args:
        save_dir: The directory to save the dataset.
        patched_dataset_gdrive_id: GoogleDrive id to download the dataset from.

    Returns:
        The path to the saved dataset directory.
    """
    if not os.path.exists(save_dir):
        import gdown  # Import gdown only when needed

        gdown.download(id=patched_dataset_gdrive_id, output="_temp_.zip")
        shutil.unpack_archive("_temp_.zip", save_dir)
        os.remove("_temp_.zip")
    return save_dir


class AleketDataset(Dataset):
    """
    Aleket dataset for object detection.

    Loads images and annotations from a directory.

    Args:
        dataset_dir (str): The path to the dataset directory.
        augmentation (torchvision.transforms.v2.Transform, optional):
            Augmentation transforms to apply to the images and bounding boxes.
    """

    def __init__(self, dataset_dir, augmentation=None):
        self.img_dir = os.path.join(dataset_dir, "imgs")

        with open(os.path.join(dataset_dir, "dataset.json"), "r") as annot_file:
            self.dataset = json.load(annot_file)

        self.ind_to_image = list(self.dataset.keys())
        self.image_to_ind = {img: i for i, img in enumerate(self.ind_to_image)}

        self.default_transforms = v2.ToDtype(torch.float32, scale=True)

        self.augmentation = augmentation
        print(f"Dataset loaded from {dataset_dir}")

    def to_indices(self, indices=None):
        """Converts a list of image names or indices to a list of indices."""
        if indices is None:
            return list(range(len(self.dataset)))
        return [self.image_to_ind[i] if isinstance(i, str) else i for i in indices]

    def to_names(self, indices=None):
        """Converts a list of image names or indices to a list of names."""
        if indices is None:
            return list(range(len(self.dataset)))
        return [self.ind_to_image[i] if isinstance(i, str) else i for i in indices]

    def by_full_images(self):
        """Groups image indices by their corresponding full image ID.

        Returns:
            dict[str, list[int]]: A dictionary mapping full image IDs to lists of
                corresponding image indices in the dataset.
        """
        by_full_images = {}
        for idx, name in enumerate(self.ind_to_image):
            full_image_id = name.split("_")[0]
            if full_image_id not in by_full_images:
                by_full_images[full_image_id] = []
            by_full_images[full_image_id].append(name)
        return by_full_images

    def get_annots(self, indices=None):
        """Retrieves annotations for the specified indices."""
        indices = self.to_indices(indices)
        targets = [
            {
                "image_id": idx,
                "labels": self.dataset[self.ind_to_image[idx]]["category_id"],
                "boxes": self.dataset[self.ind_to_image[idx]]["boxes"],
            }
            for idx in indices
        ]
        return targets

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.ind_to_image)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding target annotations.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image as a torchvision.tv_tensors.Image object
                and a dictionary of target annotations.
        """
        image_id = self.ind_to_image[idx]

        annots = self.dataset[image_id]
        labels, bboxes = annots["category_id"], annots["boxes"]

        img_path = os.path.join(f"{self.img_dir}", f"{image_id}.jpeg")
        img = PIL.Image.open(img_path).convert("RGB")

        img = tv_tensors.Image(img, dtype=torch.uint8)

        ht, wt = img.shape[-2:]  # Get height and width

        labels = torch.as_tensor(labels, dtype=torch.int64)
        if bboxes:
            bboxes = tv_tensors.BoundingBoxes(
                bboxes, format="XYXY", canvas_size=(ht, wt)
            )
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        if self.augmentation:
            img, bboxes, labels = self.augmentation(img, bboxes, labels)

        img, bboxes, labels = self.default_transforms(img, bboxes, labels)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": idx,
        }

        return img, target

    def split_dataset(self, dataset_fraction, validation_fraction, generator):
        """Splits the dataset into train and validation sets.

        Splits the dataset into train and validation sets, ensuring that all patches
        from the same full image are kept together in the same set.

        Args:
            dataset_fraction (float): The fraction of the dataset to use
                (for debugging/testing).
            validation_fraction (float): The fraction of the used dataset to
                allocate for validation.
            generator (np.random.Generator): A NumPy random generator for
                reproducible splitting.

        Returns:
            tuple[dict[str, list[int]], dict[str, list[int]]]: A tuple containing
                two dictionaries:
                - The first dictionary maps full image IDs to lists of patch
                    indices for the training set.
                - The second dictionary maps full image IDs to lists of patch
                    indices for the validation set.
        """
        by_full_images = self.by_full_images()

        full_images = list(by_full_images.keys())
        full_images = generator.permutation(full_images)

        total_num_samples = max(2, int(len(self.ind_to_image) * dataset_fraction))
        validation_num_samples = max(1, int(validation_fraction * total_num_samples))
        train_num_samples = total_num_samples - validation_num_samples

        train_len = 0
        train_set = {}
        validation_len = 0
        validation_set = {}

        for images in full_images:
            if validation_len < validation_num_samples:
                validation_set[images] = by_full_images[images]
                validation_len += len(by_full_images[images])
            elif train_len < train_num_samples:
                train_set[images] = by_full_images[images]
                train_len += len(by_full_images[images])

        return train_set, validation_set

    def create_dataloaders(self, train_indices, val_indices, batch_size, num_workers):
        """Creates DataLoaders for split dataset.

        Args:
            train_indices: Dataset indices to train.
            val_indices: Dataset indices to validate.
            batch_size: The batch size for the DataLoaders.
            num_workers: The number of worker processes for data loading.

        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and
                the validation DataLoader.
        """

        train_dataset = Subset(self, self.to_indices(train_indices))
        val_dataset = Subset(self, self.to_indices(val_indices))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        return train_dataloader, val_dataloader
