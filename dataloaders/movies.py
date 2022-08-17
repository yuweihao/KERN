import os
import pathlib

import config
import numpy as np
import PIL
import torch.utils.data
import torchvision.transforms

import dataloaders.blob
import dataloaders.image_transforms
import dataloaders.visual_genome

# Image transformation pipeline taken from the Visual Genome data loader.
_transformation_pipline = torchvision.transforms.Compose(
    [
        dataloaders.image_transforms.SquarePad(),
        torchvision.transforms.Resize(config.IM_SCALE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class Movies(torch.utils.data.Dataset):
    def __init__(self, data_dir="data/movies"):
        self.data_dir = pathlib.Path(data_dir)

        self.file_names = [
            file_name
            for extension in ["jpeg", "jpg"]
            for file_name in self.data_dir.glob(f"*.{extension}")
        ]

        # Since we use a model trained on the Visual Genome dataset, the object
        # and predicate classes that we want to predict are the ones from that
        # dataset.
        self.objects, self.predicates = dataloaders.visual_genome.load_info(
            config.VG_SGG_DICT_FN
        )

    def __getitem__(self, index):
        file_name = self.file_names[index]

        image = PIL.Image.open(file_name).convert("RGB")

        width, height = image.size

        factor = config.IM_SCALE / max(width, height)

        if height > width:
            size = (config.IM_SCALE, int(width * factor), factor)
        elif height < width:
            size = (int(height * factor), config.IM_SCALE, factor)
        else:
            size = (config.IM_SCALE, config.IM_SCALE, factor)

        # The gt_* attributes represent ground truth data, but since this
        # dataset will only be used for testing purposes (for now, at least),
        # we just fill in empty values.
        return {
            "img": _transformation_pipline(image),
            "img_size": size,
            "gt_boxes": np.array([]),
            "gt_classes": np.array([]),
            "gt_relations": np.array([]),
            "scale": config.IM_SCALE / config.BOX_SCALE,
            "index": index,
            "flipped": False,
            "fn": str(file_name),
        }

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def collate_fn(batch, mode="det", is_train=False, num_gpus=1):
        blob = dataloaders.blob.Blob(
            mode=mode,
            is_train=is_train,
            num_gpus=num_gpus,
            batch_size_per_gpu=len(batch) // num_gpus,
        )

        for item in batch:
            blob.append(item)

        blob.reduce()

        return blob
