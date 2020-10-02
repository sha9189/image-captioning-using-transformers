from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
import torch

from PIL import Image
import numpy as np
import random
import os
import spacy

from .utils import nested_tensor_from_tensor_list, read_json, read_txt

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# spacy_eng = spacy.load("en")
# def tokenize_eng(text):
#     return [tok.text for tok in spacy_eng.tokenizer(text)]
english = torch.load("english.pth")


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training', tokenizer=english):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [self._process(val)
                      for val in ann.split("\n")]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        self.english = english
        self.max_length = max_length + 1

    def _process(self, val):
        image_id = val.split()[0]
        caption = val.split()[1:]
        caption = " ".join(caption)
        return (image_id, caption)
    
    def caption_encoder(self, caption):
        tokens = english.tokenize(caption)
        tokens = ["<sos>"] + tokens + ["<eos>"]
        numbs = english.numericalize([tokens])
        numbs = numbs.numpy().T[0]
        caption = np.zeros(self.max_length)
        caption[:len(numbs)] = numbs
        
        cap_mask = np.ones(self.max_length)
        cap_mask[:len(numbs)] = 0
        return caption, cap_mask.astype(bool)

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption, cap_mask = self.caption_encoder(caption)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'train2017')
        train_file = 'train_captions.txt'
        data = CocoCaption(train_dir, read_txt(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training', tokenizer=english)
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'val2017')
        val_file = 'val_captions.txt'
        data = CocoCaption(val_dir, read_txt(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation', tokenizer = english)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
