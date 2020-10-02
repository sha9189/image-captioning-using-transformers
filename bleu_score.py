import os
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from models import caption
from datasets import utils
from datasets.utils import NestedTensor
import tqdm

from torch.utils.data import Dataset
from PIL import Image
import torchvision as tv
import torch

import spacy
spacy_eng = spacy.load("en")
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

from datasets.coco import val_transform
from configuration import Config
from datasets.utils import read_txt, nested_tensor_from_tensor_list

english = torch.load('english.pth')


class CocoCaptionVal(Dataset):
    def __init__(self, root, img_ids, max_length, transform=val_transform, tokenizer=english):
        super().__init__()

        self.root = root
        self.transform = transform
        self.img_ids = [img for img in img_ids.split("\n")]
        self.english = english
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image_id = self.img_ids[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        return image_id, image.tensors.squeeze(0), image.mask.squeeze(0)


def build_dataset(config):
    val_dir = os.path.join(config.dir, 'val2017')
    val_file = 'val_images.txt'
    data = CocoCaptionVal(val_dir, read_txt(
        val_file), max_length=config.max_position_embeddings, transform=val_transform, tokenizer = english)
    return data


config = Config()
dataset_val = build_dataset(config)
print(f"Valid: {len(dataset_val)}")
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=True, num_workers=config.num_workers)

                    
def load_model(model_num):
    model, _ = caption.build_model(config)
    model_num = model_num
    checkpoint = torch.load(config.checkpoint+str(model_num), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(config.device)
    return model


def create_caption_and_mask(max_length=config.max_position_embeddings, 
                             batch_size=config.batch_size):
    caption_template = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask_template = torch.ones((batch_size, max_length), dtype=torch.bool)

    caption_template[:, 0] = english.vocab.stoi["<sos>"]
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def predict(model, data_loader, device=config.device):
    model.eval()
    total = len(data_loader)
    img_cap_pair = dict()

    with tqdm.tqdm(total=total) as pbar:
        # iterate over each batch
        for img_ids, images, masks in data_loader:
            samples = NestedTensor(images, masks).to(device)
            caps, cap_masks =  create_caption_and_mask()
            caps, cap_masks = caps.to(device), cap_masks.to(device)
            
            # predict one word/loop and add it to the predicted caption for the next loop
            for i in range(config.max_position_embeddings-1):
                predictions = model(samples, caps, cap_masks)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)
                caps[:, i+1] = predicted_id
                cap_masks[:, i+1] = False
            for img_id, cap in zip(img_ids, caps):
                img_cap_pair[img_id] = cap
            pbar.update(1)
    return img_cap_pair


with open("val_captions.txt") as file:
    annotations = file.read()

real_pair = dict()

for annot in annotations.split("\n"):
        # Skip empty lines
        if len(annot)<1:
            continue
        cap = annot.split()[1:]
        cap = " ".join(cap)
        cap = english.tokenize(cap)
        try:
            image_id = annot.split()[0]
        except:
            print(image_id, ":", cap)
            continue
        if image_id not in real_pair.keys():
            real_pair[image_id] = []
        real_pair[image_id].append(" ".join(cap))


def decode_caption(output):
    sentence = []
    for idx in output:
        if idx == english.vocab.stoi["<eos>"]:
            break
        word = english.vocab.itos[idx]
        sentence.append(word)
    # Remove <sos> from sentence
    sentence = " ".join(sentence[1:])
    return sentence


def get_corpuses(img_cap_pair):
    references = []
    prediction_list = []
    for key in img_cap_pair.keys():
        # convert predicted caption tensor to text and add in prediction_list
        cap_tensor = img_cap_pair[key]
        cap = decode_caption(cap_tensor)
        prediction_list.append(cap)

        # add list of real captions for key to reeferences to maintain the order
        references.append(real_pair[key])
    return references, prediction_list


def return_model_score(model_num):
    """Takes model num as input and returns all 4 bleu scores in a dictionary."""
    model = load_model(model_num)
    img_cap_pair = predict(model, data_loader_val)
    references, prediction_list = get_corpuses(img_cap_pair)
    bleu_1, bleu_2, bleu_3, bleu_4 = (corpus_bleu(references, prediction_list, weights=(1, 0, 0, 0)),
                                      corpus_bleu(references, prediction_list, weights=(0.5, 0.5, 0, 0)),
                                      corpus_bleu(references, prediction_list, weights=(1/3., 1/3., 1/3., 0)),
                                      corpus_bleu(references, prediction_list, weights=(1, 1, 1, 1)))
    return {'epoch':model_num, 
            'bleu_1': bleu_1, 
            'bleu_2' : bleu_2, 
            'bleu_3': bleu_3, 
            'bleu_4' :bleu_4}


# columns = ['epoch', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
# history = pd.DataFrame(columns = columns)

history_file = "./history/history.csv"
# history.to_csv(path_or_buf=history_file, index=False)

history = pd.read_csv(filepath_or_buffer=history_file)


model_nums = range(15, 20)

for num in model_nums:
    print("Testing Model: ", num)
    score = return_model_score(num)

    history = pd.read_csv(filepath_or_buffer=history_file)
    history = history.append(score, ignore_index=True)
    history.to_csv(path_or_buf=history_file, index=False)
    print("Model Performance saved")
