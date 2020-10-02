import argparse
import torch
from PIL import Image
import spacy

try:
    spacy_eng = spacy.load("en")
except:
    spacy_eng = spacy.load("en_core_web_sm")
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

from datasets import coco
from configuration import Config
from models import caption 

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
args = parser.parse_args()
image_path = args.path

config = Config()
model, _ = caption.build_model(config)


# To load model weights, use the code below
model.backbone.load_state_dict(torch.load("checkpoints/checkpoint-breakdown/backbone.pth", map_location='cpu'))
model.input_proj.load_state_dict(torch.load("checkpoints/checkpoint-breakdown/input_proj.pth", map_location='cpu'))
model.transformer.load_state_dict(torch.load("checkpoints/checkpoint-breakdown/transformer.pth", map_location='cpu'))
model.mlp.load_state_dict(torch.load("checkpoints/checkpoint-breakdown/mlp.pth", map_location='cpu'))
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Image related ops
# image_path = "images/test1.jpg"
image = Image.open(image_path)
image = coco.val_transform(image)
image = image.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# Caption related ops
english = torch.load('english.pth')
start_token = english.vocab.stoi["<sos>"]
end_token = english.vocab.stoi["<eos>"]

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long).to("cuda" if torch.cuda.is_available() else "cpu")
    mask_template = torch.ones((1, max_length), dtype=torch.bool).to("cuda" if torch.cuda.is_available() else "cpu")

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 3:
            caption[:, i+1] = predicted_id[0]
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption

output = evaluate()
output = output.tolist()[0]

def decode_caption(output, end_token, english):
    sentence = []
    for idx in output:
        if idx == end_token:
            break
        word = english.vocab.itos[idx]
        sentence.append(word)
    # Remove <sos> from sentence
    sentence = " ".join(sentence[1:])
    return sentence

result = decode_caption(output, end_token, english)
result = result.capitalize()
print(result)