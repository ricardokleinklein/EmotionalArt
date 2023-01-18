""" Evaluate CLIP Training

Script designed to compare how well can a fine-tuned CLIP of our own perform
on the task of retrieving the right image for each caption in comparison to
a default CLIP model.

NOTE:
    The notebook VisualValidationCLIP does a reduced instance of this
    functionality, also allowing for its visualization.

"""
import argparse
from typing import Tuple, List, Any, Optional

import numpy
import pandas
import torch
import random

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from nltk.tokenize import sent_tokenize
from data_preprocess.datasets import CLIPDataset
from neural_models.transformers import CLIP


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Dataset CSV")
    parser.add_argument("feat", type=str, help="Column name containing text "
                                               "features")
    parser.add_argument("run", type=str,
                        help="Run directory with model within it")
    parser.add_argument("-b", "--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device on which to run the experiment")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """ Fix the random seed

    Args:
        seed: Random seed.

    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)


def custom_model(data: torch.utils.data.Dataset,
                 dataloader: torch.utils.data.DataLoader,
                 checkpoint: Path, device: torch.device
                 ) -> Tuple[List[Any], List[Any], Any]:
    """ Measure the accuracy of the custom model.

    Args:
        dataloader:
        checkpoint:
        device:

    Returns:

    """
    model = CLIP(output_embed=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device),
                          strict=True)
    model.eval()

    sentences = []
    visual_embeddings = []
    image_paths = []
    textual_embeddings = []
    label_ids = []

    with torch.no_grad():
        for b, batch in enumerate(tqdm(dataloader)):
            batch_inputs, batch_label = batch
            batch_idxs = list(batch_label.cpu().numpy())
            label_ids.extend(batch_idxs)
            image_paths.extend(numpy.array(data.image_paths)[batch_idxs])
            sentences.extend(numpy.array(data.captions)[batch_idxs])

            if isinstance(batch_inputs, list):
                batch_inputs = [
                    {k: val.to(device) for k, val in x.items()}
                    for x in batch_inputs
                ]
            output = model(batch_inputs).detach().cpu().numpy()
            visual_embeddings.extend(output[0])
            textual_embeddings.extend(output[1])
    visual_embeddings = numpy.array(visual_embeddings)
    textual_embeddings = numpy.array(textual_embeddings)
    model_preds = visual_embeddings @ textual_embeddings.T
    return sentences, image_paths, model_preds


def default_model(sentences, image_paths):
    default_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    default_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    default_texts_embeds = []
    default_vision_embeds = []
    for b, (sent, img_path) in tqdm(enumerate(zip(sentences, image_paths)),
                                    total=len(sentences)):
        image = Image.open(img_path).convert("RGB")
        inputs = default_processor(text=[sent], images=image,
                               return_tensors='pt', padding='max_length',
                               truncation=True)
        outputs = default_model(**inputs)
        default_texts_embeds.append(
            outputs.text_embeds.squeeze().detach().cpu().numpy())
        default_vision_embeds.append(
            outputs.image_embeds.squeeze().detach().cpu().numpy())
    default_texts_embeds = numpy.array(default_texts_embeds)
    default_vision_embeds = numpy.array(default_vision_embeds)

    default_preds = default_vision_embeds @ default_texts_embeds.T
    return default_preds


def caption_in_image_cc(dataset, which_col, im_path, caption):
    cc = dataset[dataset['localpath'] == im_path]
    cc = sent_tokenize(cc[which_col].values[0])
    return caption in cc


def measure_retrieval(dataset, which_col, sentences, im_paths, predictions):
    acc = 0
    for i, (im, cap) in enumerate(zip(im_paths, sentences)):
        bet = sentences[numpy.argmax(predictions[i])]
        cap_in_cc = caption_in_image_cc(dataset, which_col, im, bet)
        if cap_in_cc:
            acc += 1
    return acc / len(im_paths)


def main():
    args = parse_args()
    device = torch.device(args.device)
    col = args.feat
    checkpoint = Path(args.run) / "model_state_dict.pt"

    dataset = pandas.read_csv(args.src)
    dataset = dataset[dataset['split'] == "test"]
    data = CLIPDataset(data=dataset, text_col=col, image_col="localpath")
    dataloader = data.load(batch_size=args.batch)

    sentences, paths, preds = custom_model(data, dataloader, checkpoint,
                                           device)
    custom_acc = measure_retrieval(dataset, col, sentences, paths, preds)

    default_preds = default_model(sentences, paths)
    default_acc = measure_retrieval(dataset, col, sentences, paths, default_preds)

    print(f"Custom model: {custom_acc:.3f} -- Default: {default_acc:.3f} (N "
          f"= {len(dataset)})")


if __name__ == "__main__":
    main()
