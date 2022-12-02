import torch
from torchvision.io import read_video
from math import ceil
import click
from tqdm import tqdm
from torchvision.transforms import CenterCrop

from mmpt.models import MMPTModel
from mmpt.datasets.vl_bench import Dataset_v1
from scripts.video_feature_extractor.preprocessing import Preprocessing


IMAGE_SIZE = 224
DTYPE = torch.float16


def tokenize(aligner, tokenizer, text, device):
    caps, cmasks = aligner._build_text_seq(
        tokenizer(
            text,
            add_special_tokens=False)["input_ids"],
    )
    caps = caps[None, :].to(device)
    cmasks = cmasks[None, :].to(device)
    return caps, cmasks


@click.command()
@click.option(
    '--json-path',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '--quva-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
def main(json_path, quva_dir, device):
    data = Dataset_v1(json_path, quva_dir=quva_dir, device=device)
    preprocessor = Preprocessing('s3d')
    crop = CenterCrop(IMAGE_SIZE)

    model, tokenizer, aligner = MMPTModel.from_pretrained(
        "projects/retri/videoclip/how2.yaml")
    model = model.to(device)
    model.eval()
    _tokenize = lambda text: tokenize(aligner, tokenizer, text, device)

    correct = 0
    numbers = []
    for item in tqdm(data):
        vframes = preprocessor(crop(item['video']))
        vframes = vframes.permute(0, 2, 3, 4, 1).unsqueeze(0)
        vframes = vframes.to(device)
        caption = _tokenize(item['caption'])
        foils = [_tokenize(foil) for foil in item['foils']]

        with torch.no_grad():
            vfeats, vmasks = model.extract_video_features(vframes)            
            caption_s = model(
                caption[0],
                caption[1],
                vfeats=vfeats,
                vmasks=vmasks,
                return_score=True,
            )['score'].item()

            max_foil_s = float('-Inf')
            for foil in foils:
                foil_s = model(
                    foil[0],
                    foil[1],
                    vfeats=vfeats,
                    vmasks=vmasks,
                    return_score=True,
                )['score'].item()

                if foil_s > max_foil_s:
                    max_foil_s = foil_s

                if max_foil_s > caption_s:
                    break

            if caption_s > max_foil_s:
                correct += 1

    accuracy = round(100 * correct / len(data), 2)
    print(f'accuracy={accuracy}%')


if __name__ == "__main__":
    main()

