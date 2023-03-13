import torch
import json
import click
from tqdm import tqdm
from torchvision.transforms import CenterCrop

from mmpt.models import MMPTModel
from mmpt.datasets.vl_bench import Dataset_v1, process_path
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
    '--something-something-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
def main(json_path, quva_dir, something_something_dir, device, output_file):
    data = Dataset_v1(
        json_path,
        quva_dir=quva_dir,
        something_something_dir=something_something_dir,
        device=device,
    )
    preprocessor = Preprocessing('s3d')
    crop = CenterCrop(IMAGE_SIZE)

    model, tokenizer, aligner = MMPTModel.from_pretrained(
        "projects/retri/videoclip/how2.yaml")
    model = model.to(device)
    model.eval()
    _tokenize = lambda text: tokenize(aligner, tokenizer, text, device)
    results = dict()

    for item in tqdm(data):
        vframes = preprocessor(crop(item['video']))
        vframes = vframes.permute(0, 2, 3, 4, 1).unsqueeze(0)
        vframes = vframes.to(device)
        caption = _tokenize(item['caption'])
        foils = [_tokenize(foil) for foil in item['foils']]
        item_scores = list()

        with torch.no_grad():
            vfeats, vmasks = model.extract_video_features(vframes)
            caption_s = model(
                caption[0],
                caption[1],
                vfeats=vfeats,
                vmasks=vmasks,
                return_score=True,
            )['score'].item()
            item_scores.append(caption_s)

            for foil in foils:
                foil_s = model(
                    foil[0],
                    foil[1],
                    vfeats=vfeats,
                    vmasks=vmasks,
                    return_score=True,
                )['score'].item()
                item_scores.append(foil_s)

        results[item['item_id']] = dict()
        results[item['item_id']]['scores'] = item_scores

    with open(process_path(output_file), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

