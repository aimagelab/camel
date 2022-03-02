import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

import evaluation
import utils
from data import COCO, DataLoader, ImageField, TextField, RawField, Merge
from models import Captioner
from models import clip

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train')
_print_freq = 50
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


def evaluate_metrics(model, dataloader, text_field):
    model.eval()
    image_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    paths = {}
    gen = {}
    gts = {}

    header = 'Evaluation metrics:'
    with torch.no_grad():
        for it, ((image_paths, images), caps_gt) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            images = images.to(device)
            images = image_model(images)
            text, _ = model.beam_search(images, beam_size=5, out_size=1)
            caps_gen = text_field.decode(text)
            for i, (path_i, gts_i, gen_i) in enumerate(zip(image_paths, caps_gt, caps_gen)):
                paths['%d_%d' % (it, i)] = path_i
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    samples = [(paths[k], gen[k][0]) for k in list(paths.keys())[:20]]
    scores, _ = evaluation.compute_all_scores(gts, gen)
    
    return scores, samples


if __name__ == '__main__':
    _logger.info('CaMEL Evaluation')

    # Argument parsing
    parser = argparse.ArgumentParser(description='CaMEL Evaluation')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--annotation_folder', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--saved_model_path', type=str, required=True)

    parser.add_argument('--clip_variant', type=str, default='RN50x16')
    parser.add_argument('--network', type=str, choices=('online', 'target'), default='target')
    parser.add_argument('--disable_mesh', action='store_true')

    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--with_pe', action='store_true')
    args = parser.parse_args()

    _logger.info(args)

    # Pipeline for image regions
    clip_model, transform = clip.load(args.clip_variant, jit=False)
    image_model = clip_model.visual
    image_model.forward = image_model.intermediate_features
    image_field = ImageField(transform=transform)
    args.image_dim = image_model.embed_dim

    # Pipeline for text
    text_field = TextField()

    # Create the dataset and samplers
    dataset = COCO(image_field, text_field, args.image_folder, args.annotation_folder, args.annotation_folder)
    _, dataset_val, dataset_test = dataset.splits
    dataset_val = dataset_val.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField()})
    dataset_test = dataset_test.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField()})
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size // 5, sampler=val_sampler)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size // 5, sampler=test_sampler)

    # Create the model
    model = Captioner(args, text_field).to(device)
    model.forward = model.beam_search
    image_model = image_model.to(device)

    # Load the model weights
    fname = Path(args.saved_model_path)
    if not fname or not fname.is_file():
        raise ValueError(f'Model not found in {fname}')

    data = torch.load(fname)
    if args.network == 'target':
        _logger.info('Loading target network weights')
        model.load_state_dict(data['state_dict_t'])
    else:  # args.network == 'online'
        _logger.info('Loading online network weights')
        model.load_state_dict(data['state_dict_o'])

    # Validation captions
    _logger.info('Validation set')
    val_scores, val_samples = evaluate_metrics(model, dataloader_val, text_field)
    _logger.info(f'Validation scores {val_scores}')

    # Test captions
    _logger.info('Test set')
    test_scores, test_samples = evaluate_metrics(model, dataloader_test, text_field)
    _logger.info(f'Test scores {test_scores}')
