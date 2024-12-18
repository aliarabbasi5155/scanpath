"""Train script.
Usage:
  train.py <hparams> <dataset_root> [--cuda=<id>]
  train.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""

import torch
import numpy as np
import json
from docopt import docopt
from os.path import join
from irl_dcb.config import JsonConfig
from dataset import process_data
from irl_dcb.builder import build
from irl_dcb.trainer import Trainer
torch.manual_seed(42619)
np.random.seed(42619)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = "files/DCBs_JSONs/20230301_1147_sail_serach.json"
    # hparams = "hparams/coco_search18.json"
    dataset_root = "files/Stroop_DataSet"
    hparams = JsonConfig(hparams)

    # dir of pre-computed beliefs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()

    # load ground-truth human scanpaths
    # with open(join(dataset_root, # TODO: scanpath dakhele directory e dataset root bashe behtare
    #                'SAIL_fixations_TP_train.json')) as json_file:
    #     human_scanpaths_train = json.load(json_file)
    with open(join(dataset_root,'stroop_human_scanpath_train_split.json')) as json_file:
        human_scanpaths_train = json.load(json_file)

    # with open(join(dataset_root, # TODO: scanpath dakhele directory e dataset root bashe behtare
    #                'SAIL_fixations_TP_train.json')) as json_file:
    with open(join(dataset_root,'stroop_human_scanpath_valid_split.json')) as json_file:
        human_scanpaths_valid = json.load(json_file)

    # exclude incorrect scanpaths
    if hparams.Train.exclude_wrong_trials:
        human_scanpaths_train = list(
            filter(lambda x: x['correct'] == 1, human_scanpaths_train))
        human_scanpaths_valid = list(
            filter(lambda x: x['correct'] == 1, human_scanpaths_valid))

    # process fixation data
    dataset = process_data(human_scanpaths_train, human_scanpaths_valid,
                           DCB_dir_HR, DCB_dir_LR, bbox_annos, hparams)

    built = build(hparams, True, device, dataset['catIds'])
    trainer = Trainer(**built, dataset=dataset, device=device, hparams=hparams)
    trainer.train()
