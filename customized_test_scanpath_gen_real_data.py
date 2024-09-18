"""Test script.
Usage:
  test.py <hparams> <checkpoint_dir> <dataset_root> [--cuda=<id>]
  test.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""

import os
import json
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from docopt import docopt
from os.path import join
from dataset import process_data
from irl_dcb.config import JsonConfig

import cv2 as cv

from irl_dcb.data import LHF_IRL, NEW_LHF_IRL
from irl_dcb.models import LHF_Policy_Cond_Small
from irl_dcb.environment import IRL_Env4LHF, NEW_IRL_Env4LHF
from irl_dcb import utils

torch.manual_seed(42620)
np.random.seed(42620)


def gen_scanpaths(generator,
                  env_test,
                  test_img_loader,
                  patch_num,
                  max_traj_len,
                  im_w,
                  im_h,
                  num_sample=10):
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader,
                        desc='trial ({}/{})'.format(i_sample + 1, num_sample))
        for i_batch, batch in enumerate(progress):
            env_test.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            with torch.no_grad():
                env_test.reset()
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            patch_num,
                                            max_traj_len,
                                            is_eval=True,
                                            sample_action=True)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])

    scanpaths = utils.actions2scanpaths(all_actions, patch_num, im_w, im_h)
    # Ali: I rempved the following line because I removed bbox_annos from the whole codes
    # but maybe I should add it back because it cuts the scanpath on enering the target bounding box
    utils.cutFixOnTarget(scanpaths, bbox_annos)

    return scanpaths

def plot_scanpaths_on_images(preds, hyperparams, image_source_location='images/', save_dir='result/'):
    for index,elem in enumerate(preds):
        elem['name'] = elem['name'].replace('.jpg', '.png')
        filename = elem['task']+"/" + elem['name']
        print(str(index)+". "+filename)

        image = cv.imread(image_source_location + filename)

        X = elem['X']
        Y = elem['Y']

        image = cv.resize(image, (hyperparams.Data.im_w, hyperparams.Data.im_h))

        for i in range(len(X)):
            x = int(X[i])
            y = int(Y[i])
            cv.putText(image, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv.circle(image, (x, y), 2, (255, 255, 255))
            if i > 0:
                xprec = int(X[i-1])
                yprec = int(Y[i-1])
                cv.line(image, (xprec, yprec), (x, y), (255, 255, 255))

        os.makedirs(save_dir + elem['task'] + "/", exist_ok=True)
        cv.imwrite(save_dir + elem['task'] + "/" + elem['name'], image)


if __name__ == '__main__':
    # args = docopt(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = "DCBs_JSONs/20230301_1147_sail_serach.json"
    dataset_root = "DCBs_JSONs/dataset_test"
    checkpoint = "assets/log_20240918_1203/checkpoints"

    hparams = JsonConfig(hparams)

    # dir of pre-computed DCBs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()
    # place holder for bounding box annotations
    # bbox_annos['bottle_111bottle.jpg'] = [43, 195, 34, 105]
    # bbox_annos['bottle_222bottle.jpg'] = [54, 205, 32, 114]
    # bbox_annos['bottle_333bottle.jpg'] = [54, 205, 32, 114]
    
    with open(join(dataset_root,
                   'SAIL_fixations_TP_train.json'), encoding='utf-8') as json_file:
        human_scanpaths_train = json.load(json_file)

    # cat_names = list(np.unique([x['task'] for x in human_scanpaths_train]))
    cat_names = list(np.unique([x['task'] for x in human_scanpaths_train]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))


    # dataset = process_data(human_scanpaths_train, human_scanpaths_valid,
    #                        DCB_dir_HR, DCB_dir_LR, bbox_annos, hparams)

    # train_task_img_pair = np.unique(
    #     [traj['task'] + '_' + traj['name'] for traj in human_scanpaths_train])
    train_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in human_scanpaths_train])
    test_dataset = NEW_LHF_IRL(DCB_dir_HR, DCB_dir_LR, train_task_img_pair, hparams.Data, catIds)
    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=2)
    
    # load trained model
    input_size = 134  # number of belief maps
    task_eye = torch.eye(len(catIds)).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(catIds), task_eye,
                                      input_size).to(device)

    generator.eval()

    state = torch.load(join(checkpoint, 'trained_generator.pkg'), map_location=device)

    generator.load_state_dict(state["model"])

    # build environment
    env_test = NEW_IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True)

    # generate scanpaths
    print('sample scanpaths (10 for each testing image)...')
    predictions = gen_scanpaths(generator,
                                env_test,
                                dataloader,
                                hparams.Data.patch_num,
                                hparams.Data.max_traj_length,
                                hparams.Data.im_w,
                                hparams.Data.im_h,
                                num_sample=10)
    
    plot_scanpaths_on_images(preds=predictions,
                             hyperparams=hparams,
                             image_source_location='Task Images/',
                             save_dir='results_Task Images/')
