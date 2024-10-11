import numpy as np
from irl_dcb.data import LHF_IRL, LHF_Human_Gaze
from irl_dcb.utils import compute_search_cdf, preprocess_fixations


def process_data(trajs_train,
                 trajs_valid,
                 DCB_HR_dir,
                 DCB_LR_dir,
                 target_annos,
                 hparams,
                 is_testing=False):
    target_init_fixs = {}
    eeg_train_data = {}  # Add a dictionary to store training EEG data
    eeg_valid_data = {}  # Add a dictionary to store validation EEG data
    for traj in trajs_train + trajs_valid:
        key = traj['task'] + '_' + traj['name']
        target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                 traj['Y'][0] / hparams.Data.im_h)
        # Collect EEG data for the corresponding task and image
        if 'eeg_data' in traj:  # Ensure EEG data is available in the trajectory
            if traj in trajs_train:
                eeg_train_data[key] = traj['eeg_data']  # Store EEG data for training
            else:
                eeg_valid_data[key] = traj['eeg_data']  # Store EEG data for validation

    cat_names = list(np.unique([x['task'] for x in trajs_train]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    # training fixation data
    train_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in trajs_train])
    train_fix_labels = preprocess_fixations(
        trajs_train,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length)

    # validation fixation data
    valid_task_img_pair = np.unique(
        [traj['task'] + '_' + traj['name'] for traj in trajs_valid])
    human_mean_cdf, _ = compute_search_cdf(trajs_valid, target_annos,
                                           hparams.Data.max_traj_length)
    print('target fixation prob (valid).:', human_mean_cdf)
    valid_fix_labels = preprocess_fixations(
        trajs_valid,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length)

    # load image data
    train_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                train_task_img_pair, target_annos,
                                hparams.Data, catIds, eeg_data=eeg_train_data)
    valid_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                valid_task_img_pair, target_annos,
                                hparams.Data, catIds, eeg_data=eeg_valid_data)

    # load human gaze data
    train_HG_dataset = LHF_Human_Gaze(DCB_HR_dir, DCB_LR_dir, train_fix_labels,
                                      target_annos, hparams.Data, catIds)
    valid_HG_dataset = LHF_Human_Gaze(DCB_HR_dir,
                                      DCB_LR_dir,
                                      valid_fix_labels,
                                      target_annos,
                                      hparams.Data,
                                      catIds,
                                      blur_action=True)

    return {
        'catIds': catIds,
        'img_train': train_img_dataset,
        'img_valid': valid_img_dataset,
        'gaze_train': train_HG_dataset,
        'gaze_valid': valid_HG_dataset,
        'human_mean_cdf': human_mean_cdf,
        'bbox_annos': target_annos
    }
