from .loader import MultiTaskDataset, TUTDataset
import pandas as pd

def get_dataset(cfg, mode):
    task = cfg.TASK
    dataset = cfg.DATASET
    if task == 'pixel':
        if dataset == 'nyu_v2':
            if mode == 'train':
                return MultiTaskDataset(
                            dataset = dataset,
                            data_dir='datasets/nyu_v2',
                            image_mean = 'datasets/nyu_v2/nyu_v2_mean.npy',
                            data_list_1='datasets/nyu_v2/list/training_seg.txt',
                            data_list_2='datasets/nyu_v2/list/training_normal_mask.txt',
                            output_size=cfg.TRAIN.OUTPUT_SIZE,
                            color_jitter=cfg.TRAIN.COLOR_JITTER,
                            random_scale=cfg.TRAIN.RANDOM_SCALE,
                            random_mirror=cfg.TRAIN.RANDOM_MIRROR,
                            random_crop=cfg.TRAIN.RANDOM_CROP,
                            ignore_label=255,
                        )
            elif mode == 'train_eval':
                return MultiTaskDataset(
                            dataset = dataset,
                            data_dir='datasets/nyu_v2',
                            image_mean = 'datasets/nyu_v2/nyu_v2_mean.npy',
                            data_list_1='datasets/nyu_v2/list/training_seg.txt',
                            data_list_2='datasets/nyu_v2/list/training_normal_mask.txt',
                            output_size=None,
                            random_scale=False,
                            random_mirror=False,
                            random_crop=False,
                            ignore_label=255,
                        )
            elif mode == 'val' or mode == 'test':
                return MultiTaskDataset(
                            dataset = dataset,
                            data_dir='datasets/nyu_v2',
                            image_mean = 'datasets/nyu_v2/nyu_v2_mean.npy',
                            data_list_1='datasets/nyu_v2/list/testing_seg.txt',
                            data_list_2='datasets/nyu_v2/list/testing_normal_mask.txt',
                            output_size=None,
                            color_jitter=False,
                            random_scale=False,
                            random_mirror=False,
                            random_crop=False,
                            ignore_label=255,
                        )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        clip = cfg.TYPE
        if mode == 'val' or mode == 'test':
            # print('aici')
            csv1 = pd.read_csv('meta_eval' + clip + '.csv')
            if cfg.MODEL.BACKBONE == 'Event_Scene':
                csv2 = pd.read_csv('meta_eval' + clip + '_frame' '.csv', index_col=0)
                return TUTDataset(csv1, False, True, clip, csv2)
            else:
                return TUTDataset(csv1, False, True, clip)
        else:
            csv1 = pd.read_csv('meta' + clip + '.csv')
            if cfg.MODEL.BACKBONE == 'Event_Scene':
                csv2 = pd.read_csv('meta' + clip + '_frame' '.csv', index_col=0)
                return TUTDataset(csv1, True, False, clip, csv2)
            else:
                # print('aici')
                return TUTDataset(csv1, True, False, clip)
