import json
import os.path as osp
import mmengine
import numpy as np
import torch

from mmseg.registry import DATASETS
from mmseg.datasets import CityscapesDataset


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class RCSDataset(CityscapesDataset):
    
    def __init__(self,
                 rare_class_sampling,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        rcs_cfg = rare_class_sampling
        self.rcs_enabled = rcs_cfg is not None

    
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']
            self.img_dir = self.data_prefix.get('img_path', None)
            self.ann_dir = self.data_prefix.get('seg_map_path', None)
            self.img_infos = self.load_annotations(
                img_dir=self.img_dir,
                img_suffix=self.img_suffix,
                ann_dir=self.ann_dir,
                seg_map_suffix=self.seg_map_suffix,
                split=None)
        

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                kwargs['data_root'], self.rcs_class_temp)
            logger = mmengine.MMLogger.get_current_instance()
            mmengine.print_log(f'RCS Classes: {self.rcs_classes}', logger=logger)
            mmengine.print_log(f'RCS ClassProb: {self.rcs_classprob}', logger=logger)

            with open(
                    osp.join(kwargs['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmengine.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])
        
        logger = mmengine.MMLogger.get_current_instance()
        mmengine.print_log(f'Loaded {len(img_infos)} images',logger=logger)
        return img_infos

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = super().__getitem__(i1)
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['data_samples'].gt_sem_seg.data == c)
                # mmengine.print_log(f'{j}: {n_class}',mmengine.MMLogger.get_current_instance())
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = super().__getitem__(i1)

        return {
            **s1
        }

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self[idx % len(self)]
            return {
                **s1
            }

