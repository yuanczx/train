import os.path as osp
from typing import Dict, Sequence
import numpy as np
from PIL import Image
from mmseg.registry import METRICS
from rein.metrics.dg_metrics import DGIoUMetric


@METRICS.register_module()
class PerImageIoU(DGIoUMetric):
    def __init__(self,save_dir="./result",**kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.img_paths = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        assert self.dataset_meta is not None
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            self.img_paths.append(data_sample['img_path'])
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
                res1, res2, res3, res4 = self.intersect_and_union(
                    pred_label, label, num_classes, self.ignore_index
                )
                dataset_key = "unknown"
                for key in self.dataset_keys:
                    if key in data_samples[0]["seg_map_path"]:
                        dataset_key = key
                        break
                self.results.append([dataset_key, res1, res2, res3, res4])
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(data_sample["img_path"]))[0]
                png_filename = osp.abspath(osp.join(self.output_dir, f"{basename}.png"))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get("reduce_zero_label", False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        assert self.dataset_meta is not None
        class_names = self.dataset_meta['classes']
        with open(osp.join(self.save_dir,'iou_result.csv'),'a+') as f:
            for idx, result in enumerate(results):
                for i in range(len(class_names)):
                    iou = result[1][i]/result[2][i]
                    f.write(f"{self.img_paths[idx]},{class_names[i]},{iou.item()}\n")
        return super().compute_metrics(results)
