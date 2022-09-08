import argparse
import copy
import glob
import os
import random
from operator import index
from pathlib import Path
from typing import Optional

import cv2
import fastmri
import h5py
import numpy as np
import pandas as pd
from fastmri.data import transforms as T
from PIL import Image, ImageDraw
from tqdm import tqdm

from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.preprocessing import MinMaxScaler

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 2:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval
    ssim = structural_similarity(gt, pred, data_range=maxval, win_size=3, K1 = 0.01, K2 = 0.03)

    return ssim


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)

class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    # def stddevs(self):
    #     return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def get_list(self):
        means = self.means()
        # stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return [means[name] for name in metric_names]

def checkBoundingBox(img, x0, y0, width, height):
    threshold = 40
    x1 = int(x0 + width)
    y1 = int(y0 + height)
    regional_image = np.array(img[y0:y1, x0:x1])
    # pixel_mean = np.mean(np.mean(regional_image[y0:y1, x0:x1]))
    zero_number = np.sum((regional_image.flatten() < 2e-05))
    rows, cols = regional_image.shape
    total_number = rows * cols
    # print(zero_number / total_number)

    while zero_number / total_number > 0.2:
        # print("fraction:", zero_number / total_number)
        x0 = random.randint(threshold,min(320-threshold,320-width)) if (320-width) > threshold else random.randint(min(320-threshold,320-width),threshold)
        y0 = random.randint(threshold,min(320-threshold,320-height)) if (320-height) > threshold else random.randint(min(320-threshold,320-height),threshold)
        x1 = int(x0 + width)
        y1 = int(y0 + height)
        # print("x0,y0,x1,y1",x0,y0,x1,y1)
        regional_image = np.array(img[y0:y1, x0:x1])
        # pixel_mean = np.mean(np.mean(regional_image[y0:y1, x0:x1]))
        zero_number = np.sum((regional_image.flatten() < 2e-05))
    return x0, y0

def save_fig(img, annotations, save_path, image_type):
    # global image + bounding boxes
    annotations = annotations.reset_index(drop=True)
    regional_image_list = []
    zoom_in_image_list = []

    arrimg = np.squeeze(img)
    image_2d_scaled = (np.maximum(arrimg, 0) / arrimg.max()) * 255.0
    image_2d_scaled = Image.fromarray(np.uint8(image_2d_scaled))
    image_2d_scaled_copy = copy.deepcopy(image_2d_scaled)
    plotted_image = ImageDraw.Draw(image_2d_scaled)

    # iterate multiply annoations in a slice
    for index, row in annotations.iterrows():
        image_2d_scaled_copy_copy = copy.deepcopy(image_2d_scaled_copy)
        image_2d_scaled_copy_for_save = copy.deepcopy(img)
        plotted_image_copy = ImageDraw.Draw(image_2d_scaled_copy_copy)
        fname, slice_choice, study_level, x0, y0, w, h, label_txt, *other = row
        w = 7 if int(w) < 7 else w
        h = 7 if int(h) < 7 else h
        x0, y0, x1, y1, label_txt = int(x0), int(y0), int(x0+w), int(y0+h), str(label_txt)
        
        sub_path = os.path.join(save_path, fname)
        if(os.path.exists(sub_path)) == False:
            os.makedirs(sub_path)
        
        if(os.path.exists(os.path.join(save_path, image_type))) == False:
            os.makedirs(os.path.join(save_path, image_type))
        
        cv2.imwrite(os.path.join(
            save_path, image_type, f'{fname}_{slice_choice}_{study_level}_Slice_Level_image.png'), np.array(image_2d_scaled_copy_copy))

        # plot bounding box
        plotted_image.rectangle(((x0, y0), (x1, y1)), outline="white")
        if label_txt!= "-1":
            plotted_image.text((x0, max(0, y0 - 10)), label_txt, fill="white")

        plotted_image_copy.rectangle(((x0, y0), (x1, y1)), outline="white")
        if label_txt!= "-1":
            plotted_image_copy.text((x0, max(0, y0 - 10)), label_txt, fill="white")

        # annotated image
        regional_image = np.array(img[y0:y1, x0:x1])
        regional_image_list.append(copy.deepcopy(regional_image))
        regional_image = (np.maximum(regional_image, 0) / regional_image.max()) * 255.0
        regional_image = Image.fromarray(np.uint8(regional_image))
        regional_image = np.array(regional_image)

        cv2.imwrite(os.path.join(
            sub_path, f'{image_type}_{slice_choice}_{study_level}_label#{index}_Bounding_Box_image.jpg'), regional_image)

        # zoom in image
        region_size = 40
        x_left = x0 if x0 < region_size else region_size
        x_right = (320 - x1) if (320 - x1) < region_size else region_size
        y_top = y0 if y0 < region_size else region_size
        y_bottom = (320 - y1) if (320 - y1) < region_size else region_size
        x0_ = x0 - x_left
        x1_ = x1 + x_right
        y0_ = y0 - y_top
        y1_ = y1 + y_bottom

        img_sub_for_save = np.array(image_2d_scaled_copy_for_save)[y0_:y1_, x0_:x1_]
        zoom_in_image_list.append(img_sub_for_save)

        img_sub_2d_scaled = np.array(image_2d_scaled_copy_copy)
        img_sub = img_sub_2d_scaled[y0_:y1_, x0_:x1_]
        cv2.imwrite(os.path.join(
            sub_path, f'{image_type}_{slice_choice}_{study_level}_label{index}_Zoom_Out_image.jpg'), img_sub)

    image_2d_scaled = np.array(image_2d_scaled)
    cv2.imwrite(os.path.join(
        sub_path, f'{image_type}_{slice_choice}_{study_level}_whole_image.png'), image_2d_scaled)

    return regional_image_list, zoom_in_image_list

def plot_bounding_box(image, labels):
    plotted_image = ImageDraw.Draw(image)
    for label in labels:
        _, _, _, x0, y0, w, h, label_txt, *other = label
        x1 = x0 + w
        y1 = y0 + h
        plotted_image.rectangle(((x0,y0), (x1,y1)), outline="white")
        plotted_image.text((x0, max(0, y0 - 10)), label_txt, fill= "white")
    return np.array(image)

def create_error_map(gt_recon, accX_recon, file_name, slice_choice, annotations, save_path):
    error_map = np.absolute(gt_recon-accX_recon)
    arrimg = np.squeeze(error_map)
    image_2d_scaled = (np.maximum(arrimg,0) / arrimg.max()) * 255.0
    image_2d_scaled = Image.fromarray(np.uint8(image_2d_scaled))
    if args.annotation_type != 'normal':
        annotated_img = plot_bounding_box(image_2d_scaled, annotations.values.tolist())
        image_2d_scaled = np.array(annotated_img)
    else:
        image_2d_scaled = np.array(image_2d_scaled)
    cv2.imwrite(os.path.join(save_path, 'error_map', f'{file_name}_{slice_choice}__error_map.png'), image_2d_scaled)


def compare_results(file_name, data_path, recon_path, save_path, annotation_df, acc_rate):
    final_results_df = pd.DataFrame(
        columns=['Sample', 'Slice', 'Annotation#', 'Annotation', 'Level', 'Acc', 'mse', 'nmse', 'psnr', 'ssim','pixel_mean','pixel_std'])

    # read annotation
    annotations_sub = annotation_df[annotation_df['fname'] == file_name]

    # file_name = os.path.basename(file_path)
    file_path = data_path / f"{file_name}.h5"
    img_path = recon_path / 'reconstructions' / f"{file_name}.h5"
    annotation_index = -1 if args.annotation_type == "abnormal" else -2

    METRIC_FUNCS = dict(
        MSE=mse,
        NMSE=nmse,
        PSNR=psnr,
        SSIM=ssim,
        )

    if os.path.exists(img_path):
        print("exist:", img_path)
        for slice_choice in annotations_sub['slice'].unique():
            print("slice_choice:", slice_choice)
            annotations = annotations_sub[annotations_sub['slice'] == slice_choice]

            # read data
            # gt / gt_re
            hf_gt = h5py.File(file_path, 'r')
            gt_recon = hf_gt['reconstruction_rss'][:]
            gt_recon = gt_recon[:, ::-1, :] #flip up down

            # accX image
            hf_accX = h5py.File(img_path, 'r')
            accX_recon = hf_accX['reconstruction'][:]
            accX_recon = accX_recon[:, ::-1, :] #flip up down

            # if (gt_recon[slice_choice, :, :].shape[0] == 320) and (gt_recon[slice_choice, :, :].shape[1] == 320):
            # save Slice_level figure and get bounding box images
            print("shape", gt_recon[slice_choice, :, :].shape)
            gt_region, _ = save_fig(
                gt_recon[slice_choice, :, :], annotations, save_path, image_type="gt")
            accX_region, _ = save_fig(
                accX_recon[slice_choice, :, :], annotations, save_path, image_type="accX")

            if(os.path.exists(os.path.join(save_path, 'error_map'))) == False:
                os.makedirs(os.path.join(save_path, 'error_map'))
            create_error_map(gt_recon[slice_choice, :, :], accX_recon[slice_choice, :, :], file_name, slice_choice, annotations, save_path)

            # traverse all annotations in a slice
            for i in range(len(annotations)):
                print("i:",i)
                # Slice_Level
                # accX_results = Evaluation_Metrics(
                #     gt_recon[slice_choice, :, :], accX_recon[slice_choice, :, :], all_slice=False)

                # # Save Results
                # results_list = [file_name, slice_choice, i, annotations.iloc[i,-1], 'Slice_Level', acc_rate,
                #                 accX_results[0], accX_results[1], accX_results[2], accX_results[3]]
                # final_results_df.loc[len(final_results_df)] = results_list

                # Slice_level evaluation
                metrics = Metrics(METRIC_FUNCS)
                target = gt_recon[slice_choice, :, :]
                recons = accX_recon[slice_choice, :, :]
                recons_mean = recons.flatten().mean()
                recons_std = recons.flatten().std()
                target = T.center_crop(
                    target, (target.shape[-1], target.shape[-1])
                )
                recons = T.center_crop(
                    recons, (target.shape[-1], target.shape[-1])
                )

                metrics.push(target, recons)
                accX_results = metrics.get_list()
                # Save Restuls
                results_list = [file_name, slice_choice, i, annotations.iloc[i,annotation_index], 'Slice_Level', acc_rate,
                                    accX_results[0], accX_results[1], accX_results[2], accX_results[3], recons_mean, recons_std]
                final_results_df.loc[len(final_results_df)] = results_list

                # Bounding Box
                # calculate
                metrics = Metrics(METRIC_FUNCS)
                target = gt_region[i]
                recons = accX_region[i]
                recons_mean = recons.flatten().mean()
                recons_std = recons.flatten().std()

                metrics.push(target, recons)
                accX_re_results = metrics.get_list()
                # Save Restuls
                results_list = [file_name, slice_choice, i, annotations.iloc[i,annotation_index], 'Bounding box', acc_rate,
                                accX_re_results[0], accX_re_results[1], accX_re_results[2], accX_re_results[3], recons_mean, recons_std]
                final_results_df.loc[len(final_results_df)] = results_list

                # # Zoom-out from Bouding Boxes
                # accX_re_zoom_results = Evaluation_Metrics(
                #     gt_zoom_out[i], accX_zoom_out[i], all_slice=False)
                # results_list = [file_name, slice_choice, i, annotations.iloc[i,-1], 'Zoom_in', acc_rate,
                #                 accX_re_zoom_results[0], accX_re_zoom_results[1], accX_re_zoom_results[2], accX_re_zoom_results[3]]
                # final_results_df.loc[len(final_results_df)] = results_list
            # else:
            #         print(
            #             f"file_name:{file_name}, slice:{slice_choice}, img_size is not standard")
    else:
        print(f"{file_name} no reconstruction files found")

    return final_results_df

def main(args):
    # Ground Truth
    data_path = args.data_path
    # Reconstruction
    accelerations = args.accelerations[0]
    annotation_type = args.annotation_type
    recon_path = args.recon_path / str(accelerations)
    print("recon_path",recon_path)
    save_path = args.save_path / annotation_type / str(accelerations)
    os.makedirs(save_path, exist_ok=True)
    annotation_df = pd.read_csv(Path(os.getcwd(), '.annotation_cache', f'brainmain_{annotation_type}.csv'))

    # Get Annotations from AnnotatedSliceDataset
    for fname in tqdm(annotation_df['fname'].unique()):
        final_results = compare_results(
            fname, data_path, recon_path, save_path, annotation_df, accelerations)
        output_path=os.path.join(save_path, 'output.csv')
            
        if final_results is not None:
            final_results.to_csv(output_path, mode = 'a',
                                 header = not os.path.exists(output_path))
        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--recon_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
    )

    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Path for saving pathology evaluation csv",
    )

    parser.add_argument(
        "--annotation_type",
        default='abnormal',
        choices=('abnormal','normal','random'),
        type=str,
        help="Comparison for abnormal samples, normal samples, random selected bounding boxes",
    )

    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=4,
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--random_file",
        default=0,
        choices=(0,1),
        type=int,
        help="Use same annotation for random select file (for comparison of patches effects)",
    )

    args = parser.parse_args()

    main(args)
