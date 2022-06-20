import argparse
import copy
import glob
import os
import random
from operator import index
from pathlib import Path

import cv2
import fastmri
import h5py
import numpy as np
import pandas as pd
from fastmri.data import transforms as T
from PIL import Image, ImageDraw
from runstats import Statistics
from tqdm import tqdm

try:
    from skimage.measure import compare_psnr
    peak_signal_noise_ratio = compare_psnr
except ImportError:
    from skimage.metrics import peak_signal_noise_ratio

from skimage.metrics import structural_similarity


def Evaluation_Metrics(gt, pred, all_slice=True):
    _mse = mse(gt, pred)
    _nmse = nmse(gt, pred)
    _psnr = psnr(gt, pred)
    _ssim = ssim(gt, pred) if all_slice is True else ssim_local(gt, pred)
    return (_mse, _nmse, _psnr, _ssim)


def mse(gt, pred):
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), channel_axis=2, data_range=gt.max()
    )


def ssim_local(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        gt, pred, channel_axis=None, data_range=gt.max()
    )


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
        fname, slice_choice, study_level, x0, y0, w, h, label_txt = row
        w = 7 if int(w) < 7 else w
        h = 7 if int(h) < 7 else h
        x0, y0, x1, y1 = int(x0), int(y0), int(x0+w), int(y0+h)
        sub_path = os.path.join(save_path, fname)
        if(os.path.exists(sub_path)) == False:
            os.makedirs(sub_path)
        
        if(os.path.exists(os.path.join(save_path, image_type))) == False:
            os.makedirs(os.path.join(save_path, image_type))
        
        cv2.imwrite(os.path.join(
            save_path, image_type, f'{fname}_{slice_choice}_{study_level}_whole_image.png'), np.array(image_2d_scaled_copy_copy))

        # plot bounding box
        plotted_image.rectangle(((x0, y0), (x1, y1)), outline="white")
        plotted_image.text((x0, max(0, y0 - 10)), label_txt, fill="white")

        plotted_image_copy.rectangle(((x0, y0), (x1, y1)), outline="white")
        plotted_image_copy.text((x0, max(0, y0 - 10)), label_txt, fill="white")

        # annotated image
        regional_image = np.array(img[y0:y1, x0:x1])
        regional_image_list.append(copy.deepcopy(regional_image))
        regional_image = (np.maximum(regional_image, 0) / regional_image.max()) * 255.0
        regional_image = Image.fromarray(np.uint8(regional_image))
        regional_image = np.array(regional_image)

        cv2.imwrite(os.path.join(
            sub_path, f'{image_type}_{slice_choice}_{study_level}_label#{index}_regional_image.jpg'), regional_image)

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
            sub_path, f'{image_type}_{slice_choice}_{study_level}_label{index}_zoom_in_image.jpg'), img_sub)

    image_2d_scaled = np.array(image_2d_scaled)
    cv2.imwrite(os.path.join(
        sub_path, f'{image_type}_{slice_choice}_{study_level}_whole_image.png'), image_2d_scaled)

    return regional_image_list, zoom_in_image_list


def compare_results(file_name, data_path, recon_path, save_path, annotation_df, acc_rate):
    final_results_df = pd.DataFrame(
        columns=['Sample', 'Slice', 'Annotation#', 'Annotation', 'Level', 'Acc', 'mse', 'nmse', 'psnr', 'ssim'])

    # read annotation
    annotations_sub = annotation_df[annotation_df['file'] == file_name]

    # file_name = os.path.basename(file_path)
    file_path = data_path / f"{file_name}.h5"
    img_path = recon_path / 'reconstructions' / f"{file_name}.h5"

    for slice_choice in annotations_sub['slice'].unique():

        if os.path.exists(img_path):
            print("exist:", img_path)
            annotations = annotations_sub[annotations_sub['slice'] == slice_choice]

            # read data
            # gt / gt_re
            hf_gt = h5py.File(file_path, 'r')
            gt_recon = hf_gt['reconstruction_rss'][:]

            # accX image

            hf_accX = h5py.File(img_path, 'r')
            accX_recon = hf_accX['reconstruction'][:]

            for i in range(len(annotations)):
                # Gloabl
                if (gt_recon[slice_choice, :, :].shape[0] == 320) and (gt_recon[slice_choice, :, :].shape[1] == 320):
                    accX_results = Evaluation_Metrics(
                        gt_recon[slice_choice, :, :], accX_recon[slice_choice, :, :], all_slice=False)
                    # Save Results
                    results_list = [file_name, slice_choice, i, annotations.iloc[i,-1], 'Global', acc_rate,
                                    accX_results[0], accX_results[1], accX_results[2], accX_results[3]]
                    final_results_df.loc[len(final_results_df)] = results_list

                # Region
                    gt_region, gt_zoom_in = save_fig(
                        gt_recon[slice_choice, :, :], annotations, save_path, image_type="gt")
                    accX_region, accX_zoom_in = save_fig(
                        accX_recon[slice_choice, :, :], annotations, save_path, image_type="accX")

                    accX_re_results = Evaluation_Metrics(
                        gt_region[i], accX_region[i], all_slice=False)

                    # Save Restuls
                    results_list = [file_name, slice_choice, i, annotations.iloc[i,-1], 'Region', acc_rate,
                                    accX_re_results[0], accX_re_results[1], accX_re_results[2], accX_re_results[3]]
                    final_results_df.loc[len(final_results_df)] = results_list

                    # Zoom-in
                    accX_re_zoom_results = Evaluation_Metrics(
                        gt_zoom_in[i], accX_zoom_in[i], all_slice=False)
                    results_list = [file_name, slice_choice, i, annotations.iloc[i,-1], 'Zoom_in', acc_rate,
                                    accX_re_zoom_results[0], accX_re_zoom_results[1], accX_re_zoom_results[2], accX_re_zoom_results[3]]
                    final_results_df.loc[len(final_results_df)] = results_list
                else:
                    print(
                        f"file_name:{file_name}, slice:{slice_choice}, img_size is not standard")
        else:
            print(f"{file_name} no reconstruction files found")

    return final_results_df


def main(args):
    # Ground Truth
    data_path = Path(args.data_path)
    # Reconstruction
    recon_path = Path(args.recon_path)
    save_path = Path(args.save_path)
    accelerations = args.accelerations
    os.makedirs(save_path, exist_ok=True)
    annotations_csv = pd.read_csv('/gpfs/home/sc9295/Projects/fastMRI/fastMRI/.annotation_cache/brainmain_random.csv')
    annotation_df = annotations_csv[(annotations_csv['x'] != -1)
                                     & (annotations_csv['study_level'] == 'No')]

    # annotation_df['y'] = annotation_df.apply(lambda row: 320 - int(row['y']) - int(row['height']) - 1, axis=1)

    # Get Annotations from AnnotatedSliceDataset
    for fname in tqdm(annotation_df['file'].unique()):
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
        "--accelerations",
        nargs="+",
        default=[4],
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
