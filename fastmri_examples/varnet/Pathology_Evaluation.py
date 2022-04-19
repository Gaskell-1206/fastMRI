from operator import index
import argparse
import pandas as pd
import numpy as np
# import glob
from pathlib import Path
import os
import fastmri
from fastmri.data import transforms as T
import h5py
from PIL import ImageDraw, Image
import cv2
from tqdm import tqdm

from runstats import Statistics
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


def cv2_clipped_zoom(img, zoom_factor=0):

    if zoom_factor == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int32)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(
        new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (
        height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - \
        pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1,
                                             pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def save_fig(img, annotations, save_path, image_type):
    # global image + bounding boxes
    annotations = annotations.reset_index(drop=True)
    regional_image_list = []

    arrimg = np.squeeze(img)
    image_2d_scaled = (np.maximum(arrimg, 0) / arrimg.max()) * 255.0
    image_2d_scaled = Image.fromarray(np.uint8(image_2d_scaled))
    plotted_image = ImageDraw.Draw(image_2d_scaled)

    # iterate multiply annoations in a slice
    for index, row in annotations.iterrows():
        fname, slice_choice, study_level, x0, y0, w, h, label_txt = row
        w = 7 if int(w) < 7 else w
        h = 7 if int(h) < 7 else h
        x0, y0, x1, y1 = int(x0), int(y0), int(x0+w), int(y0+h)
        sub_path = os.path.join(save_path, fname)
        if(os.path.exists(sub_path)) == False:
            os.makedirs(sub_path)

        # plot bounding box
        plotted_image.rectangle(((x0, y0), (x1, y1)), outline="white")
        plotted_image.text((x0, max(0, y0 - 10)), label_txt, fill="white")

        # annotated image
        regional_image = np.array(img[y0:y1, x0:x1])
        regional_image_list.append(regional_image)
        
        cv2.imwrite(os.path.join(
            sub_path, f'{slice_choice}_{study_level}_label#{index}_regional_image.jpg'), regional_image)

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
        img_sub_2d_scaled = np.array(image_2d_scaled)
        img_sub = img_sub_2d_scaled[y0_:y1_, x0_:x1_]
        # zoom_in_image = cv2_clipped_zoom(img_sub, zoom_factor=zoom_factor)
        cv2.imwrite(os.path.join(
            sub_path, f'{slice_choice}_{study_level}_label{index}_zoom_in_image.jpg'), img_sub)

    image_2d_scaled = np.array(image_2d_scaled)
    cv2.imwrite(os.path.join(
        sub_path, f'{slice_choice}_{study_level}_whole_image.png'), image_2d_scaled)

    return regional_image_list


def compare_results(file_name, data_path, recon_path, save_path, annotation_df, acc_rate):
    final_results_df = pd.DataFrame(
        columns=['Sample', 'Slice', 'Annotation#', 'Level', 'Acc', 'mse', 'nmse', 'psnr', 'ssim'])

    # file_name = os.path.basename(file_path)
    file_path = data_path / f"{file_name}.h5"

    # read annotation
    annotations_sub = annotation_df[annotation_df['fname'] == file_name]

    for slice_choice in annotations_sub['slice'].unique():
        img_path = recon_path / 'reconstructions' / f"{file_name}.h5"

        if os.path.exists(img_path):
            # print("exist:", img_path)
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
                accX_results = Evaluation_Metrics(
                    gt_recon[slice_choice, :, :], accX_recon[slice_choice, :, :], all_slice=False)
                # Save Results
                results_list = [file_name.split('.')[0], slice_choice, i, 'Global', acc_rate,
                                accX_results[0], accX_results[1], accX_results[2], accX_results[3]]
                final_results_df.loc[len(final_results_df)] = results_list

                # Region
                if gt_recon[slice_choice, :, :].shape[0] == 320:
                    gt_region = save_fig(
                        gt_recon[slice_choice, :, :], annotations, save_path, image_type="gt")
                    accX_region = save_fig(
                        accX_recon[slice_choice, :, :], annotations, save_path, image_type="accX")

                    accX_re_results = Evaluation_Metrics(
                        gt_region[i], accX_region[i], all_slice=False)
                    # Save Restuls
                    results_list = [file_name.split('.')[0], slice_choice, i, 'Region', acc_rate,
                                    accX_re_results[0], accX_re_results[1], accX_re_results[2], accX_re_results[3]]
                    final_results_df.loc[len(final_results_df)] = results_list
                else:
                    print(f"file_name:{file_name}, slice:{slice_choice}, img_size is not standard")
        else:
            print(f"{file_name} no reconstruction files found")

    return final_results_df


def main(args):
    # Ground Truth
    data_path = args.data_path
    # Reconstruction
    recon_path = args.recon_path
    save_path = args.save_path
    accelerations = args.accelerations

    annotations_list = pd.DataFrame(fastmri.data.mri_data.AnnotatedSliceDataset(
        data_path, "multicoil", "brain", "all", annotation_version="main").annotated_examples)[2].values.tolist()
    annotation_df = pd.DataFrame(columns=list(annotations_list[0]['annotation'].keys()))

    for annotation in annotations_list:
        # skip data which don't have annotations and not in study_level
        if (annotation['annotation']['x'] != -1) & (annotation['annotation']['study_level'] == 'No'):
            annotation_df = annotation_df.append(
                annotation['annotation'], ignore_index=True)

    # Get Annotations from AnnotatedSliceDataset
    for fname in tqdm(annotation_df['fname'].unique()):
        final_results = compare_results(
            fname, data_path, recon_path, save_path, annotation_df, accelerations)
        output_path = os.path.join(save_path, 'output.csv')
        if final_results is not None:
            final_results.to_csv(output_path, mode='a',
                                 header=not os.path.exists(output_path))
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

    # class Args:
    #     data_path = Path(
    #         '/Volumes/Medical Imaging Data Storage/fastMRI/fastmriplus_brain/multicoil_val_annotated_sub/raw_data/T1')
    #     recon_path = Path(
    #         '/Volumes/Medical Imaging Data Storage/fastMRI/fastmriplus_brain/multicoil_val_annotated_sub/reconstruction/reconstruction/T1/acc2')
    #     save_path = Path(
    #         '/Volumes/Medical Imaging Data Storage/fastMRI/fastmriplus_brain/multicoil_val_annotated_sub/output/T1/acc2')

    args = parser.parse_args()
    # args = Args()

    main(args)
