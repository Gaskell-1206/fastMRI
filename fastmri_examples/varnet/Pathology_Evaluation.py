from operator import index
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import os
import fastmri
from fastmri.data import transforms as T
import h5py
from PIL import ImageDraw, Image
import cv2

from runstats import Statistics
try:
    from skimage.measure import compare_psnr
    peak_signal_noise_ratio = compare_psnr
except ImportError:
    from skimage.metrics import peak_signal_noise_ratio

from skimage.metrics import structural_similarity


def Evaluation_Metrics(gt,pred,all_slice=True):
    _mse = mse(gt,pred)
    _nmse = nmse(gt,pred)
    _psnr = psnr(gt,pred)
    _ssim = ssim(gt,pred) if all_slice is True else ssim_local(gt,pred)
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


def save_fig(img, annotations, save_path, zoom_factor=2):
    # global image + bounding boxes

    annotations = annotations.reset_index(drop=True)
    regional_image_list = []
    for index, row in annotations.iterrows():
        fname, slice_choice, study_level, x0, y0, w, h, label_txt = row
        w = 7 if w < 7 else w
        h = 7 if h < 7 else w
        x0, y0, x1, y1 = int(x0), int(y0), int(x0+w), int(y0+h)
        sub_path = os.path.join(save_path,fname)
        if(os.path.exists(sub_path)) == False:
            os.makedirs(sub_path)

        # plot bounding box
        arrimg = np.squeeze(img)
        image_2d_scaled = (np.maximum(arrimg, 0) / arrimg.max()) * 255.0
        image_2d_scaled = Image.fromarray(np.uint8(image_2d_scaled))
        plotted_image = ImageDraw.Draw(image_2d_scaled)
        plotted_image.rectangle(((x0, y0), (x1, y1)), outline="white")
        plotted_image.text((x0, max(0, y0 - 10)), label_txt, fill="white")
        image_2d_scaled = np.array(image_2d_scaled)

        # annotated image
        regional_image = np.array(img[x0:x1, y0:y1])
        regional_image_list.append(regional_image)
        cv2.imwrite(os.path.join(sub_path,f'{slice_choice}_{study_level}_label{index}_regional_image.png'),regional_image)

        # zoom in image
        region_size = 40
        x_left = x0 if x0 < region_size else region_size
        x_right = (320 - x1) if (320 - x1) < region_size else region_size
        y_top = y0 if y0 < region_size else region_size
        y_bottom = (320 - y1) if (320 - y1) < region_size else region_size
        img_sub = image_2d_scaled[x0-x_left:x1+x_right,y0-y_top:y1+y_bottom]
        zoom_in_image = cv2_clipped_zoom(img_sub,zoom_factor=zoom_factor)
        cv2.imwrite(os.path.join(sub_path,f'{slice_choice}_{study_level}_label{index}_zoom_in_image.png'),zoom_in_image)

    cv2.imwrite(os.path.join(sub_path,f'{slice_choice}_{study_level}_whole_image.png'),image_2d_scaled)

    return regional_image_list


def compare_results(file_path, recon_path, save_path, annotation_df, acc_rate):
    final_results_df = pd.DataFrame(
        columns=['Sample', 'Slice', 'Annotation#', 'Level', 'Acc', 'mse', 'nmse', 'psnr', 'ssim'])
    # recon_path = Path('/gpfs/home/sc9295/Projects/fastMRI/pathology_eval/pretrained_varnet_inference_results')
    file_name = os.path.basename(file_path)

    # read annotation
    annotations_sub = annotation_df[annotation_df['fname'] == file_name.split('.')[0]]

    for slice_choice in annotations_sub['slice'].unique():
        annotation = annotations_sub[annotations_sub['slice'] == slice_choice]

        # read data
        # gt / gt_re
        hf_gt = h5py.File(file_path, 'r')
        gt_recon = hf_gt['reconstruction_rss'][:]

        # accX image
        img_path = recon_path / 'reconstructions' / file_name
        hf_accX = h5py.File(img_path, 'r')
        accX_recon = hf_accX['reconstruction'][:]

        # Gloabl
        gt_region = save_fig(gt_recon[slice_choice,:,:], annotation, save_path, zoom_factor=2)
        for i in range(len(gt_region)):
            accX_results = Evaluation_Metrics(gt_recon[slice_choice, :, :], accX_recon[slice_choice, :, :], all_slice=False)
            # Save Results
            results_list = [file_name.split('.')[0], slice_choice, i, 'Global', acc_rate,
                            accX_results[0], accX_results[1], accX_results[2], accX_results[3]]
            final_results_df.loc[len(final_results_df)] = results_list
            #             print(f"acc rate {acc}:",accX_results)

            # Region
            accX_region = save_fig(accX_recon[slice_choice, :, :], annotation, save_path, zoom_factor=2)
        
            accX_re_results = Evaluation_Metrics(gt_region[i], accX_region[i], all_slice=False)
            # Save Restuls
            results_list = [file_name.split('.')[0], slice_choice, i, 'Region', acc_rate,
                            accX_re_results[0], accX_re_results[1], accX_re_results[2], accX_re_results[3]]
            final_results_df.loc[len(final_results_df)] = results_list
            #             print(f"acc rate {acc} regional:",accX_re_results)

        return final_results_df


def main():
    # Ground Truth
    dire_path = Path('/Volumes/Medical Imaging Data Storage/fastMRI/fastmriplus_brain/multicoil_train_sub_sub')
    file_list = list(dire_path.glob('**/*.h5'))
    recon_path = Path('/Volumes/Medical Imaging Data Storage/fastMRI/reconstruction_results')
    save_path = Path('/Volumes/Medical Imaging Data Storage/fastMRI/temp')

    annotations_list = pd.DataFrame(fastmri.data.mri_data.AnnotatedSliceDataset(
        dire_path, "multicoil", "brain", "all", annotation_version="main").annotated_examples)[2].values.tolist()
    annotation_df = pd.DataFrame(columns=list(annotations_list[0]['annotation'].keys()))
    for annotation in annotations_list:
        # skip data without annotations
        if annotation['annotation']['x'] != -1:
            annotation_df = annotation_df.append(
                annotation['annotation'], ignore_index=True)

    # Get Annotations from AnnotatedSliceDataset
    for file in file_list:        
        final_results = compare_results(file, recon_path, save_path, annotation_df, 4)
        output_path='/Volumes/Medical Imaging Data Storage/fastMRI/temp/output.csv'
        final_results.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


if __name__ == "__main__":
    main()
