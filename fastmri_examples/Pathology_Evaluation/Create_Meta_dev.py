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

def intersection_over_union(gt, pred):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt[0], pred[0])
    yA = max(gt[1], pred[1])
    xB = min(gt[2], pred[2])
    yB = min(gt[3], pred[3])
    # if there is no overlap between predicted and ground-truth box
    if xB < xA or yB < yA:
        return 0.0
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    boxBArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def fix_bounding_box(file_name, data_path, recon_path, mode, annotation_abnormal, annotation_df, annotation_final):
    # read annotation
    annotations_abnormal_sub = annotation_abnormal[annotation_abnormal['fname'] == file_name]
    annotations_df_sub = annotation_df[annotation_df['fname'] == file_name]
    file_path = Path(data_path) / f"{file_name}.h5"
    img_path = Path(recon_path) / 'reconstructions' / f"{file_name}.h5"
    if os.path.exists(img_path):
        print("exist:", img_path)
        for slice_choice in annotations_df_sub['slice'].unique():
            print("slice_choice:", slice_choice)
            annotations_df = annotations_df_sub[annotations_df_sub['slice'] == slice_choice]
            annotations_abnormal = annotations_abnormal_sub[annotations_abnormal_sub['slice'] == slice_choice]
            
            # read data
            # gt / gt_re
            hf_gt = h5py.File(file_path, 'r')
            gt_recon = hf_gt['reconstruction_rss'][:]
            gt_recon = gt_recon[:, ::-1, :]

            if (gt_recon[slice_choice, :, :].shape[0] == 320) and (gt_recon[slice_choice, :, :].shape[1] == 320):
                pass
            else:
                print(
                    f"file_name:{file_name}, slice:{slice_choice}, img_size is not standard")

            img = gt_recon[slice_choice, :, :]
            annotations_df = annotations_df.reset_index(drop=True)
            
            for index_id, random_row in annotations_df.iterrows():
                print(index_id)
                index_df = annotation_final.loc[(annotation_df['fname'] == file_name) & (annotation_df['slice'] == slice_choice)].index[index_id]
<<<<<<< HEAD
                
=======
>>>>>>> b53d66745ecbd68d66e7ae0f106a006e0365aff1
                if mode == "random":
                    abnormal_row = annotations_abnormal.iloc[index_id,:]
                    change_bounding_box, large_bounding_box, x0, y0 = checkBoundingBox_random(img, abnormal_row, random_row, iou_threshold=0, frac_threshold = 0.20, attempts=50)
                
                elif mode == "normal":
                    change_bounding_box, large_bounding_box, x0, y0 = checkBoundingBox_normal(img, random_row, attempts=50)

                # record the large bounding box
                if large_bounding_box:
                    annotation_final.loc[index_df,'Large'] = 1
                # record changes of bounding box
                if change_bounding_box:
                    print("change bounding box: __________")
                    annotation_final.loc[index_df,'x'] = x0
                    annotation_final.loc[index_df,'y'] = y0
    else:
        print(f"{file_name} no reconstruction files found")

    return annotation_final

def checkBoundingBox_normal(img, random_row, iou_threshold=0, frac_threshold = 0.20, attempts=50):
    
    fname, slice_choice, study_level, x0, y0, w, h, label_txt, _ = random_row
    x0, y0, x1, y1, label_txt = int(x0), int(y0), int(x0+w), int(y0+h), str(label_txt)
    x_max = img.shape[1]
    y_max = img.shape[0]

    average = np.average(img.flatten())
    change_bounding_box = False
    large_bounding_box = False
    threshold = 40
    threshold2 = 10

    # check initial bounding boxes inside the global images
    k = 0
    while k <= 100:
        if (x1>(x_max-threshold2)) or (y1>(y_max-threshold2)):
            change_bounding_box = True
            x0 = random.randint(threshold,min(x_max-threshold,x_max-w)) if (x_max-w) > threshold else random.randint(min(x_max-threshold,x_max-w),threshold)
            y0 = random.randint(threshold,min(y_max-threshold,y_max-h)) if (y_max-h) > threshold else random.randint(min(y_max-threshold,y_max-h),threshold)
            x1 = int(x0 + w)
            y1 = int(y0 + h)
        else:
            break
        k+=1

    bounding_box = np.array(img[y0:y1, x0:x1])
    zero_number = np.sum((bounding_box.flatten() < average))
    rows, cols = bounding_box.shape
    total_number = rows * cols
    min_frac = 1
    i=0
    x0_memo, y0_memo = 0.0, 0.0
    
    while i <= attempts:
        if (zero_number / total_number > frac_threshold) or (x1>(x_max+threshold2)) or (y1>(y_max+threshold2)):
            change_bounding_box = True
            # print("fraction:", zero_number / total_number)
            x0 = random.randint(threshold,min(x_max-threshold,x_max-w)) if (x_max-w) > threshold else random.randint(min(x_max-threshold,x_max-w),threshold)
            y0 = random.randint(threshold,min(y_max-threshold,y_max-h)) if (y_max-h) > threshold else random.randint(min(y_max-threshold,y_max-h),threshold)
            x1 = int(x0 + w)
            y1 = int(y0 + h)
            
            # update fraction
            bounding_box = np.array(img[y0:y1, x0:x1])
            zero_number = np.sum((bounding_box.flatten() < average))
            frac = zero_number / total_number

            # save the min frac
            if frac < min_frac:
                min_frac = frac
                x0_memo, y0_memo = x0, y0
        else:
            break
        i+=1
        if i > attempts-1:
            frac_threshold += 0.05
            print(f"update frac_threshold as {frac_threshold}")
            i=0
            
            if frac_threshold >= 0.5:
                large_bounding_box = True
                print("Bounding Box is too large to avoid overlap")
                break
            
    if change_bounding_box:
        x0, y0 = x0_memo, y0_memo
    print("final_selection:", x0, y0)
    
    random_row = fname, slice_choice, study_level, x0, y0, w, h, label_txt

    return change_bounding_box, large_bounding_box, x0, y0

def checkBoundingBox_random(img, abnormal_row, random_row, iou_threshold=0, frac_threshold = 0.20, attempts=50):
    fname, slice_choice, study_level, x0, y0, w, h, label_txt, _ = random_row
    fname, slice_choice, study_level, x0_, y0_, w_, h_, label_txt = abnormal_row
    x0, y0, x1, y1, label_txt = int(x0), int(y0), int(x0+w), int(y0+h), str(label_txt)
    x0_, y0_, x1_, y1_, label_txt_ = int(x0_), int(y0_), int(x0_+w_), int(y0_+h_), str(label_txt)
    x_max = img.shape[1]
    y_max = img.shape[0]
    
    pred = (int(x0), int(y0), int(x0+w), int(y0+h))
    gt = (int(x0_), int(y0_), int(x0_+w_), int(y0_+h_))
    iou = intersection_over_union(gt,pred)
    
    average = np.average(img.flatten())
    change_bounding_box = False
    large_bounding_box = False
    threshold = 40
    threshold2 = 10
    x0_memo, y0_memo = x0, y0

    k = 0
    while k <= 100:
        if (x1>(x_max-threshold2)) or (y1>(y_max-threshold2)):
            change_bounding_box = True
            x0 = random.randint(threshold,min(x_max-threshold,x_max-w)) if (x_max-w) > threshold else random.randint(min(x_max-threshold,x_max-w),threshold)
            y0 = random.randint(threshold,min(y_max-threshold,y_max-h)) if (y_max-h) > threshold else random.randint(min(y_max-threshold,y_max-h),threshold)
            x1 = int(x0 + w)
            y1 = int(y0 + h)
        else:
            break
        k+=1

    bounding_box = np.array(img[y0:y1, x0:x1])
    zero_number = np.sum((bounding_box.flatten() < average))
    rows, cols = bounding_box.shape
    total_number = rows * cols
    # min_frac = 1
    i=0
    
    while (i <= attempts) and ((zero_number / total_number > frac_threshold) or (iou > iou_threshold) or (x1>(x_max-threshold2)) or (y1>(y_max-threshold2))):
        # if (zero_number / total_number > frac_threshold) or (iou > iou_threshold) or (x1>(x_max-threshold2)) or (y1>(y_max-threshold2)):
        change_bounding_box = True
        # print("fraction:", zero_number / total_number)
        x0 = random.randint(threshold,min(x_max-threshold,x_max-w)) if (x_max-w) > threshold else random.randint(min(x_max-threshold,x_max-w),threshold)
        y0 = random.randint(threshold,min(y_max-threshold,y_max-h)) if (y_max-h) > threshold else random.randint(min(y_max-threshold,y_max-h),threshold)
        x1 = int(x0 + w)
        y1 = int(y0 + h)
        
        # update fraction
        bounding_box = np.array(img[y0:y1, x0:x1])
        zero_number = np.sum((bounding_box.flatten() < average))
        frac = zero_number / total_number
        
        # update iou
        pred = (x0,y0,x1,y1)
        iou = intersection_over_union(gt,pred)

        # save the min frac
        # if (frac < min_frac) and (iou==0) :
        #     min_frac = frac
        x0_memo, y0_memo = x0, y0
            
        # else:
        #     break
        
        i+=1
        
        if i > attempts-1:
            frac_threshold += 0.05
            print(f"update frac_threshold as {frac_threshold}")
            i=0
            
            if frac_threshold >= 0.5:
                large_bounding_box = True
                print("Bounding Box is too large to avoid overlap")
                x0, y0 = 100, 100
                break
            
    if change_bounding_box:
        x0, y0 = x0_memo, y0_memo
    print("final_selection:", x0, y0)
    
    random_row = fname, slice_choice, study_level, x0, y0, w, h, label_txt

    return change_bounding_box, large_bounding_box, x0, y0

def random_select(row, location_type, threshold=60):
    if location_type == 'x':
        point, delta = row['x'], row['width']
    elif location_type == 'y':
        point, delta = row['y'], row['height']
    # y,height = row['y'], row['height']
    if row['study_level'] == 'No':
        if (320-delta) > threshold:
            return random.randint(threshold,min(320-threshold,320-delta))
        else:
            return random.randint(min(320-threshold,320-delta),threshold)

def random_select_normal(row, abnormal_df):
    fname = row['fname']
    df_f = abnormal_df[(abnormal_df['fname']==fname)]
    if len(df_f) > 0:
        row_f = df_f.sample(n=1)
        # return pd.Series([float(row_f['x'].values), float(row_f['y'].values), float(row_f['width'].values), float(row_f['height'].values)])
    else:
        row_f = abnormal_df.sample(n=1)
        # return pd.Series([140,141,37,45])
    return pd.Series(['No',float(row_f['x'].values), float(row_f['y'].values), float(row_f['width'].values), float(row_f['height'].values), '-1'])

def main(args):
    # Ground Truth
    data_path = args.data_path
    # Reconstruction
    accelerations = args.accelerations[0]
    recon_path = args.recon_path / str(accelerations)
    
    # datalodaer
    annotations_list = pd.DataFrame(fastmri.data.mri_data.AnnotatedSliceDataset(
        data_path, "multicoil", "brain", "all", annotation_version="main").annotated_examples)[2].values.tolist()
    annotation_abnormal = pd.DataFrame(columns=list(annotations_list[0]['annotation'].keys()))
    annotation_normal = pd.DataFrame(columns=list(annotations_list[0]['annotation'].keys()))
    annotation_study = pd.DataFrame(columns=list(annotations_list[0]['annotation'].keys()))
    annotation_random = pd.DataFrame(columns=list(annotations_list[0]['annotation'].keys()))
    brain_file_list = list(pd.read_csv(Path(os.getcwd(),'.annotation_cache','brain_file_list.csv'), header=None).iloc[:,0])

    for annotation in annotations_list:
        # only need the data has been checked in fastMRI+ (1001 samples)
        if annotation['annotation']['fname'] in brain_file_list:
            # create abnormal list
            if (annotation['annotation']['x'] != -1) and (annotation['annotation']['study_level'] == 'No'):
                annotation_abnormal = annotation_abnormal.append(annotation['annotation'], ignore_index=True)
            # create global list
            elif annotation['annotation']['study_level'] == 'Yes':
                annotation_study = annotation_study.append(annotation['annotation'], ignore_index=True)
            # create normal list
            else:
                annotation_normal = annotation_normal.append(annotation['annotation'], ignore_index=True)

    # save annotation_abnormal to csv
    annotation_abnormal.to_csv(Path(os.getcwd(),'.annotation_cache','brainmain_abnormal.csv'), index=False)
    annotation_random = copy.deepcopy(annotation_abnormal)

    # randomly select annotations from abnormal slices with the same fname
    np.random.seed(2022)
    annotation_normal[['study_level','x','y','width','height','label']] = annotation_normal.apply(lambda row: random_select_normal(row, annotation_abnormal), axis=1)
    annotation_normal.loc[:, 'Large'] = 0
    annotation_final = copy.deepcopy(annotation_normal)
    # Check annotations in normal samples
    for fname in tqdm(annotation_normal['fname'].unique()):
        annotation_final = fix_bounding_box(fname, data_path, recon_path, "normal", annotation_abnormal, annotation_normal, annotation_final)
    # save annotation_normal to csv
    annotation_final.to_csv(Path(os.getcwd(),'.annotation_cache','brainmain_normal.csv'), index=False)
    
    # randomly change the starting point of x and y for abnormal slices to create random
    np.random.seed(2022)
    annotation_random['x'] = annotation_random.apply(lambda row: random_select(row, 'x', 100), axis=1)
    annotation_random['y'] = annotation_random.apply(lambda row: random_select(row, 'y', 80), axis=1)
    annotation_random.loc[:, 'Large'] = 0
    annotation_final = copy.deepcopy(annotation_random)
    # Check annotations in random samples
    for fname in tqdm(annotation_random['fname'].unique()):
        annotation_final = fix_bounding_box(fname, data_path, recon_path, "random", annotation_abnormal, annotation_random, annotation_final)
    # save annotation_random to csv
    annotation_final.to_csv(Path(os.getcwd(),'.annotation_cache','brainmain_random.csv'), index=False)
    
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
        "--accelerations",
        nargs="+",
        default=4,
        type=int,
        help="Acceleration rates to use for masks",
    )

    args = parser.parse_args()

    main(args)
