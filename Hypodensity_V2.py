import itk
import argparse
import nibabel as nib
import scipy as ndi
from skimage import morphology , measure
from scipy.ndimage import generate_binary_structure, binary_closing, binary_fill_holes , binary_dilation , binary_erosion
import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude , median_filter
from skimage.measure import label, regionprops
import cv2
import openpyxl

def calculate_volume(image_path):
    image = sitk.ReadImage(image_path)
    image = sitk.Cast(image, sitk.sitkUInt8)
    statistics = sitk.LabelStatisticsImageFilter()
    statistics.Execute(image, image)
    
    count_label_1 = statistics.GetCount(1)
    
    volume_mm3 = count_label_1 * np.prod(image.GetSpacing())

    volume_cm3 = volume_mm3 / 1000
    
    return volume_cm3

def apply_mask_threshold(image_path, output_path):
    outside_value = -1024
    lower_threshold = -10
    upper_threshold = 60


    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

  
    binary_mask = (image_array > lower_threshold) & (image_array <= upper_threshold)

 
    labeled_mask = label(binary_mask)

   
    largest_component_index = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
    largest_component_mask = labeled_mask == largest_component_index

 
    
    result_image = np.where(largest_component_mask, image_array, outside_value)

 
    result_image_sitk = sitk.GetImageFromArray(result_image)
    result_image_sitk.SetSpacing(image.GetSpacing())
    result_image_sitk.SetOrigin(image.GetOrigin())
    result_image_sitk.SetDirection(image.GetDirection())

    output_file_path = os.path.join(output_path, "skull_stripped.nii.gz")
    sitk.WriteImage(result_image_sitk, output_file_path)

    return result_image_sitk

def convert_mha_to_nii(input_path, output_path):
    if input_path.endswith('.mha'):
        img = sitk.ReadImage(input_path)
        sitk.WriteImage(img, output_path)
    else:
        print("mha format")

def segment_brain(input_path, output_path):
    def reg_grow(img, seed, lt, ut, st):
        mask = np.zeros_like(img, dtype=bool)
        mask[seed] = True
        conn = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])
        queue = [seed]

        while queue:
            current_point = queue.pop(0)
            neighbors = get_neighbors(current_point, conn, img.shape)

            for neighbor in neighbors:
                if not mask[neighbor]:
                    if lt <= img[neighbor] <= ut:
                        mask[neighbor] = True
                        queue.append(neighbor)

            if img[current_point] > st:
                break

        return mask

    def get_neighbors(point, conn, shape):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = (point[0] + i, point[1] + j)
                if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
                    neighbors.append(neighbor)
        return neighbors

    def gen_seeds(center, offsets):
        seed_points = []
        for offset in offsets:
            seed_points.append((center[0] + offset, center[1]))
        return seed_points

    img = nib.load(input_path)
    data = img.get_fdata()

    center = (data.shape[0] // 2, data.shape[1] // 2)
    offsets = [-30, -15, 0, 15, 30]
    seed_points = gen_seeds(center, offsets)

    lt = -15
    ut = 80
    st = 90

    seg_volume = np.zeros_like(data, dtype=bool)

    for seed_point in seed_points:
        seg_mask = np.zeros_like(data, dtype=bool)
        for i in range(data.shape[2]):
            seg_mask[:, :, i] = reg_grow(data[:, :, i], seed_point, lt, ut, st)
        seg_volume |= seg_mask

    lbl_volume = label(seg_volume, connectivity=2)
    props = regionprops(lbl_volume)
    areas = [prop.area for prop in props]
    largest_label = np.argmax(areas) + 1
    largest_label_volume = lbl_volume == largest_label

    data[~largest_label_volume] = -1024

    out_img = nib.Nifti1Image(data.astype(np.float32), img.affine)
    saving_path = os.path.join(output_path , "brain_extracted.nii.gz")
    nib.save(out_img, saving_path)

# def generate_probability_ventricle_mask(brain_path, output_mask_path):
#     brain = nib.load(brain_path)
#     brain_data = brain.get_fdata()
#     brain_affine = brain.affine

#     csf_mask = np.logical_and(brain_data >= 1, brain_data <= 15)
#     labels, num_labels = measure.label(csf_mask, return_num=True)
#     largest_label = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1)
#     filled_csf_mask = ndimage.binary_fill_holes(largest_label)

#     dilated_csf_mask = ndimage.binary_dilation(filled_csf_mask)

#     removed_brain = np.where(dilated_csf_mask, -1024, brain_data)
#     dilated_csf_img = nib.Nifti1Image(dilated_csf_mask.astype(np.uint8), brain_affine)
#     csf_probability_mask_path = os.path.join(output_mask_path , "csf_probability_mask.nii.gz")
#     nib.save(dilated_csf_img, csf_probability_mask_path)
    
#     removed_ct_img = nib.Nifti1Image(removed_brain, brain_affine)
#     csf_removed_image_path = os.path.join(output_mask_path , "csf_removed.nii.gz")
#     nib.save(removed_ct_img, csf_removed_image_path)

#     return csf_removed_image_path
    


    
def apply_smoothing(image_path, output_mask_path):
    img = itk.imread(image_path, itk.F)

    bilateral_filter = itk.BilateralImageFilter.New(img)
    
    bilateral_filter.SetDomainSigma([3, 3, 3])  
    bilateral_filter.SetRangeSigma(50.0)
    bilateral_filter.Update()
    smoothed_img = bilateral_filter.GetOutput()

    output_file_path = os.path.join(output_mask_path, "smoothed.nii.gz")
    itk.imwrite(smoothed_img, output_file_path)


def flip_and_register(smoothened_image_path, output_path , param_file):
    csf_removed_image = itk.imread(smoothened_image_path, itk.F)
    print(csf_removed_image.shape)
    flip = itk.FlipImageFilter.New(csf_removed_image)
    flip.SetFlipAxes([True, False, False])
    flip.Update()
    mirror_image = flip.GetOutput()

    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(param_file)
   

    parameter_object.SetParameter('FinalBSplineInterpolationOrder', '0')

    result_image, result_transform_parameters = itk.elastix_registration_method(
        csf_removed_image, mirror_image,
        parameter_object=parameter_object,
        log_to_console=True)
    
    itk.imwrite(result_image, output_path)
    return result_image , result_transform_parameters


def generate_combined_masks(smoothed_img_path, mirror_img_path, output_dir):
    smoothed_img = nib.load(smoothed_img_path)
    mirror_img = nib.load(mirror_img_path)

    smoothed_data = smoothed_img.get_fdata()
    mirror_data = mirror_img.get_fdata()

    ratio_map = np.divide(smoothed_data, mirror_data, out=np.zeros_like(smoothed_data), where=mirror_data != 0)
    ratio_map[np.isnan(ratio_map)] = 0
    ratio_map[np.isinf(ratio_map)] = 0

    ratio_threshold = 0.95
    ratio_condition = (ratio_map >= 0.1) & (ratio_map <= ratio_threshold)

    difference_conditions = [(-12, -3)]

    for idx, (lower_diff, upper_diff) in enumerate(difference_conditions):
        difference_condition = (smoothed_data - mirror_data >= lower_diff) & (smoothed_data - mirror_data <= upper_diff)

        combined_mask_weighted = ratio_map * (ratio_condition & difference_condition)

        combined_mask_img = nib.Nifti1Image(combined_mask_weighted, smoothed_img.affine)
        output_path_combined = os.path.join(output_dir, f"combined_mask_weighted_condition_{idx + 1}.nii.gz")
        nib.save(combined_mask_img, output_path_combined)

        upper_threshold = 0.95
        lower_threshold = 0.5
        binary_mask = np.logical_and(combined_mask_weighted >= lower_threshold, combined_mask_weighted <= upper_threshold)

        binary_mask_img = nib.Nifti1Image(binary_mask.astype(np.float32), smoothed_img.affine)
        output_path_binary = os.path.join(output_dir, f"binary_mask_thresholded_condition_{idx + 1}.nii.gz")
        nib.save(binary_mask_img, output_path_binary)


def remove_false_positives(brain_extracted_image_path, hypo_mask_path, csf_mask_path, final_saving_path):
    brain_img = nib.load(brain_extracted_image_path)
    brain_data = brain_img.get_fdata()
    brain_affine = brain_img.affine

    hypo_img = nib.load(hypo_mask_path)
    hypo_data = hypo_img.get_fdata()

    csf_img = nib.load(csf_mask_path)
    csf_data = csf_img.get_fdata()

    modified_hypo_data = hypo_data - csf_data
    modified_hypo_data[modified_hypo_data < 0] = 0

    modified_hypo_img = nib.Nifti1Image(modified_hypo_data, brain_affine)

    nib.save(modified_hypo_img, final_saving_path)

def calculate_csf_parameters(histogram, mc, f):
    csf_mean = mc
    csf_std = (mc - f) / np.sqrt(2 * np.log(2))
    return csf_mean, csf_std

def segment_csf_with_subtraction(input_image_path, output_mask_path):
    input_image = nib.load(input_image_path)
    image_array = input_image.get_fdata()

    threshold_range = (5, 200)
    spatial_smoothing_kernel_size = (5, 5, 5)
    poisson_error_threshold = 0.1
    k_scaling_factor = 2

    histogram, _ = np.histogram(image_array.flatten(), bins=256, range=threshold_range)

    initial_threshold = 5
    i = initial_threshold
    s_i_prev = 0

    while True:
        thresholded_image = image_array > i
        smoothed_image = median_filter(thresholded_image, size=spatial_smoothing_kernel_size)
        labeled_array, num_features = morphology.label(smoothed_image, return_num=True)

        largest_connected_component = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1

        largest_connected_component_mask = labeled_array == largest_connected_component

        hist, _ = np.histogram(image_array[largest_connected_component_mask], bins=256, range=threshold_range)

        h_mx = np.max(hist)
        h_last = hist[-1]

        s_i = (h_mx - h_last) / np.sqrt(h_mx + h_last)

        if s_i <= s_i_prev or s_i < poisson_error_threshold:
            break

        s_i_prev = s_i
        i += 1

    premax_region = image_array[largest_connected_component_mask]
    mc = np.argmax(hist)
    f = np.argmax(hist > np.max(hist) / 2)

    csf_mean, csf_std = calculate_csf_parameters(hist, mc, f)

    csf_lower_threshold = csf_mean - k_scaling_factor * csf_std
    csf_upper_threshold = csf_mean + k_scaling_factor * csf_std

    csf_segmentation_mask = np.logical_and(premax_region >= csf_lower_threshold, premax_region <= csf_upper_threshold)

    segmentation_array = np.zeros_like(image_array)
    segmentation_array[largest_connected_component_mask] = csf_segmentation_mask.astype(int)

    hu_mask = np.logical_and(image_array >= -10, image_array <= 40).astype(int)

    final_mask = np.subtract(segmentation_array, hu_mask)
    final_mask[final_mask != -1] = 0
    final_mask[final_mask == -1] = 1

    
    labeled_mask= label(final_mask)
    num_features = np.max(labeled_mask)
    for n in range(1, num_features + 1):
        component_size = np.sum(labeled_mask == n)
        if component_size < 400:
            final_mask[labeled_mask == n] = 0

    
    csf_mask = nib.Nifti1Image(final_mask, input_image.affine)
    nib.save(csf_mask, output_mask_path)

def csf_prediction_mask(brain_scan_path,skull_stripped_path,csf_mask_path,output_path,output_csf_path):

    def load_image(file_path):
        return nib.load(file_path).get_fdata()

    def dilate_slice(slice_2d):
        structure = generate_binary_structure(2, 3)
        return binary_dilation(slice_2d, structure=structure)

    def find_center(component):
        return tuple(np.mean(np.nonzero(component), axis=1).astype(int))

    def region_growing(image, seed, intensity_range):
        grown_region = np.zeros_like(image, dtype=bool)
        visited = np.zeros_like(image, dtype=bool)
        stack = [seed]

        while stack:
            x, y, z = stack.pop()
            if not (0 <= x < image.shape[0] and 0 <= y < image.shape[1] and 0 <= z < image.shape[2]):
                continue
            if visited[x, y, z]:
                continue

            intensity = image[x, y, z]
            if intensity_range[0] <= intensity <= intensity_range[1]:
                grown_region[x, y, z] = True
                visited[x, y, z] = True

                stack.append((x + 1, y, z))
                stack.append((x - 1, y, z))
                stack.append((x, y + 1, z))
                stack.append((x, y - 1, z))
                stack.append((x, y, z + 1))
                stack.append((x, y, z - 1))

        return grown_region
    threshold = 400
    intensity_range = (1, 20)
    
    brain_scan = load_image(brain_scan_path)

    skull_removed = load_image(skull_stripped_path)  
    pre_prediction_mask = load_image(csf_mask_path) 
    csf_mask = np.zeros_like(brain_scan, dtype=np.uint8)

    for z in range(brain_scan.shape[2]):
        thresholded_image = np.where((brain_scan[:, :, z] >= 1) & (brain_scan[:, :, z] <= 10), 1, 0).astype(np.uint8)
        filled_image = binary_fill_holes(thresholded_image)
        dilated_slice = dilate_slice(filled_image)
        csf_mask[:, :, z] = dilated_slice

    labeled_array_csf = label(csf_mask)
    num_features_csf = np.max(labeled_array_csf)

    for i in range(1, num_features_csf + 1):
        component = labeled_array_csf == i
        voxel_count = np.sum(component)
        if voxel_count < threshold:
            csf_mask[component] = 0
        else:
            center = find_center(component)
            grown_region = region_growing(brain_scan, center, intensity_range)
            csf_mask[grown_region] = 1
    
    thresholded_brain_scan = np.where((brain_scan >= 55) & (brain_scan <= 100), 1, 0).astype(np.uint8)
    brain_affine = nib.load(brain_scan_path).affine
    final_mask = (csf_mask > 0).astype(np.uint8) + (thresholded_brain_scan > 0).astype(np.uint8) + (pre_prediction_mask > 0).astype(np.uint8)
    final_mask[final_mask > 1] = 1
    final_mask[final_mask < 0] = 0
    csf_removed = skull_removed - final_mask
    csf_removed[final_mask == 1] = -1024
    nib.save(nib.Nifti1Image(csf_removed,brain_affine),output_csf_path)
    nib.save(nib.Nifti1Image(final_mask,brain_affine ), output_path)

def find_largest_region(input_paths, output_dir):
     
        ratio_map_weighted = nib.load(input_paths)
        binary_mask = ratio_map_weighted.get_fdata()

        label_image = label(binary_mask)
        regions = regionprops(label_image)

        largest_region_index = np.argmax([region.area for region in regions])
        largest_region_mask = label_image == (largest_region_index + 1)
        
        output_path = f"{output_dir}/largest_region_mask.nii.gz"
        largest_region_mask_img = nib.Nifti1Image(largest_region_mask.astype(np.float32), ratio_map_weighted.affine)
        nib.save(largest_region_mask_img, output_path)

def process_and_save_mask(input_path, output_path, closing_radius=1):
    nifti_image = nib.load(input_path)
    binary_mask = nifti_image.get_fdata()
    
    structuring_element = generate_binary_structure(3, 1)
    closed_mask = binary_closing(binary_mask, structure=structuring_element, iterations=closing_radius)

    processed_nifti = nib.Nifti1Image(closed_mask.astype(np.uint8), nifti_image.affine)
    nib.save(processed_nifti, output_path)
  
def main():
    parser = argparse.ArgumentParser(description="Image Registration and Mask Transformation Script")
   
    parser.add_argument("--tbc_image_path", required=True, help="Path to the moving (tbc) image")
    parser.add_argument("--parameter_file", required=True, help="Path to the parameter file")
    parser.add_argument("--output_mask_path", required=True, help="Path to save the output mask")
    parser.add_argument("--ground_truth", required=True, help="Path to skull stripped base image")
    parser.add_argument("--case_id",required=True,help="Enter the case ID related to the patient")
    args = parser.parse_args()


    if args.tbc_image_path.endswith('.mha'):
        input_image_nii_path = os.path.join(args.output_mask_path, "input_image.nii.gz")
        convert_mha_to_nii(args.tbc_image_path, input_image_nii_path)
        tbc_image_path = input_image_nii_path
    else:           
        tbc_image_path = args.tbc_image_path

    segment_brain(tbc_image_path, args.output_mask_path)
    apply_mask_threshold(args.tbc_image_path,args.output_mask_path)
    # skull_stripped_path = '/home/sarthak/Desktop/Stuff/Image_reg/CT/tp2/Hypo/skulll/results/P228_ncct_1_SS_0.01.nii.gz'
    skull_stripped_path = os.path.join(args.output_mask_path, "skull_stripped.nii.gz")
    brain_extracted_image_path = os.path.join(args.output_mask_path , "brain_extracted.nii.gz")
    output_mask_path = os.path.join(args.output_mask_path,'csf_mask.nii.gz')
    output_csf_path = os.path.join(args.output_mask_path,'csf_removed.nii.gz')
    output_csf_histo_path = os.path.join(args.output_mask_path,'csf_histo.nii.gz')
    segment_csf_with_subtraction(brain_extracted_image_path, output_csf_histo_path)
    csf_prediction_mask(brain_extracted_image_path, skull_stripped_path, output_csf_histo_path, output_mask_path , output_csf_path)
    # to_be_smoothen_path = generate_probability_ventricle_mask(skull_stripped_path , args.output_mask_path)
    apply_smoothing(output_csf_path, args.output_mask_path)

    smoothed_image_path = os.path.join(args.output_mask_path,"smoothed.nii.gz")
    smoothed_image_path_flipped = os.path.join(args.output_mask_path,"smoothed_flipped.nii.gz")
    image , parameters = flip_and_register(smoothed_image_path,smoothed_image_path_flipped , args.parameter_file)

    
    generate_combined_masks(smoothed_image_path, smoothed_image_path_flipped, args.output_mask_path)

    input_image_filename = "binary_mask_thresholded_condition_1.nii.gz"
    hypo_mask = os.path.join(args.output_mask_path, input_image_filename)
    brain_extracted_image_path = os.path.join(args.output_mask_path , "brain_extracted.nii.gz")
    final_saving_path = os.path.join(args.output_mask_path, "hypodensity_prediction_temp.nii.gz")
    # csf_prediction_mask(brain_extracted_image_path , hypo_mask , final_saving_path)
    remove_false_positives(brain_extracted_image_path, hypo_mask, output_mask_path, final_saving_path)
    find_largest_region(final_saving_path, args.output_mask_path)
    target_mask_files = []
    input_image_filenames_2 = ["largest_region_mask.nii.gz"]
    for input_filename in input_image_filenames_2:
        input_path = os.path.join(args.output_mask_path, input_filename)

        
        output_filename = f"{os.path.splitext(input_filename)[0]}_filled.nii.gz"
        filled_output_path = os.path.join(args.output_mask_path, output_filename)
        target_mask_files.append(input_path)
        target_mask_files.append(filled_output_path)
        
        process_and_save_mask(input_path, filled_output_path)
    final_prediction_save = os.path.join(args.output_mask_path, 'final_prediction.nii.gz')
    remove_false_positives(brain_extracted_image_path, filled_output_path, output_mask_path, final_prediction_save)
    
    dilated_largest_region_mask_volume = calculate_volume(final_prediction_save)
    ground_truth_mask_volume = calculate_volume(args.ground_truth)
    excel_file_path = '/home/sarthak/Desktop/Stuff/Image_reg/CT/tp2/Hypo/iteration_7/volume.xlsx'
    
    try:
        workbook = openpyxl.load_workbook(excel_file_path)
        sheet = workbook.active
    except FileNotFoundError:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Case ID", "Ground Truth Mask Volume", "Our Prediction Volume"])

    sheet.append([args.case_id, ground_truth_mask_volume, dilated_largest_region_mask_volume])

    workbook.save(excel_file_path)

if __name__ == "__main__":
    main()

