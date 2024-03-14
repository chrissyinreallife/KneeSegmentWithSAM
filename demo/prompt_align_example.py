import os
import numpy as np
import torch
import subprocess
import matplotlib.pyplot as plt
import cv2
import nibabel as nib 
import sys
import ast  
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import json
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, label, binary_fill_holes


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def standardize_intensity(image, new_min=0, new_max=3000):
    old_min, old_max = np.min(image), np.max(image)
    return (image - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def rescale_coords(coords, original_shape, new_shape):
    scale = [new_shape[i] / original_shape[i] for i in range(2)]
    return np.round(coords * scale).astype(np.int64)

def extract_specific_value_coords_and_label(data, label):
    specific_value = np.amax(data)
    #print(f"Specific value being used for label {label}: {specific_value}")
    coords_labels_list = []
    for slice_index in range(data.shape[2]):
        slice_data = data[:, :, slice_index]
        coords = np.column_stack(np.where(slice_data == specific_value))
        #print(f"Slice {slice_index}: Found {len(coords)} points with the specific value.")
        if coords.size > 0:
            
            filtered_coords = filter_similar_coords(coords, grid_size=(5, 5))
            #print(f"Slice {slice_index}: {len(filtered_coords)} points after filtering.")
            coords_key = f"coords{slice_index}"  # Adjust index as needed
            
            # Ensure coords are in the desired format, e.g., a list of [x, y] pairs
            formatted_coords = [[coord[0], coord[1]] for coord in filtered_coords]
            
            # Append directly in the desired final structure
            labels_key = f"labels{slice_index}"
            coords_labels_list.append({
                coords_key: np.array(formatted_coords),
                labels_key: np.array([label] * len(filtered_coords))
            })
    if not coords_labels_list: 
        print(f"No coordinates and labels extracted for label {label}.")
    return coords_labels_list

def filter_similar_coords(coords, grid_size=(10, 10)):
    if coords.size == 0:
        return coords
    grid_indices = np.floor_divide(coords[:, :2], grid_size)  # Assume coords are (x,y)
    _, unique_indices = np.unique(grid_indices, axis=0, return_index=True)
    return coords[unique_indices]

def combine_coords_labels(pos_labels, neg_labels):
    combined_slice_coords_labels = []
    total_slices = 36  # Total number of slices is fixed at 36
    for index in range(total_slices):
        pos_dict = next((item for item in pos_labels if f'coords{index}' in item), None)
        neg_dict = next((item for item in neg_labels if f'coords{index}' in item), None)

        # Initialize empty arrays for combined coords and labels
        combined_coords = np.array([], dtype=np.int64).reshape(0, 2)
        combined_labels = np.array([], dtype=np.int64)

        # If positive coordinates and labels are found, concatenate them to the combined arrays
        if pos_dict:
            combined_coords = np.concatenate([combined_coords, pos_dict[f'coords{index}']], axis=0)
            combined_labels = np.concatenate([combined_labels, pos_dict[f'labels{index}']], axis=0)

        # If negative coordinates and labels are found, concatenate them to the combined arrays
        if neg_dict:
            combined_coords = np.concatenate([combined_coords, neg_dict[f'coords{index}']], axis=0)
            combined_labels = np.concatenate([combined_labels, neg_dict[f'labels{index}']], axis=0)
        
        # Append the combined coords and labels for this slice index
        # Ensures that every slice index is represented in the output
        combined_slice_coords_labels.append({
            f'coords{index}': combined_coords,
            f'labels{index}': combined_labels
        })

    return combined_slice_coords_labels


def format_coords_and_labels(coords_labels_list):
    formatted_list = []
    for item in coords_labels_list:
        formatted_item = {}
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                # Convert numpy array to list and format with np.array
                formatted_value = f"np.array({value.tolist()})"
            else:
                formatted_value = value
            formatted_item[key] = formatted_value
        formatted_list.append(formatted_item)
    return formatted_list

def save_formatted_data(formatted_list, filename):
    with open(filename, 'w') as file:
        for i, item in enumerate(formatted_list):
            if i == len(formatted_list) - 1:
                file.write(f"{item}\n")
            else:
                file.write(f"{item},\n")

def transform_coords(coords, image_shape):
    # Rotate coordinates 90 degrees clockwise
    transformed_coords = np.empty_like(coords)
    transformed_coords[:, 0] = coords[:, 1]
    transformed_coords[:, 1] = image_shape[1] - coords[:, 0] - 1
    transformed_coords[:, 1] = image_shape[1] - 1 - transformed_coords[:, 1]
    return transformed_coords

def filter_coords_by_intensity(image, coords, labels, low_intensity_threshold=400, high_intensity_threshold=900):
    filtered_coords = []
    filtered_labels = []
    for coord, label in zip(coords, labels):
        x, y = int(coord[0]), int(coord[1])
        if label == 1 and image[y, x] >= low_intensity_threshold:
            # For label 1, keep if intensity is >= low_intensity_threshold
            filtered_coords.append(coord)
            filtered_labels.append(label)
        elif label == 0 and image[y, x] <= high_intensity_threshold:
            # For label 0, keep if intensity is <= high_intensity_threshold
            filtered_coords.append(coord)
            filtered_labels.append(label)
        elif label != 0 and label != 1:
            # Keep coordinates with labels other than 0 or 1 without intensity check
            filtered_coords.append(coord)
            filtered_labels.append(label)
            
    return np.array(filtered_coords), np.array(filtered_labels)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def parse_coords_and_labels_from_file(file_path):
    # Function to convert string representations of arrays into numpy arrays
    def string_to_np_array(array_string):
        return np.array(ast.literal_eval(array_string.replace('np.array', '')))
    
    slice_coords_labels = []
    with open(file_path, 'r') as file:
        for line in file:
            slice_dict = ast.literal_eval(line.strip(",\n"))
            for key in slice_dict.keys():
                if 'coords' in key or 'labels' in key:
                    slice_dict[key] = string_to_np_array(slice_dict[key])
            slice_coords_labels.append(slice_dict)
    return slice_coords_labels

def execute_command(command):
    print("Executing command:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    #print(f"STDOUT: {result.stdout}")
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def normalize_mask(mask):
    """Normalize the mask to range 0 to 1 only if it's not a label mask."""
    mask_min = np.min(mask)
    mask_max = np.max(mask)
    if mask_max > mask_min:
        normalized_mask = (mask - mask_min) / (mask_max - mask_min)
        #print(f"Normalized mask: Min = {np.min(normalized_mask)}, Max = {np.max(normalized_mask)}")
        return normalized_mask
    else:
        #print("Mask not normalized due to min and max values being too close or equal.")
        return mask

def majority_voting(nii_file_paths, ground_truth_mask):
    masks = []
    desired_shape = (512, 512, 40)
    for path in nii_file_paths:
        mask = nib.load(path).get_fdata()
        #print(f"Mask shape for {path}: {mask.shape}")  # Debug line to print each mask's shape
        if mask.shape != desired_shape:
            # Calculate the padding needed for each dimension
            pad_width = [(0, max(0, desired_dim - current_dim)) for desired_dim, current_dim in zip(desired_shape, mask.shape)]
            # Apply padding
            mask = np.pad(mask, pad_width=pad_width, mode='constant', constant_values=0)
        
        unique_values = np.unique(mask)
        if len(unique_values) == 2:
            # Assuming the mask should be binary with two unique values
            # Convert the smaller value to 0 and the larger to 1
            threshold = np.mean(unique_values)  # Midpoint between the two values
            mask = (mask > threshold).astype(int)
        else:
            # Handle other unexpected cases if necessary
            pass
        #print(f"Mask shape after reshape for {path}: {mask.shape}")  # Debug line to print each mask's shape after padding
        masks.append(mask)
    
    
    # Stack the masks along a new dimension
    stacked_masks = np.stack(masks, axis=-1)
    # Initialize the combined mask with zeros
    combined_mask = np.zeros(stacked_masks.shape[:-1], dtype=int)
    for i in range(stacked_masks.shape[2]):
        # Check if the ground truth mask slice is not blank (contains non-zero values)
        if np.any(ground_truth_mask[:, :, i]):
            # If the slice is not blank, perform majority voting on this slice
            #combined_mask[:, :, i] = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(np.int64))), axis=-1, arr=stacked_masks[:, :, i, :])
            combined_mask[:, :, i] = np.apply_along_axis(lambda x: 1 if np.sum(x) >= 3 else 0, axis=-1, arr=stacked_masks[:, :, i, :])
        # If the ground truth slice is blank, the combined_mask slice will remain zeros as initialized

    return combined_mask

def dice_coefficient(mask1, mask2):
    """Calculate the Dice Similarity Coefficient between two masks."""
    mask1 = np.asarray(mask1).astype(np.bool_)
    mask2 = np.asarray(mask2).astype(np.bool_)
    intersection = np.logical_and(mask1, mask2)
    return 2. * intersection.sum() / (mask1.sum() + mask2.sum())

def calculate_metrics(combined_mask, ground_truth_mask):
    dice = dice_coefficient(combined_mask, ground_truth_mask)
    print(f"{id}: dice: {dice:.4f}")


ids = ['0006','0007','0008','0009','0010'] 
nums = ['0', '1', '2', '3', '4'] #num of reference images
new_shape = (512, 512)
desired_shape = (512, 512, 36)
desired_depth = 36

REF_DIR = '/home/user/ref' #Replace with your path 
NII_DIR = '/home/user/test' #Replace with your path 
PROP_DIR = '/home/user/prompt' #Replace with your path 
LABEL_DIR = '/home/user/label' #Replace with your path 
save_dir = f'/home/user/inference' #Replace with your path 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#niftyreg fn path
reg_aladin_path = '/home/user/niftyreg-git/build/reg-apps/reg_aladin' #Replace with your path 
reg_f3d_path = '/home/user/niftyreg-git/build/reg-apps/reg_f3d' #Replace with your path 
reg_transform_path = '/home/user/niftyreg-git/build/reg-apps/reg_transform' #Replace with your path 
reg_resample_path = '/home/user/niftyreg-git/build/reg-apps/reg_resample' #Replace with your path 

sam_checkpoint = '/home/user/sam_vit_h_4b8939.pth' #Replace with your path 
model_type = "vit_h"

for id in ids:
    reg_save_dir = f'{NII_DIR}/test_warp'
    if not os.path.exists(reg_save_dir):
        os.makedirs(reg_save_dir)
    masks = None
    for num in nums:
        # images affine registration: new to ref -> aff_res
        nii_file = f"{NII_DIR}/{id}.nii"
        ref_file = f'{REF_DIR}/000{num}.nii'
        aff_matrix = f"{reg_save_dir}/aff_{id}_{num}.txt"
        aff_res_file = f"{reg_save_dir}/res_{id}_{num}.nii"
        
        if os.path.exists(nii_file) and os.path.exists(ref_file):
            affine_command = [reg_aladin_path, "-ref", ref_file, "-flo", nii_file, "-res", aff_res_file, "-aff", aff_matrix]
            if not execute_command(affine_command):
                continue
        
        final_file = f"{reg_save_dir}/final_res_{id}_{num}.nii"
        cpp_matrix = f"{reg_save_dir}/cpp_{id}_{num}.nii"
        
        # images non-rigid registration: aff_res -> final_res
        if os.path.exists(aff_res_file) and os.path.exists(ref_file):
            nonrigid_command = [reg_f3d_path, "-ref", ref_file, "-flo", aff_res_file, "-res", final_file, "-cpp", cpp_matrix, "-be", "0.65"]
            if not execute_command(nonrigid_command):
                continue
        
        # get inverse cpp matrix (final_res to aff_res )
        inv_cpp_matrix = f"{reg_save_dir}/inv_cpp_{id}_{num}.nii"
        if not os.path.exists(cpp_matrix):
            print(f"Missing affine matrix: {cpp_matrix}")
            continue
        cpp_inverse_command = [reg_transform_path, "-ref", aff_res_file, "-invNrr", cpp_matrix, final_file, inv_cpp_matrix]
        if not execute_command(cpp_inverse_command):
            continue

        # get inverse aff matrix (aff_res to new)
        inv_aff_matrix = f"{reg_save_dir}/inv_aff_{id}_{num}.txt"
        if not os.path.exists(aff_matrix):
            print(f"Missing affine matrix: {aff_matrix}")
            continue
        inverse_command = [reg_transform_path, "-ref", nii_file, "-invAff", aff_matrix, inv_aff_matrix]
        if not execute_command(inverse_command):
            continue

        #move the prompts (ref to aff_res)
        prop_pos_file = f"{PROP_DIR}/000{num}_pos.nii"
        prop_neg_file = f"{PROP_DIR}/000{num}_neg.nii"
        output_pos_mid_file = f"{PROP_DIR}/mid_{id}_pos{num}.nii"
        output_neg_mid_file = f"{PROP_DIR}/mid_{id}_neg{num}.nii"
        output_pos_file = f"{PROP_DIR}/final_{id}_pos{num}.nii"
        output_neg_file = f"{PROP_DIR}/final_{id}_neg{num}.nii"
        # prompts alignment
        if os.path.exists(prop_pos_file):
            resample_pos_command= [reg_resample_path, "-ref", aff_res_file, "-flo", prop_pos_file, "-trans", inv_cpp_matrix, "-res", output_pos_mid_file]
            if not execute_command(resample_pos_command):
                print(f"Failed to resample positive prompt for {id} with reference {num}")
            
        if os.path.exists(prop_neg_file):
            resample_neg_command = [reg_resample_path, "-ref", aff_res_file, "-flo", prop_neg_file, "-trans", inv_cpp_matrix, "-res", output_neg_mid_file]
            if not execute_command(resample_neg_command):
                print(f"Failed to resample negative prompt for {id} with reference {num}") 
        
        #move the prompts (aff_res to final)
        if os.path.exists(prop_pos_file):
            resample_pos_command= [reg_resample_path, "-ref", nii_file, "-flo", output_pos_mid_file, "-trans", inv_aff_matrix, "-res", output_pos_file]
            if not execute_command(resample_pos_command):
                print(f"Failed to resample positive prompt for {id} with reference {num}")
            
        if os.path.exists(prop_neg_file):
            resample_neg_command = [reg_resample_path, "-ref", nii_file, "-flo", output_neg_mid_file, "-trans", inv_aff_matrix, "-res", output_neg_file]
            if not execute_command(resample_neg_command):
                print(f"Failed to resample negative prompt for {id} with reference {num}") 


        if os.path.exists(output_pos_file):
            nii_prop_pos = nib.load(output_pos_file)

            mask_pos_data = nii_prop_pos.get_fdata()
            original_shape = mask_pos_data.shape[:2]
        else:
            print(f"Expected file not found: {output_pos_file}")
        
        if os.path.exists(output_neg_file):
            nii_prop_neg = nib.load(output_neg_file)

            mask_neg_data = nii_prop_neg.get_fdata()
            original_shape = mask_neg_data.shape[:2]
        else:
            print(f"Expected file not found: {output_neg_file}")

        combined_slice_coords_pos_labels = []
        combined_slice_coords_neg_labels = []
        combined_slice_coords_pos_labels.extend(extract_specific_value_coords_and_label(mask_pos_data, 1))
        combined_slice_coords_neg_labels.extend(extract_specific_value_coords_and_label(mask_neg_data, 0))
        #print(combined_slice_coords_neg_labels)
        combined_slice_coords_labels = combine_coords_labels(combined_slice_coords_pos_labels, combined_slice_coords_neg_labels)
        #print(combined_slice_coords_labels)
        formatted_list = format_coords_and_labels(combined_slice_coords_labels)
        #print(formatted_list)
        prompt_filename = f'{PROP_DIR}/aff_{id}_prompt{num}.txt'
        save_formatted_data(formatted_list, prompt_filename)

        slice_coords_labels = [] 
        slice_coords_labels = parse_coords_and_labels_from_file(prompt_filename)

        nii_file = f"{NII_DIR}/{id}.nii"

        try:
            nii_img = nib.load(nii_file)
            nii_data = nii_img.get_fdata()
            nii_data = standardize_intensity(nii_data)

        except FileNotFoundError:
            print(f"File not found: {nii_file}")
            continue
        except Exception as e:
            print(f"An error occurred while processing {id}: {e}")
            continue

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        predictor = SamPredictor(sam)

        all_masks = [] 
        for slice_idx in range(nii_data.shape[-1]):
            slice_data = nii_data[:, :, slice_idx]
            original_shape = slice_data.shape
            resized_image = cv2.resize(slice_data, new_shape)
    

            if len(resized_image.shape) == 3:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
            resized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            print(f"Slice {slice_idx} - Resized Image Shape: {resized_image.shape}, Max Value: {np.max(resized_image)}, Min Value: {np.min(resized_image)}")

            coords = None
            labels = None

            if slice_idx < len(slice_coords_labels):
                coords_key = f"coords{slice_idx}"
                labels_key = f"labels{slice_idx}"
                if coords_key in slice_coords_labels[slice_idx]:
                    coords = slice_coords_labels[slice_idx][coords_key]
                    labels = slice_coords_labels[slice_idx][labels_key]
                    coords, labels = filter_coords_by_intensity(slice_data, coords, labels)
                else:
            # If there are no coordinates or labels, append an empty mask
                    empty_mask = np.zeros(new_shape, dtype=np.uint8)
                    all_masks.append(empty_mask)
                    continue

                if coords is not None and labels is not None:
                    print(f"Coords Shape: {coords.shape}, Labels Shape: {labels.shape}")

                    if len(coords) > 0 and len(coords) == len(labels):
                        scaled_coords = rescale_coords(coords, original_shape, new_shape)
                        transformed_coords = transform_coords(scaled_coords, new_shape)
                        input_tensor = torch.tensor(resized_image).unsqueeze(0).repeat(3, 1, 1)
                    # Normalize and add batch dimension
                        input_tensor = input_tensor.float() / 255.0
                        input_tensor = input_tensor.unsqueeze(0).to(device)
                        input_image_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                        input_image_np = (input_image_np * 255).astype(np.uint8)
                        predictor.set_image(input_image_np)

                        masks, scores, logits = predictor.predict(
                            point_coords=transformed_coords,
                            point_labels=labels,
                            multimask_output=False,
                    )

                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            plt.figure(figsize=(10, 10))
                            plt.imshow(resized_image, cmap='gray')
                            show_mask(mask, plt.gca())
                            show_points(transformed_coords, labels, plt.gca())
                            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                            plt.axis('off')
                            plt.draw()
                            #save_path = os.path.join(save_dir, f'ref_slice_{slice_idx}_{id}_mask_{i+1}.png')
                            #plt.savefig(save_path)
                            plt.close()
                            all_masks.append(mask)
                else:
        # Append an empty mask if there are no coordinates or labels after filtering
                    empty_mask = np.zeros(new_shape, dtype=np.uint8)
                    all_masks.append(empty_mask)

        masks_array = np.array(all_masks)
        print("Shape of masks_array:", masks_array.shape)


# Stack the masks to match the original NIfTI file's shape
# Assuming all_masks is a list of 2D arrays and the number of slices matches
        masks_stack = np.stack(all_masks, axis=-1)
        masks_stack = masks_stack.astype(np.uint8) * 255

# Ensure the shape of masks_stack is the same as nii_data
        if masks_stack.shape != nii_data.shape:
            print("Warning: The shape of the masks does not match the original data.")
        nii_mask_img = nib.Nifti1Image(masks_stack, affine=nii_img.affine)

# Save the masks as a NIfTI file
        nii_filename = os.path.join(save_dir, f'{id}_mask{num}.nii')
        nib.save(nii_mask_img, nii_filename)  
    
    five_mask_paths = [f'{save_dir}/{id}_mask0.nii', f'{save_dir}/{id}_mask1.nii', f'{save_dir}/{id}_mask2.nii', f'{save_dir}/{id}_mask3.nii', f'{save_dir}/{id}_mask4.nii']
    ground_truth_mask_path = f'{LABEL_DIR}/{id}.nii'
    ground_truth_mask = nib.load(ground_truth_mask_path).get_fdata()
    ground_truth_mask = normalize_mask(ground_truth_mask)

    final_prediction = majority_voting(five_mask_paths, ground_truth_mask)
    #fianl_prediction = normalize_mask(fianl_prediction)
    calculate_metrics(final_prediction, ground_truth_mask)
    affine_matrix = np.eye(4) 
    final_prediction = final_prediction.astype(np.float32)
    nifti_image = nib.Nifti1Image(final_prediction, affine_matrix)

    final_filename = os.path.join(save_dir, f'final_{id}_mask.nii')
    nib.save(nifti_image, final_filename)
    print(f"final prediction saved as {final_filename}")


    



        