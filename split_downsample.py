<<<<<<< HEAD
import os
import random
import shutil
from PIL import Image

def split_dataset_train_val(source_folder, output_folder, train_ratio=0.8):
    print(f"Starting dataset split, train ratio: {train_ratio}")
    
    # Create output directory structure
    train_path = os.path.join(output_folder, 'train')
    val_path = os.path.join(output_folder, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Iterate over each class
    for class_name in os.listdir(source_folder):
        source_class_path = os.path.join(source_folder, class_name)
        if not os.path.isdir(source_class_path):
            continue

        # Create class subfolders in train and val
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
        
        # Collect all image files for this class
        all_files = [f for f in os.listdir(source_class_path) if f.lower().endswith('.tif')]
        random.shuffle(all_files)  # Shuffle the list

        # Compute split index
        split_index = int(len(all_files) * train_ratio)
        train_files = all_files[:split_index]
        val_files = all_files[split_index:]

        # Copy files
        print(f"Processing class '{class_name}': {len(train_files)} train, {len(val_files)} val")
        for filename in train_files:
            shutil.copy2(os.path.join(source_class_path, filename), os.path.join(train_path, class_name, filename))
        
        for filename in val_files:
            shutil.copy2(os.path.join(source_class_path, filename), os.path.join(val_path, class_name, filename))
            
    print("\nDataset split completed!")


def resize_to_256x256(image):
    return image.resize((256, 256), Image.Resampling.BICUBIC)


def get_png_filename(original_filename):
    base_name = os.path.splitext(original_filename)[0]
    return f"{base_name}.png"


def process_and_downsample_folder(source_folder, lr_output_folder, hr_output_folder, factor=4):
    print(f"Starting processing folder: '{source_folder}'")
    print(f"Target HR size: 256x256, target LR size: {256//factor}x{256//factor}")
    print(f"Output format: PNG")
    
    processed_count = 0
    error_count = 0
    
    # Compute LR size
    lr_size = 256 // factor
    
    for root, _, files in os.walk(source_folder):
        # Create output directory structure
        relative_path = os.path.relpath(root, source_folder)
        lr_output_root = os.path.join(lr_output_folder, relative_path)
        hr_output_root = os.path.join(hr_output_folder, relative_path)
        os.makedirs(lr_output_root, exist_ok=True)
        os.makedirs(hr_output_root, exist_ok=True)

        for filename in files:
            if filename.lower().endswith('.tif'):
                source_image_path = os.path.join(root, filename)
                
                # Output filename in PNG
                png_filename = get_png_filename(filename)
                lr_output_path = os.path.join(lr_output_root, png_filename)
                hr_output_path = os.path.join(hr_output_root, png_filename)

                try:
                    with Image.open(source_image_path) as img:
                        original_size = img.size
                        
                        # Convert images with alpha / palette to RGB
                        if img.mode in ('RGBA', 'LA', 'P'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        hr_img = resize_to_256x256(img)
                        
                        lr_img = hr_img.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
                        
                        # Save as PNG
                        hr_img.save(hr_output_path, 'PNG')
                        lr_img.save(lr_output_path, 'PNG')
                        
                        processed_count += 1
                        
                        # Progress every 100 images
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} images...")
                        
                        # Verify sizes
                        if hr_img.size != (256, 256) or lr_img.size != (lr_size, lr_size):
                            print(f"Warning: {png_filename} has incorrect size!")
                            print(f"  Original: {original_size}, HR: {hr_img.size}, LR: {lr_img.size}")
                        
                except Exception as e:
                    print(f"Error processing file {source_image_path}: {e}")
                    error_count += 1

    print(f"Folder '{source_folder}' processing finished!")
    print(f"Success: {processed_count} images, Errors: {error_count} images")
    return processed_count, error_count


if __name__ == "__main__":
    # 1. Configuration
    source_dataset_path = 'UCMerced_LandUse/Images'
    
    # Intermediate: split raw dataset
    split_original_path = 'UCMerced_Split_Original'
    
    # Final HR output (256x256 PNG)
    final_hr_output_path = 'UCMerced_Split_HR_256_PNG'
    
    # Final LR output (64x64 PNG)
    final_lr_output_path = 'UCMerced_Split_LR_64_PNG'

    # Split ratio (70% train, 30% val)
    train_split_ratio = 0.7
    
    # Downsampling factor (256 -> 64)
    scaling_factor = 4

    print(f"Configuration:")
    print(f"- Source dataset: {source_dataset_path}")
    print(f"- Train/Val ratio: {train_split_ratio:.1%}/{1-train_split_ratio:.1%}")
    print(f"- HR size: 256x256")
    print(f"- LR size: {256//scaling_factor}x{256//scaling_factor}")
    print(f"- Downsample factor: {scaling_factor}x")
    print(f"- Output format: PNG")
    print("-" * 50)

    # Step 1: Split dataset
    print("--- Step 1: Splitting dataset ---")
    split_dataset_train_val(source_dataset_path, split_original_path, train_split_ratio)
    
    # Step 2: Process train & val
    print("\n--- Step 2: Processing split dataset ---")
    
    split_train_folder = os.path.join(split_original_path, 'train')
    split_val_folder = os.path.join(split_original_path, 'val')
    
    final_hr_train_folder = os.path.join(final_hr_output_path, 'train')
    final_hr_val_folder = os.path.join(final_hr_output_path, 'val')
    
    final_lr_train_folder = os.path.join(final_lr_output_path, 'train')
    final_lr_val_folder = os.path.join(final_lr_output_path, 'val')

    # Train set
    print("\n2.1 Processing training set...")
    train_processed, train_errors = process_and_downsample_folder(
        split_train_folder, final_lr_train_folder, final_hr_train_folder, scaling_factor
    )
    
    # Validation set
    print("\n2.2 Processing validation set...")
    val_processed, val_errors = process_and_downsample_folder(
        split_val_folder, final_lr_val_folder, final_hr_val_folder, scaling_factor
    )
    
    # Final stats
    total_processed = train_processed + val_processed
    total_errors = train_errors + val_errors
    
    print(f"\n{'='*60}")
    print(f"All processing completed!")
    print(f"{'='*60}")
    print(f"Statistics:")
    print(f"  - Train: {train_processed} success, {train_errors} errors")
    print(f"  - Val: {val_processed} success, {val_errors} errors")
    print(f"  - Total: {total_processed} success, {total_errors} errors")
    print(f"\n Output paths:")
    print(f"  - HR dataset (256x256): '{final_hr_output_path}'")
=======
import os
import random
import shutil
from PIL import Image

def split_dataset_train_val(source_folder, output_folder, train_ratio=0.8):
    print(f"Starting dataset split, train ratio: {train_ratio}")
    
    # Create output directory structure
    train_path = os.path.join(output_folder, 'train')
    val_path = os.path.join(output_folder, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Iterate over each class
    for class_name in os.listdir(source_folder):
        source_class_path = os.path.join(source_folder, class_name)
        if not os.path.isdir(source_class_path):
            continue

        # Create class subfolders in train and val
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
        
        # Collect all image files for this class
        all_files = [f for f in os.listdir(source_class_path) if f.lower().endswith('.tif')]
        random.shuffle(all_files)  # Shuffle the list

        # Compute split index
        split_index = int(len(all_files) * train_ratio)
        train_files = all_files[:split_index]
        val_files = all_files[split_index:]

        # Copy files
        print(f"Processing class '{class_name}': {len(train_files)} train, {len(val_files)} val")
        for filename in train_files:
            shutil.copy2(os.path.join(source_class_path, filename), os.path.join(train_path, class_name, filename))
        
        for filename in val_files:
            shutil.copy2(os.path.join(source_class_path, filename), os.path.join(val_path, class_name, filename))
            
    print("\nDataset split completed!")


def resize_to_256x256(image):
    return image.resize((256, 256), Image.Resampling.BICUBIC)


def get_png_filename(original_filename):
    base_name = os.path.splitext(original_filename)[0]
    return f"{base_name}.png"


def process_and_downsample_folder(source_folder, lr_output_folder, hr_output_folder, factor=4):
    print(f"Starting processing folder: '{source_folder}'")
    print(f"Target HR size: 256x256, target LR size: {256//factor}x{256//factor}")
    print(f"Output format: PNG")
    
    processed_count = 0
    error_count = 0
    
    # Compute LR size
    lr_size = 256 // factor
    
    for root, _, files in os.walk(source_folder):
        # Create output directory structure
        relative_path = os.path.relpath(root, source_folder)
        lr_output_root = os.path.join(lr_output_folder, relative_path)
        hr_output_root = os.path.join(hr_output_folder, relative_path)
        os.makedirs(lr_output_root, exist_ok=True)
        os.makedirs(hr_output_root, exist_ok=True)

        for filename in files:
            if filename.lower().endswith('.tif'):
                source_image_path = os.path.join(root, filename)
                
                # Output filename in PNG
                png_filename = get_png_filename(filename)
                lr_output_path = os.path.join(lr_output_root, png_filename)
                hr_output_path = os.path.join(hr_output_root, png_filename)

                try:
                    with Image.open(source_image_path) as img:
                        original_size = img.size
                        
                        # Convert images with alpha / palette to RGB
                        if img.mode in ('RGBA', 'LA', 'P'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        hr_img = resize_to_256x256(img)
                        
                        lr_img = hr_img.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
                        
                        # Save as PNG
                        hr_img.save(hr_output_path, 'PNG')
                        lr_img.save(lr_output_path, 'PNG')
                        
                        processed_count += 1
                        
                        # Progress every 100 images
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} images...")
                        
                        # Verify sizes
                        if hr_img.size != (256, 256) or lr_img.size != (lr_size, lr_size):
                            print(f"Warning: {png_filename} has incorrect size!")
                            print(f"  Original: {original_size}, HR: {hr_img.size}, LR: {lr_img.size}")
                        
                except Exception as e:
                    print(f"Error processing file {source_image_path}: {e}")
                    error_count += 1

    print(f"Folder '{source_folder}' processing finished!")
    print(f"Success: {processed_count} images, Errors: {error_count} images")
    return processed_count, error_count


if __name__ == "__main__":
    # 1. Configuration
    source_dataset_path = 'UCMerced_LandUse/Images'
    
    # Intermediate: split raw dataset
    split_original_path = 'UCMerced_Split_Original'
    
    # Final HR output (256x256 PNG)
    final_hr_output_path = 'UCMerced_Split_HR_256_PNG'
    
    # Final LR output (64x64 PNG)
    final_lr_output_path = 'UCMerced_Split_LR_64_PNG'

    # Split ratio (70% train, 30% val)
    train_split_ratio = 0.7
    
    # Downsampling factor (256 -> 64)
    scaling_factor = 4

    print(f"Configuration:")
    print(f"- Source dataset: {source_dataset_path}")
    print(f"- Train/Val ratio: {train_split_ratio:.1%}/{1-train_split_ratio:.1%}")
    print(f"- HR size: 256x256")
    print(f"- LR size: {256//scaling_factor}x{256//scaling_factor}")
    print(f"- Downsample factor: {scaling_factor}x")
    print(f"- Output format: PNG")
    print("-" * 50)

    # Step 1: Split dataset
    print("--- Step 1: Splitting dataset ---")
    split_dataset_train_val(source_dataset_path, split_original_path, train_split_ratio)
    
    # Step 2: Process train & val
    print("\n--- Step 2: Processing split dataset ---")
    
    split_train_folder = os.path.join(split_original_path, 'train')
    split_val_folder = os.path.join(split_original_path, 'val')
    
    final_hr_train_folder = os.path.join(final_hr_output_path, 'train')
    final_hr_val_folder = os.path.join(final_hr_output_path, 'val')
    
    final_lr_train_folder = os.path.join(final_lr_output_path, 'train')
    final_lr_val_folder = os.path.join(final_lr_output_path, 'val')

    # Train set
    print("\n2.1 Processing training set...")
    train_processed, train_errors = process_and_downsample_folder(
        split_train_folder, final_lr_train_folder, final_hr_train_folder, scaling_factor
    )
    
    # Validation set
    print("\n2.2 Processing validation set...")
    val_processed, val_errors = process_and_downsample_folder(
        split_val_folder, final_lr_val_folder, final_hr_val_folder, scaling_factor
    )
    
    # Final stats
    total_processed = train_processed + val_processed
    total_errors = train_errors + val_errors
    
    print(f"\n{'='*60}")
    print(f"All processing completed!")
    print(f"{'='*60}")
    print(f"Statistics:")
    print(f"  - Train: {train_processed} success, {train_errors} errors")
    print(f"  - Val: {val_processed} success, {val_errors} errors")
    print(f"  - Total: {total_processed} success, {total_errors} errors")
    print(f"\n Output paths:")
    print(f"  - HR dataset (256x256): '{final_hr_output_path}'")
>>>>>>> 7c2f2f8742e4ed38e944bf84dbd72169c58a4bf0
    print(f"  - LR dataset (64x64): '{final_lr_output_path}'")