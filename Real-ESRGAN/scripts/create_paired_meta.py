import os

hr_folder = 'datasets/UCMerced/train/HR'
lr_folder = 'datasets/UCMerced/train/LR'
meta_info_file = 'datasets/UCMerced/meta_info_UCMerced_train_paired.txt'
# ---------------------------------------------

hr_images = []
for root, _, files in os.walk(hr_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            relative_path = os.path.relpath(os.path.join(root, file), hr_folder)
            hr_images.append(relative_path)


paired_list = []
missing_lr_files = 0
for hr_path in hr_images:
    lr_path = hr_path
    
    if os.path.exists(os.path.join(lr_folder, lr_path)):
        paired_list.append(f"{hr_path.replace(os.sep, '/')}, {lr_path.replace(os.sep, '/')}")
    else:
        print(f"Warning: LR file corresponding to {hr_path} not found.")
        missing_lr_files += 1


with open(meta_info_file, 'w') as f:
    for line in paired_list:
        f.write(line + '\n')
