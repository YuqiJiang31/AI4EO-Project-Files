## Repository Structure

```
main/
├── LPIPS_RESULT_Bicubic/           # LPIPS analysis results of bicubic images
│   ├── lpips_folder_mean.csv       # LPIPS results per class
│   ├── lpips_per_image.csv         # LPIPS results per image
│   └── lpips_overall.txt           # Overall LPIPS result
├── LPIPS_RESULT_RealESRGAN/        # Same as LPIPS_RESULT_Bicubic/
├── LPIPS_RESULT_SwinIR/            # Same as LPIPS_RESULT_Bicubic/
├── SSIM_PSNR_RESULT_Bicubic/       # SSIM and PSNR analysis results of bicubic images
│   ├── ssim_psnr_folder_mean.csv   # SSIM and PSNR results per class
│   ├── ssim_psnr_per_image.csv     # SSIM and PSNR results per class
│   └── ssim_psnr_overall.txt       # Overall SSIM and PSNR result
├── SSIM_PSNR_RESULT_RealESRGAN/    # Same as SSIM_PSNR_RESULT_Bicubic/
├── SSIM_PSNR_RESULT_SwinIR/        # Same as SSIM_PSNR_RESULT_Bicubic/
├── SwinIR/SwinIR/model_zoo         # SwinIR pretrained weights used for Project SR
│   └── 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
├── Bicubic.py                      # Batch Bicubic Interpolation
├── LPIPS.py                        # Calculate LPIPS perceived distance. Optional suffix for SR image
├── SSIM_PSNR.py                    # Batch match ground truth and SR images (supports optional suffixes)
├── classification_task.py          # Use pre-trained ResNet18 to train image classification on the dataset
├── split_downsample.py             # Randomly divide the original dataset into training/validation sets and PNG
├── best_resnet18_Bicubic.pth       # The best model on the bicubic SR dataset
├── best_resnet18_RealESRGAN.pth    # The best model on the RealESRGAN SR dataset
├── best_resnet18_SwinIR.pth        # The best model on the SwinIR SR dataset
├── best_resnet18_Truth.pth         # The best model on the Ground Truth
├── requirements.txt                # Project dependencies
└── README.md
```
# Installation
## 1. Clone the repository:
```bash
git clone https://github.com/YuqiJiang31/AI4EO-Project-Files.git
cd AI4EO-Project-Files
```
## 2. Create and activate an environment
```bash
conda create -y -n v_env python=3.10
conda activate v_env
```
## 3. Install dependencies:
```bash
conda install -r requirements.txt
```

# Datasets
The raw dataset used in the experiment and the datasets output by each step can be downloaded from the following links.
https://drive.google.com/drive/folders/1hePqnF60wMCAHqaYlt4j0K0MMI2j7RHl?usp=sharing

There are 7 folders in total: 
- UCMerced_LandUse is a Raw dataset in .tif format.
- UCMerced_LR_64_PNG is the dataset after downsampling conversion format, which serves as the input of the following three SR methods.
- UCMerced_HR_Truth is the dataset converted to png format.(Ground Truth)
- UCMerced_Split is the dataset used to fine-tune RealESRGAN after running split_downsample.py
- UCMerced_HR_Bicubic (outputs of Bicubic.py)
- UCMerced_HR_SwinIR (outputs of running SwinIR)
- UCMerced_HR_RealESRGAN (outputs of running RealESRGAN)

Reproduce model training: using UCMerced_Split as input.

Reproduce the classification of resnet18: using UCMerced_HR_Bicubic; UCMerced_HR_SwinIR; UCMerced_HR_RealESRGAN; UCMerced_HR_Truth as input.

Reproduce LPIPS, SSIM, PSNR analysis results: use UCMerced_HR_Bicubic; UCMerced_HR_SwinIR; UCMerced_HR_RealESRGAN as input

# Trainning
The dataset, training parameters, and trained model required for training can be downloaded from the following link.
https://drive.google.com/drive/folders/1gPAVMiKxHU01T3_I45O9x_tvKzcpbQsy?usp=sharing

## 0. Configuring Real-ESRGAN

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
```
The environment of this project is adapted to Real-ESRGAN, so there is no need to configure the environment.

## 1. Prepare the dataset

- **gt folder**(Standard reference, high-resolution image):*datasets/UCMerced/train/HR*
- **lq folder**(low quality, low resolution image):：*datasets/UCMerced/train/LR*

You can then use the script scripts/create_paired_meta.py to generate the meta information (meta_info) txt file.

```bash
python scripts/create_paired_meta.py #Input and output paths are modified within the script
```

## 2. Download the pre-trained model
Download the pre-trained model to the `experiments/pretrained_models` directory. The link above has been downloaded.

## 3. Fine-tuning
Modify the options file: `options/finetune_realesrgan_x4plus_pairdata.yml`, especially the `datasets` part, 
The `option/finetune_realesrgan_x4plus_pairdata.yml` file available for download in the link above has been configured.
```yml
datasets:
  train:
    name: UCMerced_train
    type: RealESRGANPairedDataset
    dataroot_gt: datasets/UCMerced/train/HR
    dataroot_lq: datasets/UCMerced/train/LR
    meta_info: datasets\UCMerced\meta_info_UCMerced_train_paired.txt
    io_backend:
      type: disk

    gt_size: 64  #For multi-GPU parallel training or good GPU performance, you can increase the value to 128 or 256.
    use_hflip: True
    use_rot: False

    # data loader
    use_shuffle: true
    num_worker_per_gpu: 0
    batch_size_per_gpu: 12
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: validation
    type: PairedImageDataset
    dataroot_gt: datasets/UCMerced/val/HR
    dataroot_lq: datasets/UCMerced/val/LR
    io_backend:
      type: disk
```

Training with **1 GPU**: 
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume
```

## 4. Model

The trained model can be downloaded from the link above in the folder: `experiments/finetune_RealESRGANx4plus_pairdata/net_g_latest.pth`

# Test

## 1. 4x Super-Resolution with Bicubic
```bash
python Bicubic.py --input Datasets/UCMerced_LR_64_PNG --output Datasets/UCMerced_Bicubic_HR_Bicubic --scale 4
```

## 2. 4x Super-Resolution with SwinIR
```bash
git clone https://github.com/JingyunLiang/SwinIR.git
cd SwinIR
```
Put this project `/SwinIR/SwinIR/model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth` into the `model_zoo `folder

```bash
python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq datasets\
```

## 3. 4x Super-Resolution with Real-ESRGAN
The input image is placed in the `/Real-ESRGAN/inputs` folder
```bash
cd Real-ESRGAN
python inference_realesrgan.py -n net_g_latest -i inputs
```
Results are in the `/Real-ESRGAN/results` folder

# Evaluation
## 1. LPIPS Evaluation
Open LPIPS.py and edit the input and output parameters.
```py
CONFIG = {
    "GT_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_Truth",
    "SR_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_SwinIR",  # Switch the output results of three super-resolution algorithms
    "OUT_DIR": r"E:\Text\AI4EOFINAL\LPIPS_RESULT",
    "NET": "vgg",
    "BATCH": 8,
    "EXTENSIONS": (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    "FORCE_CPU": False,
    "SKIP_MISMATCH": False,
    "FOLDER_LEVEL_ONLY": False,
    "VERBOSE": True,
    "SR_SUFFIX": "_SwinIR"          # Added: extra suffix in SR filenames relative to GT; leave "" if none
}
```
```bash
python LPIPS.py
```
## 2. SSIM and PSNR Evaluation
Open SSIM_PSNR.py and edit the input and output parameters.
```py
CONFIG = {
    "GT_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_Truth",
    "SR_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_Bicubic",  # Switch the output results of three super-resolution algorithms
    "OUT_DIR": r"E:\Text\AI4EOFINAL\SSIM_PSNR_RESULT",
    "BATCH": 1,
    "EXTENSIONS": (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    "FORCE_CPU": False,
    "SKIP_MISMATCH": False,
    "FOLDER_LEVEL_ONLY": False,
    "VERBOSE": True,
    "SR_SUFFIX": "",      # additional suffix compared to GT leave blank if not provided
    "CONVERT_TO_Y": False,
}
```

```bash
python LSSIM_PSNR.py
```

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:
```
@software{AI4EO-Project-Files,
  author = {YuqiJiang31},
  title = {A Comparative Analysis of Super-Resolution Models for Land Use Classification with Remote Sensing Imagery},
  year = {2025},
  url = {https://github.com/YuqiJiang31/AI4EO-Project-Files}
}
```











