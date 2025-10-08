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
## Installation
1. Clone the repository:
```bash
git clone https://github.com/YuqiJiang31/AI4EO-Project-Files.git
cd AI4EO-Project-Files
```
2. Create and activate an environment
```bash
conda create -y -n v_env python=3.10
conda activate v_env
```
3. Install dependencies:
```bash
conda install -r requirements.txt
```

## Datasets
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















https://drive.google.com/drive/folders/1gPAVMiKxHU01T3_I45O9x_tvKzcpbQsy?usp=sharing
