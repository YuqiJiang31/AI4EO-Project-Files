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




https://drive.google.com/drive/folders/1gPAVMiKxHU01T3_I45O9x_tvKzcpbQsy?usp=sharing

https://drive.google.com/drive/folders/1hePqnF60wMCAHqaYlt4j0K0MMI2j7RHl?usp=sharing
