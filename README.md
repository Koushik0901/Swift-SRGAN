# Swift-SRGAN - Rethinking Super-Resolution for real-time inference

This repository is the official implementation of the paper **"Swift-SRGAN - Rethinking Super-Resolution for real-time inference"**
https://arxiv.org/abs/2111.14320

## Architecture
<p align="center"> <img src="https://github.com/Koushik0901/Swift-SRGAN/blob/master/image-samples/SwiftSRGAN-architecture.png" width="850" height="450"  /> </p>

## Super-Resolution Examples
<p align="center"> <b><i>All images on the left side are the original high resolution images and images on the right side are the 4x super-resolution output from our model.</i></b>  
<p align="center"> <img src="https://github.com/Koushik0901/Swift-SRGAN/blob/master/image-samples/4x_samples/baryon.png" width="800" height="400"  /> </p>
<p align="center"> <img src="https://github.com/Koushik0901/Swift-SRGAN/blob/master/image-samples/4x_samples/dwight.png" width="800" height="400"  /> </p>
<p align="center"> <img src="https://github.com/Koushik0901/Swift-SRGAN/blob/master/image-samples/4x_samples/steve.png" width="800" height="450"  /> </p>

## Pre-trained Models
  **Check the releases tab for pre-trained 4x and 2x upsampling generator models**

## Training
1. install requirements with:
    `pip install -r requirements.txt`
3. Train the model by executing:
    ``` bash
    cd swift-srgan
    python train.py --upscale_factor 4 --crop_size 96 --num_epochs 100
    ```
    
4. To convert the generator model to torchscript, run 
``` bash
python optimize-graph.py --ckpt_path ./checkpoints/netG_4x_epoch100.pth.tar --save_path ./checkpoints/optimized_model.pt --device cuda
```

## Please cite our article
 ``` bibtex
  @article{krishnan2021swiftsrgan,
  title={SwiftSRGAN--Rethinking Super-Resolution for Efficient and Real-time Inference},
  author={Krishnan, Koushik Sivarama and Krishnan, Karthik Sivarama},
  journal={arXiv preprint arXiv:2111.14320},
  year={2021}
  }
