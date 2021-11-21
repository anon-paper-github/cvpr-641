# Reference implementation for CVPR 641

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/anon-paper-github/cvpr-641/blob/main/stylegan_nada.ipynb)

## Description
This repo contains our implementation for CVPR submission #641

We set up a colab notebook so you can easily play with it yourself.

We've also included inversion in the notebook (using [ReStyle](https://github.com/yuval-alaluf/restyle-encoder)) so you can use the paired generators to edit real images.
We recommend using the [e4e](https://github.com/omertov/encoder4editing) based encoder for better editing at the cost of reconstruction accuracy.

If you want to set up the code and run it locally, please read below. <br>

*The code contains functions for experiments and losses which demonstrated inferior performance to our final model (i.e. negative results). 
Those are not reported on in the paper, and they are disabled by default. We have left them in the code so that future users can experiment with them on their own if they wish.

## Setup

The code relies on the official implementation of [CLIP](https://github.com/openai/CLIP), 
and the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2.

### Requirements
- Anaconda
- Pretrained StyleGAN2 generator (can be downloaded from [here](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing)). You can also download a model from [here](https://github.com/NVlabs/stylegan2-ada) and convert it with the provited script. See the colab notebook for examples.

In addition, run the following commands:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Usage

To convert a generator from one domain to another, use the colab notebook or run the training script in the ZSSGAN directory:

```
python train.py --size 1024 
                --batch 2 
                --n_sample 4 
                --output_dir /path/to/output/dir 
                --lr 0.002 
                --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt 
                --iter 301 
                --source_class "photo" 
                --target_class "sketch" 
                --auto_layer_k 18
                --auto_layer_iters 1 
                --auto_layer_batch 8 
                --output_interval 50 
                --clip_models "ViT-B/32" "ViT-B/16" 
                --clip_model_weights 1.0 1.0 
                --mixing 0.0
                --save_interval 150
```

Where you should adjust size to match the size of the pre-trained model, and the source_class and target_class descriptions control the direction of change.
For an explenation of each argument (and a few additional options), please consult ZSSGAN/options/train_options.py. For most modifications these default parameters should be good enough. See the colab notebook for more detailed directions.

Instead of using source and target texts, you can also target a style represented by a few images. Simply replace the `--source_class` and `--target_class` options with:

```
--style_img_dir /path/to/img/dir
```
where the directory should contain a few images (png, jpg or jpeg) with the style you want to mimic. There is no need to normalize or preprocess the images in any form.

## Pre-Trained Models

We provide a [Google Drive](https://drive.google.com/drive/folders/1i1irE2W40uYocWrFnkvsJ-9ywflzUbm8?usp=sharing) containing an assortment of models used in the paper.

## Editing Video

In order to generate a cross-domain editing video, prepare a set of edited latent codes in the original domain and run the following `generate_videos.py` script in the `ZSSGAN` directory:

```
python generate_videos.py --ckpt /model_dir/pixar.pt             \
                                 /model_dir/ukiyoe.pt            \
                                 /model_dir/edvard_munch.pt      \
                                 /model_dir/botero.pt            \
                          --out_dir /output/video/               \
                          --source_latent /latents/latent000.npy \
                          --target_latents /latents/
```

* The script relies on ffmpeg to function. On linux it can be installed by running `sudo apt install ffmpeg`
* The argument to `--ckpt` is a list of model checkpoints used to fill the grid. 
  * The number of models must be a perfect square, e.g. 1, 4, 9...
* The argument to `--target_latents` can be either a directory containing a set of `.npy` w-space latent codes, or a list of individual files.
* Please see the script for more details.

We also provide editing directions for use in video generation. To use the built-in directions, omit the ```--target_latents``` argument. You can use specific editing directions from the available list by passing them with the ```--edit_directions``` flag. See ```generate_videos.py``` for more information. <br>

For precomputed latents, we provide [example latent codes](https://drive.google.com/file/d/13jYtI4TD3uH2SKtMDSisg2uvnBunOX4N/view?usp=sharing) in the drive. If you want to generate your own, we recommend using [StyleCLIP](https://github.com/orpatashnik/StyleCLIP), [InterFaceGAN](https://github.com/genforce/interfacegan), [StyleFlow](https://github.com/RameenAbdal/StyleFlow), [GANSpace](https://github.com/harskish/ganspace) or any other latent space editing method.