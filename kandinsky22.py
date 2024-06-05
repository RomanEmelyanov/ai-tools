#!/usr/bin/env python3

import sys, torch, os, PIL
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.utils import load_image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from uuid import uuid4
import numpy as np

DEVICE_CPU = torch.device('cpu:0')
DEVICE_GPU = torch.device('mps:0')

# Loading encoder and prior pipeline into the RAM to be run on the CPU
# and unet and decoder to the VRAM to be run on the GPU.
# Note the usage of float32 for the CPU and float16 (half) for the GPU
# Set the `local_files_only` to True after the initial downloading
# to allow offline use (without active Internet connection)
print("*** Loading encoder ***")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    subfolder='image_encoder',
    cache_dir='./kand22',
    # local_files_only=True
).to(DEVICE_CPU)

print("*** Loading unet ***")
unet = UNet2DConditionModel.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    subfolder='unet',
    cache_dir='./kand22',
    # local_files_only=True
).half().to(DEVICE_GPU)
#).to(DEVICE_GPU)

print("*** Loading prior ***")
prior = KandinskyV22PriorPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    image_encoder=image_encoder, 
    torch_dtype=torch.float32,
    cache_dir='./kand22',
    # local_files_only=True
).to(DEVICE_CPU)

print("*** Loading decoder ***")
decoder = KandinskyV22Pipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    unet=unet,
    torch_dtype=torch.float16,
    # revision="fp16", # FIX
    cache_dir='./kand22',
    # local_files_only=True
).to(DEVICE_GPU)

job_id = str(uuid4())

# torch.manual_seed(42)

num_batches = 2
images_per_batch = 1
total_num_images = images_per_batch * num_batches

negative_prior_prompt = 'lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

images = []

print(f"*** Generating {total_num_images} image(s) ***")
for i in range(num_batches):
    print(f"* Batch {i + 1} of {num_batches} *")
    # Generating embeddings on the CPU
    img_emb = prior(
        prompt='Hacker, 4k photo',
        num_inference_steps=25,
        num_images_per_prompt=images_per_batch)

    negative_emb = prior(
        prompt=negative_prior_prompt,
        num_inference_steps=25,
        num_images_per_prompt=images_per_batch
    )
    # Converting fp32 to fp16, to run decoder on the GPU
    image_batch = decoder(
        image_embeds=img_emb.image_embeds.half(),
        negative_image_embeds=negative_emb.image_embeds.half(),
        num_inference_steps=25, height=1024, width=1024)

    images += image_batch.images
# Saving the images
os.mkdir(job_id)
for (idx, img) in enumerate(images):
    img.save(f"{job_id}/img_{job_id}_{idx + 1}.png")