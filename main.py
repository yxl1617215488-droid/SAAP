import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attentionControl import AttentionControlEdit
import diff_latent_attack
from PIL import Image
import numpy as np
import os
import glob
from other_attacks import model_transfer
import random
import sys
from natsort import ns, natsorted
import argparse
import time

total_time = 0.0

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="test_inception000", type=str,
                    help='Where to save the images')
parser.add_argument('--images_root', default="./demo/images", type=str,
                    help='The images root directory')
parser.add_argument('--label_path', default="./demo/labels.txt", type=str,
                    help='The images labels.txt')
parser.add_argument('--pretrained_diffusion_path',
                    default="./stabilityai/stabilityaistable-diffusion-2-base",
                    type=str,
                    help='Change the path to `stabilityai/stable-diffusion-2-base` if want to use the pretrained model')

parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')

parser.add_argument('--dataset_name', default="imagenet_compatible", type=str,
                    choices=["imagenet_compatible", "cub_200_2011", "standford_car"],
                    help='The dataset name')
parser.add_argument('--start_step', default=15, type=int, help='Which DDIM step to start the attack')
parser.add_argument('--iterations', default=30, type=int, help='Iterations of optimizing the adv_image')
parser.add_argument('--tear_attack_steps', default=5, type=int,
                    help='Number of steps for tear attack optimization')
parser.add_argument('--topk_targets', default=3, type=int,
                    help='Number of target classes for tear attack')
parser.add_argument('--model_name', default="resnet", type=str,
                    help='The surrogate model from which the adversarial examples are crafted')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(42)


def run_image_processing(image, label, diffusion_model, classifier, diffusion_steps, guidance=2.5,
                         self_replace_steps=1., save_dir=r"C:\Users\PC\Desktop\output", res=224,
                         start_step=15, iterations=30, tear_attack_steps=5, topk_targets=3, image_index="0000"):

    from tear_diffusion_attack import run_tear_diffusion_attack
    
    processed_image, clean_acc, adv_acc = run_tear_diffusion_attack(
        image=image,
        label=torch.tensor([label], device='cuda:0'),
        diffusion_model=diffusion_model,
        classifier=classifier,
        num_inference_steps=diffusion_steps,
        guidance_scale=guidance,
        save_path=save_dir,
        res=res,
        start_step=start_step,
        iterations=iterations,
        tear_attack_steps=tear_attack_steps,
        topk_targets=topk_targets,
        image_index=image_index
    )

    return processed_image


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.res % 32 == 0 and args.res >= 96,

    guidance = args.guidance if hasattr(args, 'guidance') else 2.5
    diffusion_steps = args.diffusion_steps  
    res = args.res  

    save_dir = args.save_dir 
    os.makedirs(save_dir, exist_ok=True)

    images_root = args.images_root  
    label_path = args.label_path 
    with open(label_path, "r") as f:
        label = []
        for i in f.readlines():
            label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
        label = np.array(label)

    print(f"\n******Diffusion Processing, Dataset: {args.dataset_name}*********")
    pretrained_diffusion_path = args.pretrained_diffusion_path


    guidance = args.guidance if hasattr(args, 'guidance') else 2.5  
    diffusion_steps = args.diffusion_steps  
    res = args.res 
    start_step = args.start_step if hasattr(args, 'start_step') else 15 
    iterations = args.iterations if hasattr(args, 'iterations') else 30
    tear_attack_steps = args.tear_attack_steps if hasattr(args, 'tear_attack_steps') else 5
    topk_targets = args.topk_targets if hasattr(args, 'topk_targets') else 3
    model_name = args.model_name 

    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "./stabilityai/stabilityaistable-diffusion-2-base",
        local_files_only=True).to('cuda:0')

    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    from other_attacks import model_selection
    classifier = model_selection(model_name).eval().to('cuda:0')

    all_images = glob.glob(os.path.join(images_root, "*"))
    all_images = natsorted(all_images, alg=ns.PATH)

    all_processed_images = []
    all_images_data = []

    for ind, image_path in enumerate(all_images):
        tmp_image = Image.open(image_path).convert('RGB')
        processed_image = run_image_processing(tmp_image, label[ind],
                                               ldm_stable, classifier,
                                               diffusion_steps,
                                               guidance=guidance,
                                               res=res,
                                               save_dir=save_dir,
                                               start_step=start_step,
                                               iterations=iterations,
                                               tear_attack_steps=tear_attack_steps,
                                               topk_targets=topk_targets,
                                               image_index=str(ind).rjust(4, '0'))

        processed_image = processed_image.astype(np.float32) / 255.0
        all_processed_images.append(processed_image[None].transpose(0, 3, 1, 2))

        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)

        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0

        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)

        all_images_data.append(tmp_image)

        print(f"Processed image {ind + 1}/{len(all_images)}")

    images = np.concatenate(all_images_data)
    processed_images = np.concatenate(all_processed_images)

    print("Processing completed.")
