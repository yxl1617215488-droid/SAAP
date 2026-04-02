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
    """
    设置随机种子以确保实验结果可复现

    参数:
        seed (int): 随机种子值，默认为42

    返回:
        None
    """
    # 设置 Python 内置 random 模块的种子
    random.seed(seed)

    # 设置环境变量 PYTHONHASHSEED，确保 Python 的哈希函数行为一致
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的 CPU 随机种子
    torch.manual_seed(seed)

    # 设置 PyTorch 的当前 GPU 随机种子
    torch.cuda.manual_seed(seed)

    # 设置 PyTorch 的所有 GPU 随机种子
    torch.cuda.manual_seed_all(seed)

    # 关闭 cuDNN 的自动优化功能，保证每次计算一致
    torch.backends.cudnn.benchmark = False

    # 启用 cuDNN 的确定性模式，保证运算过程可复现
    torch.backends.cudnn.deterministic = True


# 调用 seed_torch 函数，设置全局随机种子
seed_torch(42)


def run_image_processing(image, label, diffusion_model, classifier, diffusion_steps, guidance=2.5,
                         self_replace_steps=1., save_dir=r"C:\Users\PC\Desktop\output", res=224,
                         start_step=15, iterations=30, tear_attack_steps=5, topk_targets=3, image_index="0000"):
    """
    执行基于扩散模型的图像处理

    该函数使用扩散模型对输入图像进行处理

    参数:
    - image: 输入图像
    - label: 输入图像的标签
    - diffusion_model: 使用的扩散模型（如Stable Diffusion）
    - classifier: 分类器模型
    - diffusion_steps: 总的DDIM采样步数
    - guidance: 指导尺度，控制生成过程中的条件强度，默认值为2.5
    - self_replace_steps: 自注意力替换步骤的比例，默认值为1.0
    - save_dir: 保存生成结果的目录
    - res: 输入图像的分辨率，默认为224x224
    - start_step: 开始攻击的步数
    - iterations: 总迭代次数
    - tear_attack_steps: Tear Attack优化步数
    - topk_targets: 目标类别数量
    - image_index: 图像索引，用于文件命名

    返回:
    - processed_image: 处理后的图像
    """
    from tear_diffusion_attack import run_tear_diffusion_attack
    
    # 使用Tear Diffusion Attack处理图像
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
    # 解析命令行参数
    args = parser.parse_args()

    # 确保输入分辨率是32的倍数且不小于96，以满足模型输入要求
    assert args.res % 32 == 0 and args.res >= 96, "请确保输入分辨率是32的倍数且不小于96。"

    # 提取参数
    guidance = args.guidance if hasattr(args, 'guidance') else 2.5  # 修复拼写错误
    diffusion_steps = args.diffusion_steps  # 总共的DDIM采样步数
    res = args.res  # 输入图像调整后的分辨率

    save_dir = args.save_dir  # Where to save the results.
    os.makedirs(save_dir, exist_ok=True)

    images_root = args.images_root  # The images' root directory.
    label_path = args.label_path  # The images' labels.txt.
    with open(label_path, "r") as f:
        label = []
        for i in f.readlines():
            label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
        label = np.array(label)

    print(f"\n******Diffusion Processing, Dataset: {args.dataset_name}*********")

    # Change the path to "stabilityai/stable-diffusion-2-base" if you want to use the pretrained model.
    pretrained_diffusion_path = args.pretrained_diffusion_path

    # 提取参数
    guidance = args.guidance if hasattr(args, 'guidance') else 2.5  # 修复拼写错误
    diffusion_steps = args.diffusion_steps  # 总共的DDIM采样步数
    res = args.res  # 输入图像调整后的分辨率
    start_step = args.start_step if hasattr(args, 'start_step') else 15  # 开始攻击的步数
    iterations = args.iterations if hasattr(args, 'iterations') else 30  # 总迭代次数
    tear_attack_steps = args.tear_attack_steps if hasattr(args, 'tear_attack_steps') else 5  # Tear Attack优化步数
    topk_targets = args.topk_targets if hasattr(args, 'topk_targets') else 3  # 目标类别数量
    model_name = args.model_name  # 模型名称

    # 加载预训练的Stable Diffusion模型管道，并且只使用本地文件进行加载，避免下载
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "./stabilityai/stabilityaistable-diffusion-2-base",
        local_files_only=True).to('cuda:0')

    # 将模型的调度器替换为DDIM调度器，以可能获得更稳定的生成过程或更好的生成质量
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    # 加载分类器
    from other_attacks import model_selection
    classifier = model_selection(model_name).eval().to('cuda:0')

    # Process a subset of images
    # 获取指定目录下的所有图像路径
    all_images = glob.glob(os.path.join(images_root, "*"))
    # 以自然顺序对图像路径进行排序，以便于后续处理
    all_images = natsorted(all_images, alg=ns.PATH)

    all_processed_images = []
    all_images_data = []

    # 遍历所有图片，执行处理过程
    for ind, image_path in enumerate(all_images):
        # 打开图片并确保其为RGB格式
        tmp_image = Image.open(image_path).convert('RGB')
        # 注意：原始图片将在run_tear_diffusion_attack函数内部保存，此处不再重复保存

        # 对当前图片执行扩散处理
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

        # 将处理后的图像添加到列表中
        processed_image = processed_image.astype(np.float32) / 255.0
        all_processed_images.append(processed_image[None].transpose(0, 3, 1, 2))

        # 将图像调整为指定分辨率 res x res，使用 LANCZOS 重采样算法以保持高质量
        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)

        # 将图像转换为 NumPy 数组，并将像素值归一化到 [0, 1] 范围内（浮点型）
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0

        # 增加一个维度表示批次大小（batch size），然后重新排列维度为 (N, C, H, W)，符合深度学习模型输入格式要求
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)

        # 将处理后的图像添加到 images 列表中，用于后续批量处理
        all_images_data.append(tmp_image)

        print(f"Processed image {ind + 1}/{len(all_images)}")

    # 将 images 和 processed_images 列表中的多个数组合并为单一的 NumPy 数组
    images = np.concatenate(all_images_data)
    processed_images = np.concatenate(all_processed_images)

    print("Processing completed.")