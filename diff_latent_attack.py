from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import other_attacks


# -----------------------
# （保留）混沌噪声类：保留但默认不使用（仅用于后续对比/实验）
# -----------------------
class IndependentChaosNoise:
    def __init__(self, latent_shape, system='logistic'):
        self.latent_shape = latent_shape
        self.system = system
        np.random.seed(int(torch.randint(1000, 9999, (1,)).item()))

        if system == 'logistic':
            self.state = np.random.random(latent_shape)
            self.r = 3.9
        elif system == 'lorenz':
            self.state = np.random.random(3) * 2 - 1
            self.sigma = 10
            self.rho = 28
            self.beta = 8 / 3
        elif system == 'rossler':
            self.state = np.random.random(3) * 2 - 1
            self.a = 0.2
            self.b = 0.2
            self.c = 5.7

    def logistic_map(self, x):
        return self.r * x * (1 - x)

    def lorenz_system(self, state, dt=0.01):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz]) * dt

    def rossler_system(self, state, dt=0.01):
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz]) * dt

    def generate_chaos_noise(self, scale=0.1):
        if self.system == 'logistic':
            self.state = self.logistic_map(self.state)
            noise = (self.state - 0.5) * 2 * scale
        elif self.system == 'lorenz':
            for _ in range(10):
                self.state += self.lorenz_system(self.state)
            noise = np.random.randn(*self.latent_shape) * scale
            modulation = np.tanh(self.state[0])
            noise = noise * (1 + modulation * 0.1)
        elif self.system == 'rossler':
            for _ in range(10):
                self.state += self.rossler_system(self.state)
            noise = np.random.randn(*self.latent_shape) * scale
            modulation = np.tanh(self.state[0])
            noise = noise * (1 + modulation * 0.1)
        return torch.tensor(noise, dtype=torch.float32)


class LogitBasedChaosNoise:
    def __init__(self, latent_shape, system='lorenz'):
        self.latent_shape = latent_shape
        self.system = system
        np.random.seed(42)
        self.state = np.random.random(3) * 2 - 1
        if system == 'lorenz':
            self.sigma = 10
            self.rho = 28
            self.beta = 8 / 3
        elif system == 'rossler':
            self.a = 0.2
            self.b = 0.2
            self.c = 5.7
        self.dt = 0.01

    def lorenz_system(self, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz]) * self.dt

    def rossler_system(self, state):
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz]) * self.dt

    def generate_chaos_noise(self, logit=None, scale=0.1):
        for _ in range(10):
            if self.system == 'lorenz':
                self.state += self.lorenz_system(self.state)
            elif self.system == 'rossler':
                self.state += self.rossler_system(self.state)
        noise = np.random.randn(*self.latent_shape) * scale
        if logit is not None:
            if isinstance(logit, torch.Tensor):
                logit_np = logit.detach().cpu().numpy()
            else:
                logit_np = np.array(logit)
            if logit_np.ndim > 1:
                logit_np = logit_np.flatten()
            logit_var = np.var(logit_np) if len(logit_np) > 1 else 0
            modulation_factor = np.tanh(logit_var)
            lorenz_modulation = np.tanh(self.state[0])
            noise = noise * (1 + modulation_factor * 0.1 + lorenz_modulation * 0.05)
        return torch.tensor(noise, dtype=torch.float32)


class ChaosPerturbation:
    def __init__(self, latent_shape, system='lorenz'):
        self.latent_shape = latent_shape
        self.system = system
        np.random.seed(42)
        self.state = np.random.random(3) * 2 - 1
        if system == 'lorenz':
            self.sigma = 10
            self.rho = 28
            self.beta = 8 / 3
        elif system == 'rossler':
            self.a = 0.2
            self.b = 0.2
            self.c = 5.7

    def lorenz_system(self, state, sigma=10, rho=28, beta=8 / 3):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])

    def rossler_system(self, state, a=0.2, b=0.2, c=5.7):
        x, y, z = state
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return np.array([dx, dy, dz])

    def generate_perturbation(self, scale=0.01):
        if self.system == 'lorenz':
            self.state = self.lorenz_system(self.state)
        elif self.system == 'rossler':
            self.state = self.rossler_system(self.state)
        perturbation = np.random.randn(*self.latent_shape) * scale
        modulation = np.tanh(self.state[0])
        perturbation = perturbation * (1 + modulation * 0.1)
        return torch.tensor(perturbation, dtype=torch.float32)

    def shuffle_similarity_matrix(self, sim_matrix, shuffle_ratio=1.0):
        # optional utility — not used in the recommended "no-noise" mode
        if shuffle_ratio <= 0.0:
            return sim_matrix, torch.tensor(0.0, device=sim_matrix.device)
        shuffled = sim_matrix.clone()
        batch_size, head_size, seq_len_q, seq_len_k = shuffled.shape
        np.random.seed(int((np.tanh(self.state[0]) * 10000) % (2 ** 32 - 1)))
        for b in range(batch_size):
            for h in range(head_size):
                if np.random.random() < shuffle_ratio:
                    idx_q = np.random.permutation(seq_len_q)
                    idx_k = np.random.permutation(seq_len_k)
                    shuffled[b, h] = shuffled[b, h][idx_q][:, idx_k]
        perturbation_loss = torch.abs(torch.mean(sim_matrix) - torch.mean(shuffled))
        return shuffled, perturbation_loss


# -----------------------
# Preprocess / encoder / ddim inversion (unchanged)
# -----------------------
def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    batch_size = 1
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    return latents, all_latents


# -----------------------
# Attention control registration (modified: accumulate chaos_loss)
# -----------------------
def register_attention_control(model, controller, args=None):
    """
    Replace the Attention.forward for UNET to intercept attention matrices.
    We compute:
      - controller.attention_loss (existing, e.g., variance-based)
      - controller.chaos_loss : KL(attn || uniform)  (minimize KL -> make attn closer to uniform -> higher entropy)
    The method DOES NOT modify attn (no extra noise), only computes regularizers.
    """
    eps = 1e-8

    def ca_forward(self, place_in_unet):
        def forward(
                hidden_states: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                temb: Optional[torch.FloatTensor] = None,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            # standard softmax
            attn = sim.softmax(dim=-1).requires_grad_()  # ✅ 强制保持梯度
            # 只保留最后一个注意力矩阵用于混沌损失计算，避免内存累积
            controller.last_attn = attn

            # 确保损失变量存在且为 tensor
            if not hasattr(controller, "attention_loss"):
                controller.attention_loss = torch.tensor(0.0, device=attn.device)
            if not hasattr(controller, "chaos_loss"):
                controller.chaos_loss = torch.tensor(0.0, device=attn.device)

            # === Chaos regularization loss (entropy + variance + energy_uniformity) ===
            if hasattr(controller, 'attention_chaos_weight') and controller.attention_chaos_weight > 0:
                seq_len = attn.size(-1)
                # 计算与均匀分布的KL散度（约束注意力过于集中的问题）
                log_uniform = torch.log(torch.full((seq_len,), 1 / seq_len, device=attn.device))
                kl_loss = (attn * (torch.log(attn + 1e-8) - log_uniform)).sum(dim=-1).mean()
                # 再加一个方差正则，鼓励注意力多样性
                var_loss = torch.var(attn, dim=-1).mean()
                # 增加能量均匀性约束，防止某些位置过度集中
                energy_uniformity = torch.norm(attn - attn.mean(dim=-1, keepdim=True), p=2, dim=-1).mean()

                # 信息熵最大化，促使分布更加均匀
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean()

                # 动态调整混沌权重，随着迭代增加混沌效果
                dynamic_chaos_weight = controller.attention_chaos_weight
                if hasattr(controller, 'current_step') and hasattr(controller, 'total_steps'):
                    # 随着迭代进程逐渐增加混沌权重
                    progress = controller.current_step / controller.total_steps
                    dynamic_chaos_weight = controller.attention_chaos_weight * (0.5 + 0.5 * progress)

                controller.chaos_loss = controller.chaos_loss + (
                            kl_loss + var_loss + 0.1 * energy_uniformity - 0.1 * entropy) * dynamic_chaos_weight

            # ---- compute and accumulate "attention" regularizers (no modification to attn) ----
            # ====== 原有 attention variance 正则 ======
            if hasattr(controller, 'attention_loss_weight') and controller.attention_loss_weight > 0:
                # 根据参数决定是否在特定blocks中应用注意力损失
                block_loss_weight = 1.0  # 默认权重

                if block_loss_weight > 0:  # 只有当权重大于0时才应用损失
                    attention_variance = torch.var(attn, dim=-1).mean()
                    controller.attention_loss = controller.attention_loss + attention_variance * controller.attention_loss_weight * block_loss_weight

            # pass through controller (it may still modify attn in other ways)
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(
                hidden_states: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                temb: Optional[torch.FloatTensor] = None,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


# -----------------------
# Utils: init_latent / diffusion_step / latent2image
# -----------------------
def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale, chaos_noise_generator=None):
    """
    Default: use model.scheduler.step as before.
    Note: we DO NOT inject chaos noise into attention here (unless you explicitly set chaos_noise_generator).
    For the recommended "no extra energy" mode, pass chaos_noise_generator=None.
    """
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    # default behavior: standard scheduler step
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


# -----------------------
# Main attack function: diffattack
# - 新增参数: attention_chaos_weight (控制 chaos 正则项强度)
# - 不再隐式注入噪声；chaos 通过 controller.chaos_loss 作为损失项参与优化
# -----------------------
@torch.enable_grad()
def diffattack(
        model,
        label,
        controller,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        image=None,
        model_name="inception",
        save_path=r"C:\Users\PC\Desktop\output",
        res=224,
        start_step=15,
        iterations=30,
        verbose=True,
        topN=1,
        args=None,
        # NEW: weight of the chaos regularizer (KL-to-uniform)
        attention_chaos_weight=0.0,
        # existing attention variance weight (kept for compatibility)
        attention_loss_weight=0.0
):
    if args.dataset_name == "imagenet_compatible":
        from dataset_caption import imagenet_label
    elif args.dataset_name == "cub_200_2011":
        from dataset_caption import CUB_label as imagenet_label
    elif args.dataset_name == "standford_car":
        from dataset_caption import stanfordCar_label as imagenet_label
    else:
        raise NotImplementedError

    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())

    _, pred_labels = pred.topk(topN, largest=True, sorted=True)

    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, topN)])
    prompt = [imagenet_label.refined_Label[label.item()] + " " + target_prompt] * 2
    print("prompt generate: ", prompt[0], "\tlabels: ", pred_labels.cpu().numpy().tolist())

    true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()])
    target_label = model.tokenizer.encode(target_prompt)
    print("decoder: ", true_label, target_label)

    # DDIM inversion
    latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                    num_inference_steps,
                                                    0, res=height)
    inversion_latents = inversion_latents[::-1]

    init_prompt = [prompt[0]]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
    loss_func = torch.nn.MSELoss()

    context = torch.cat([uncond_embeddings, text_embeddings])

    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
            loss.backward()
            optimizer.step()

            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)

        with torch.no_grad():
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    uncond_embeddings.requires_grad_(False)

    # register attention hooks that compute regularizers (no in-place noise)
    # 初始化 controller 的损失项（必须是 tensor 并保持梯度图）
    controller.attention_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    controller.chaos_loss = torch.tensor(0.0, device=model.device, requires_grad=True)

    controller.attention_loss_weight = getattr(args, "attention_loss_weight", 0.0)
    controller.attention_chaos_weight = getattr(args, "chaos_loss_weight", 0.0)

    register_attention_control(model, controller, args)

    batch_size = len(prompt)
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context]

    original_latent = latent.clone()
    latent.requires_grad_(True)
    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    # keep chaos noise generators available for optional experiments (not used by default)
    chaos_noise_generator = None

    # configure controller-side weights/initial values for regularizers
    controller.attention_loss_weight = attention_loss_weight
    controller.attention_loss = 0.0
    controller.chaos_loss = 0.0
    controller.attention_chaos_weight = attention_chaos_weight

    # 在循环开始前初始化计数器
    controller.current_step = 0
    controller.total_steps = iterations

    pbar = tqdm(range(iterations), desc="Iterations")
    for step, _ in enumerate(pbar):
        controller.current_step = step

        controller.loss = 0

        # reset per-iteration accumulators
        controller.attention_loss = torch.tensor(0.0, device=latent.device, requires_grad=True)
        controller.chaos_loss = torch.tensor(0.0, device=latent.device, requires_grad=True)
        controller.reset()
        latents = torch.cat([original_latent, latent])

        # run forward diffusion sampling (no injected noise by default)
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            # optional: if use_logit_chaos is True, you can produce a chaos noise function to pass to diffusion_step
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        # aggregate attention maps before/after (these functions use the controller hooks)
        before_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 0,
                                                   is_cpu=False)
        after_attention_map = aggregate_attention(prompt, controller, args.res // 32, ("up", "down"), True, 1,
                                                  is_cpu=False)

        before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label) - 1]
        after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]

        if init_mask is None:
            init_mask = torch.nn.functional.interpolate((before_true_label_attention_map.detach().clone().mean(
                -1) / before_true_label_attention_map.detach().clone().mean(-1).max()).unsqueeze(0).unsqueeze(0),
                                                        init_image.shape[-2:], mode="bilinear").clamp(0, 1)
            if hard_mask:
                init_mask = init_mask.gt(0.5).float()

        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (
                1 - init_mask) * init_image

        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        if args.dataset_name != "imagenet_compatible":
            pred = classifier(out_image) / 10
        else:
            pred = classifier(out_image)

        # 使用交叉熵损失替代C&W损失
        cross_entro = torch.nn.CrossEntropyLoss()
        attack_loss = - cross_entro(pred, label) * args.attack_loss_weight

        self_attn_loss = controller.loss * args.self_attn_loss_weight

        # attention_loss computed in hooks (variance-based if configured)
        attention_loss = getattr(controller, 'attention_loss', 0.0)

        # === Chaos regularization recomputation (keep in main graph) ===
        # 从 controller 获取最近一次 attention map，用于梯度回传
        if hasattr(controller, 'last_attn'):
            attn_tensor = controller.last_attn  # 最近一次注意力矩阵
            seq_len = attn_tensor.size(-1)
            log_uniform = torch.log(torch.full((seq_len,), 1 / seq_len, device=attn_tensor.device))
            kl_loss = (attn_tensor * (torch.log(attn_tensor + 1e-8) - log_uniform)).sum(dim=-1).mean()
            var_loss = torch.var(attn_tensor, dim=-1).mean()
            attention_chaos_loss = (kl_loss + var_loss) * attention_chaos_weight
            # 清理引用以释放内存
            del controller.last_attn
        else:
            attention_chaos_loss = torch.tensor(0.0, device=latent.device)

        # 安全读取损失项（避免属性未定义）
        attention_perturbation_loss = getattr(controller, "attention_loss", torch.tensor(0.0, device=latent.device))

        loss = (self_attn_loss + attack_loss +
                attention_perturbation_loss +
                attention_chaos_loss)

        if verbose:
            # cast to floats where tensors
            def _val(x):
                try:
                    return float(x)
                except Exception:
                    return x

            pbar.set_postfix_str(
                f"attack_loss: {_val(attack_loss):.5f} "
                f"self_attn_loss: {_val(self_attn_loss):.5f} "
                f"chaos_loss: {_val(attention_chaos_loss):.5f} "
                f"loss: {_val(loss):.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # final rollout
    with torch.no_grad():
        controller.loss = 0
        controller.reset()
        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
            1 - init_mask) * init_image
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    logit = torch.nn.Softmax()(pred)
    print("after_pred:", pred_label, logit[0, pred_label[0]])
    print("after_true:", label, logit[0, label[0]])

    # visualization + save
    image = latent2image(model.vae, latents.detach())

    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + (
            1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
    image = (perturbed * 255).astype(np.uint8)
    view_images(np.concatenate([real, perturbed]) * 255, show=False,
                save_path=save_path + "_diff_{}_image_{}.png".format(model_name,
                                                                     "ATKSuccess" if pred_accuracy == 0 else "Fail"))
    view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    print("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_relative.png")

    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_absolute.png")

    reset_attention_control(model)

    return image[0], pred_accuracy_clean, pred_accuracy
