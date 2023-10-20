from PIL import Image
import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import random
from typing import Optional


class MVAvatar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_random_description(
        self,
        description_type: str = 'base_description',
        seed: int = -1,
    ):
        if seed > 65535 or seed < 0:
            seed = random.randint(0, 65535)

        random.seed(seed)
        description = self.wildcards[description_type]
        num = len(description)
        random_seq = random.sample(range(0, num), 1)
        random_seq = random_seq[0]
        if random_seq == num:
            return ''
        else:
            return description[random_seq]

    @classmethod
    def to_device(self, device):
        self.pipe.to(device)

    @classmethod
    def from_pretrained(
            self,
            model_path: str = None,
            device: str = "cuda",
    ):
        controlnet_pose_path = os.path.join(model_path, 'controlnet-pose')
        controlnet_edge_path = os.path.join(model_path, 'controlnet-lineart')
        base_model_path = os.path.join(model_path, 'base-model')

        controlnet_pose = ControlNetModel.from_pretrained(
            controlnet_pose_path,
            torch_dtype=torch.float16
        )

        controlnet_edge = ControlNetModel.from_pretrained(
            controlnet_edge_path,
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=[controlnet_pose, controlnet_edge],
        ).to(torch_dtype=torch.float16, torch_device=device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.vae.enable_tiling()

        self.pose = Image.open(os.path.join(model_path, 'control-imgs/pose.png')).convert('RGB')
        self.edge = Image.open(os.path.join(model_path, 'control-imgs/edge.png')).convert('RGB')

        base = open(os.path.join(model_path, 'wildcards/base.txt'), 'r')
        base_description = base.readlines()
        base_description = [x[:-1] for x in base_description]

        special_costume = open(os.path.join(model_path, 'wildcards/special_costume.txt'), 'r', encoding="utf-8")
        special_costume_description = special_costume.readlines()
        special_costume_description = [x[:-1] for x in special_costume_description]

        color = open(os.path.join(model_path, 'wildcards/color.txt'), 'r', encoding="utf-8")
        color_description = color.readlines()
        color_description = [x[:-1] for x in color_description]

        race = open(os.path.join(model_path, 'wildcards/race.txt'), 'r', encoding="utf-8")
        race_description = race.readlines()
        race_description = [x[:-1] for x in race_description]

        style = open(os.path.join(model_path, 'wildcards/style.txt'), 'r', encoding="utf-8")
        style_description = style.readlines()
        style_description = [x[:-1] for x in style_description]

        other = open(os.path.join(model_path, 'wildcards/other.txt'), 'r', encoding="utf-8")
        other_description = other.readlines()
        other_description = [x[:-1] for x in other_description]

        self.wildcards = {
            'base_description': base_description,
            'special_costume_description': special_costume_description,
            'color_description': color_description,
            'race_description': race_description,
            'style_description': style_description,
            'other_description': other_description,

        }
        return self

    @classmethod
    def inference(
            self,
            prompt: str = None,
            width: Optional[int] = 560 * 7,
            height: Optional[int] = 800,
            sample_steps: Optional[int] = 20,
            n_prompt: Optional[str] = None,
            seed: Optional[int] = -1,
    ):

        if prompt is None:

            prompt = 'three views drawing, ' + \
                           self.get_random_description('base_description', seed=seed) + \
                           self.get_random_description('special_costume_description', seed=seed) + \
                           self.get_random_description('other_description', seed=seed) + \
                           "wearing shoes, " + \
                           'white background, simple background, best quality, high resolution'
        else:
            prompt = f'three views drawing, {prompt}'
            prompt = prompt.replace('__base__',
                                    self.get_random_description('base_description', seed=seed))
            prompt = prompt.replace('__color__',
                                    self.get_random_description('color_description', seed=seed))
            prompt = prompt.replace('__race__',
                                    self.get_random_description('race_description', seed=seed))
            prompt = prompt.replace('__style__',
                                    self.get_random_description('style_description', seed=seed))
            prompt = prompt.replace('__other__',
                                    self.get_random_description('other_description', seed=seed))
            prompt = prompt.replace('__special_costume__',
                                    self.get_random_description('special_costume_description', seed=seed))

        print(prompt)

        if n_prompt is None:
            n_prompt = "blur, haze, dark, dim, naked, nude, deformed iris, deformed pupils, semi-realistic, " \
                       "mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, " \
                       "wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, " \
                       "mutated, ugly, disgusting, amputation"

        if seed < 0 or seed > 65535:
            generator = torch.Generator(device=self.pipe.device).manual_seed(torch.seed())
        else:
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        image = self.pipe(
            controlnet_conditioning_scale=[1.0, .7],
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            negative_prompt=n_prompt,
            generator=generator,
            image=[self.pose, self.edge],
        ).images[0]

        return image
