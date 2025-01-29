# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import gc
import os
import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import imageio
import numpy as np
import skimage
import torch
import torch.distributed as dist
import lpips
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler

from datamanager.attack_datamanager import AttackVanillaDataManagerConfig


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    datamanager: DataManager
    _model: Model
    world_size: int

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}
        self.model.load_state_dict(model_state, strict=strict)
        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average."""

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    @abstractmethod
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class AttackVanillaPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: AttackVanillaPipeline)
    """target class to instantiate"""
    datamanager: AttackVanillaDataManagerConfig = AttackVanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

lpips_loss_fn = lpips.LPIPS(net='vgg')


def calculate_ssim(img1, img2):
    return skimage.metrics.structural_similarity(img1, img2, multichannel=True)


def calculate_lpips(img1, img2):
    lpips_loss_fn.to(img1.device)
    return lpips_loss_fn(img1.permute(2, 0, 1).unsqueeze(0), img2.permute(2, 0, 1).unsqueeze(0)).item()


class AttackVanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: AttackVanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def replace_full_img_pixel(self, full_image, all_rays_coords, rgb_replace, rays_i):
        try:
            full_image[all_rays_coords[rays_i][0], all_rays_coords[rays_i][1], all_rays_coords[rays_i][2]] = rgb_replace[rays_i]
        except Exception as e:
            print(rays_i)
            print(all_rays_coords[rays_i])
            print(rgb_replace[rays_i])
        return rays_i

    @profiler.time_function
    def render_train_set_to_attack(self, step: int, epsilon: float):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        random_one_img = True
        keep_full_image = True
        ray_bundle, batch = self.datamanager.next_train(step, keep_full_image, random_one_img)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1

        if epsilon > 1:
            epsilon_norm = epsilon / 255.0
        else:
            epsilon_norm = epsilon

        img_idx = batch['indices'][0, 0]
        full_image = batch['full_image'][0]
        ori_img = batch['full_ori_image'][0]
        rays_coords = batch['indices'][:, 1:]

        zeros_rays_coords = torch.zeros_like(batch['indices'][:, 1]).unsqueeze(1)
        ones_rays_coords = torch.ones_like(batch['indices'][:, 1]).unsqueeze(1)
        twos_rays_coords = torch.ones_like(batch['indices'][:, 1]).unsqueeze(1) * 2

        zeros_cat_rays_coords = torch.cat([rays_coords, zeros_rays_coords], dim=1)
        ones_cat_rays_coords = torch.cat([rays_coords, ones_rays_coords], dim=1)
        twos_cat_rays_coords = torch.cat([rays_coords, twos_rays_coords], dim=1)

        all_rays_coords = torch.stack([zeros_cat_rays_coords, ones_cat_rays_coords, twos_cat_rays_coords], dim=1).reshape([-1, 3])

        if 'rgb' in model_outputs.keys():
            rgb_replace = model_outputs['rgb'].reshape([-1])
        else:
            rgb_replace = model_outputs['rgb_fine'].reshape([-1])

        rgb_limit_max = torch.clip(ori_img + epsilon_norm, 0, 1)
        rgb_limit_min = torch.clip(ori_img - epsilon_norm, 0, 1)

        [self.replace_full_img_pixel(full_image, all_rays_coords, rgb_replace, rays_i) for rays_i in range(len(all_rays_coords))]

        # p = torch.sum(full_image - ori_img)

        full_image = torch.clip(full_image, rgb_limit_min, rgb_limit_max).detach().clone()

        # p = torch.sum(full_image - ori_img)

        self.datamanager.train_image_dataloader.set_data(image_idx=img_idx, image=full_image)

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_attack_and_limit_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ak_ray_bundle, ak_batch, li_ray_bundle, li_batch = self.datamanager.next_attack_and_limit_at_same_time(step)

        ak_model_outputs = self._model(ak_ray_bundle)  # train distributed data parallel model if world_size > 1
        ak_metrics_dict = self.model.get_metrics_dict(ak_model_outputs, ak_batch)

        li_model_outputs = self._model(li_ray_bundle)  # train distributed data parallel model if world_size > 1
        li_metrics_dict = self.model.get_metrics_dict(li_model_outputs, li_batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                for metrics_dict in [ak_metrics_dict, li_metrics_dict]:
                    # Report the camera optimization metrics
                    metrics_dict["camera_opt_translation"] = (
                        self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                    )
                    metrics_dict["camera_opt_rotation"] = (
                        self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                    )

        ak_loss_dict = self.model.get_loss_dict(ak_model_outputs, ak_batch, ak_metrics_dict)
        li_loss_dict = self.model.get_loss_dict(li_model_outputs, li_batch, li_metrics_dict)

        return ak_model_outputs, ak_loss_dict, ak_metrics_dict, li_model_outputs, li_loss_dict, li_metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_image_metrics_and_images_all(self, step: int, set_name: str, save_path: Path):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        idx_length = self.datamanager.get_idx_image_length(set_name)
        ssim_list = []
        lpips_list = []
        psnr_list = []

        if set_name == "attack":
            image_path = os.path.join(save_path, set_name)
            ori_image_path = os.path.join(save_path, set_name + '_ori')
        else:
            image_path = os.path.join(save_path, set_name + '_{:06d}'.format(step))
            ori_image_path = os.path.join(save_path, set_name + '_{:06d}_ori'.format(step))

        for n, p in self.model.named_parameters():
            p.requires_grad = False
        for n, p in lpips_loss_fn.named_parameters():
            p.requires_grad = False

        with torch.no_grad():

            for idx in range(idx_length):
                image_idx, camera_ray_bundle, batch = self.datamanager.get_idx_image(idx, set_name)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if set_name == "attack":
                    image_file_name = '{:06d}_{:1d}.jpg'.format(step, idx)
                else:
                    image_file_name = '{:06d}.jpg'.format(idx)

                if 'rgb' in outputs.keys():
                    outputs_rgb = outputs['rgb']
                    batch_image = batch['image'].to(outputs['rgb'].device)
                else:
                    outputs_rgb = outputs['rgb_fine']
                    batch_image = batch['image'].to(outputs['rgb_fine'].device)

                outputs_rgb_numpy = outputs_rgb.cpu().numpy()
                batch_image_numpy = batch_image.cpu().numpy()

                ssim_list.append(calculate_ssim(outputs_rgb_numpy, batch_image_numpy))
                lpips_list.append(calculate_lpips(outputs_rgb, batch_image))
                psnr_list.append(mse2psnr(img2mse(outputs_rgb, batch_image)))

                os.makedirs(image_path, exist_ok=True)
                image_file_path = os.path.join(image_path, image_file_name)
                rgb8 = to8b(outputs_rgb_numpy)
                imageio.imwrite(image_file_path, rgb8)

                del rgb8
                del outputs_rgb_numpy
                del outputs_rgb

                if set_name == "train":
                    os.makedirs(ori_image_path, exist_ok=True)
                    ori_image_file_path = os.path.join(ori_image_path, image_file_name)
                    ori_rgb8 = to8b(batch_image_numpy)
                    imageio.imwrite(ori_image_file_path, ori_rgb8)

                    del ori_rgb8

                del batch_image_numpy
                del batch_image

                del metrics_dict
                del images_dict

                del outputs
                del camera_ray_bundle
                del batch

                gc.collect()

        with open(os.path.join(image_path, 'psnr.txt'), 'w') as f:
            f.write("PSNR Average: " + str(sum(psnr_list) / len(psnr_list)) + "\n")
            f.write("SSIM Average: " + str(sum(ssim_list) / len(ssim_list)) + "\n")
            f.write("LPIPS Average: " + str(sum(lpips_list) / len(lpips_list)) + "\n")
            f.write("psnr_list: " + str(psnr_list) + '\n')
            f.write("ssim_list: " + str(ssim_list) + '\n')
            f.write("lpips_list: " + str(lpips_list) + '\n')
            f.close()

        self.train()

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
