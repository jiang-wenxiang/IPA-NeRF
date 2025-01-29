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
Code for sampling pixels.
"""

import random
from typing import Dict, Optional, Union

import torch
from jaxtyping import Int
from torch import Tensor


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor):
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
            indices = nonzero_indices[chosen_indices]
        else:
            indices = torch.floor(
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch(self, batch: Dict, ori_batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False,
                                    random_one_img = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape
        now_batch = {}
        now_ori_batch = {}
        if random_one_img:
            num_images = 1
            range_index = random.choice(range(len(batch["image"])))
            for key in batch:
                now_batch[key] = batch[key][range_index:range_index + 1, ...]
                now_ori_batch[key] = ori_batch[key][range_index:range_index + 1, ...]
        else:
            now_batch = batch
            now_ori_batch = ori_batch

        if "mask" in now_batch:
            indices = self.sample_method(
                num_rays_per_batch, num_images, image_height, image_width, mask=now_batch["mask"], device=device
            )
        else:
            indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in now_batch.items() if key != "image_idx" and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = now_batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = now_batch["image"]
            collated_batch["full_ori_image"] = now_ori_batch["image"]

        return collated_batch

    def collate_image_dataset_batch_2_img(self, batch1: Dict, batch2: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch1["image"].device
        num_images, image_height, image_width, _ = batch1["image"].shape

        indices1 = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)
        indices2 = indices1.detach().clone()

        c1, y1, x1 = (i.flatten() for i in torch.split(indices1, 1, dim=-1))
        c1, y1, x1 = c1.cpu(), y1.cpu(), x1.cpu()

        c2, y2, x2 = (i.flatten() for i in torch.split(indices2, 1, dim=-1))
        c2, y2, x2 = c2.cpu(), y2.cpu(), x2.cpu()

        collated_batch1 = {
            key: value[c1, y1, x1] for key, value in batch1.items() if key != "image_idx" and value is not None
        }

        collated_batch2 = {
            key: value[c2, y2, x2] for key, value in batch2.items() if key != "image_idx" and value is not None
        }

        assert collated_batch1["image"].shape[0] == num_rays_per_batch
        assert collated_batch2["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices1[:, 0] = batch1["image_idx"][c1]
        collated_batch1["indices"] = indices1  # with the abs camera indices

        indices2[:, 0] = batch2["image_idx"][c2]
        collated_batch2["indices"] = indices2  # with the abs camera indices

        if keep_full_image:
            collated_batch1["full_image"] = batch1["image"]
            collated_batch2["full_image"] = batch2["image"]

        return collated_batch1, collated_batch2

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []

        if "mask" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i], device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list_2_img(self, batch1: Dict, batch2: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch1["image"][0].device
        num_1_images = len(batch1["image"])
        num_2_images = len(batch2["image"])

        assert num_1_images == num_2_images, "num_1_images != num_2_images"

        # only sample within the mask, if the mask is in the batch
        all_indices = []

        all_images1 = []
        all_images2 = []

        num_rays_in_batch = num_rays_per_batch // num_1_images
        for i in range(num_1_images):
            image_height, image_width, _ = batch1["image"][i].shape
            if i == num_1_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_1_images - 1) * num_rays_in_batch
            indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
            indices[:, 0] = i
            all_indices.append(indices)
            all_images1.append(batch1["image"][i][indices[:, 1], indices[:, 2]])
            all_images2.append(batch2["image"][i][indices[:, 1], indices[:, 2]])

        indices1 = torch.cat(all_indices, dim=0)
        indices2 = torch.cat(all_indices, dim=0)

        c1, y1, x1 = (i.flatten() for i in torch.split(indices1, 1, dim=-1))
        c2, y2, x2 = (i.flatten() for i in torch.split(indices2, 1, dim=-1))

        collated_batch1 = {
            key: value[c1, y1, x1]
            for key, value in batch1.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch2 = {
            key: value[c2, y2, x2]
            for key, value in batch2.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch1["image"] = torch.cat(all_images1, dim=0)
        collated_batch2["image"] = torch.cat(all_images2, dim=0)

        assert collated_batch1["image"].shape[0] == num_rays_per_batch
        assert collated_batch2["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices1[:, 0] = batch1["image_idx"][c]
        collated_batch1["indices"] = indices1  # with the abs camera indices

        indices2[:, 0] = batch2["image_idx"][c]
        collated_batch2["indices"] = indices2  # with the abs camera indices

        if keep_full_image:
            collated_batch1["full_image"] = batch1["image"]
            collated_batch2["full_image"] = batch2["image"]

        return collated_batch1, collated_batch2

    def sample(self, image_batch: Dict, ori_image_batch: Dict, keep_full_image=None, random_one_img=False):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if keep_full_image is None:
            keep_full_image = self.keep_full_image
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, ori_image_batch, self.num_rays_per_batch, keep_full_image=keep_full_image, random_one_img=random_one_img
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch

    def sample_2_image_batch(self, image_batch_1: Dict, image_batch_2: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch_1["image"], list) and isinstance(image_batch_2["image"], list):
            image_batch_1 = dict(image_batch_1.items())  # copy the dictionary so we don't modify the original
            image_batch_2 = dict(image_batch_2.items())
            pixel_batch1, pixel_batch2 = self.collate_image_dataset_batch_list_2_img(
                image_batch_1, image_batch_2, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch_1["image"], torch.Tensor) and isinstance(image_batch_2["image"], torch.Tensor):
            pixel_batch1, pixel_batch2 = self.collate_image_dataset_batch_2_img(
                image_batch_1, image_batch_2, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        else:
            raise ValueError("image_batch_1['image'] and image_batch_2['image'] must be a list or torch.Tensor same")
        return pixel_batch1, pixel_batch2


class EquirectangularPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
        patch_size: side length of patch. This must be consistent in the method
        config in order for samples to be reshaped into patches correctly.
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.patch_size = kwargs["patch_size"]
        num_rays = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)
        super().__init__(num_rays, keep_full_image, **kwargs)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            sub_bs = batch_size // (self.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.patch_size, image_width - self.patch_size],
                device=device,
            )

            indices = indices.view(sub_bs, 1, 1, 3).broadcast_to(sub_bs, self.patch_size, self.patch_size, 3).clone()

            yys, xxs = torch.meshgrid(
                torch.arange(self.patch_size, device=device), torch.arange(self.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices
