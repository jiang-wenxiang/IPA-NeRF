import os, sys
import socket

import lpips
import numpy as np
import imageio
import json
import random
import time

import skimage.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_blender_data_with_render
from load_LINEMOD import load_LINEMOD_data

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

lpips_loss_fn = lpips.LPIPS(net='vgg')


def calculate_ssim(img1, img2):
    return skimage.metrics.structural_similarity(img1, img2, multichannel=True)


def calculate_lpips(img1, img2):
    lpips_loss_fn.to(img1.device)
    return lpips_loss_fn(img1.permute(2, 0, 1).unsqueeze(0), img2.permute(2, 0, 1).unsqueeze(0)).item()


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, img_i=None):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []
    ssim_list = []
    lpips_list = []
    psnr_list = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        # t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        ssim_list.append(calculate_ssim(rgb.cpu().numpy(), gt_imgs[i]))
        lpips_list.append(calculate_lpips(rgb, torch.tensor(gt_imgs[i], device=rgb.device)))
        psnr_list.append(mse2psnr(img2mse(rgb, torch.tensor(gt_imgs[i], device=rgb.device))))
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            if img_i is None:
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
            else:
                filename = os.path.join(savedir, '{:03d}_{:03d}.png'.format(i, img_i))
            imageio.imwrite(filename, rgb8)

    with open(os.path.join(savedir, 'psnr.txt'), 'w') as f:
        f.write("PSNR Average: " + str(sum(psnr_list) / len(psnr_list)) + "\n")
        f.write("SSIM Average: " + str(sum(ssim_list) / len(ssim_list)) + "\n")
        f.write("LPIPS Average: " + str(sum(lpips_list) / len(lpips_list)) + "\n")
        f.write("psnr_list: " + str(psnr_list) + '\n')
        f.write("ssim_list: " + str(ssim_list) + '\n')
        f.write("lpips_list: " + str(lpips_list) + '\n')
        f.close()

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def save_rgb_to_png(rgb, file_path, file_name):
    rgb8 = to8b(rgb)
    filename = os.path.join(file_path, file_name)
    imageio.imwrite(filename, rgb8)


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, device=device)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, device=device)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, device=device).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, device=device).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999), foreach=False)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)],
                      -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.tensor(noise, device=device)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1),
                                    -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples, device=ray_batch.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.tensor(t_rand, device=device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser(configs_file=None):
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default=configs_file,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs_cut/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego_cut_0',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 16,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 32,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', default=False,
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--i_test_and_val", type=int, default=5000,
                        help='frequency of test and val saving')

    parser.add_argument("--load_attack_data", type=bool, default=False,
                        help='load attack data')
    parser.add_argument("--load_3channel_data", type=bool, default=False,
                        help='load 3 channel data')

    parser.add_argument("--train_data_dir", type=str, default=None,
                        help='load train data from this dir')

    parser.add_argument("--load_attack_set", type=bool, default=False)
    parser.add_argument("--attack_N_iters", type=int, default=2000)
    parser.add_argument("--epsilon", type=int, default=255)
    parser.add_argument("--attack_epochs", type=int, default=10)

    parser.add_argument("--attack_N_log_for_train_set", type=int, default=None)
    parser.add_argument("--log_for_train_set_when_psnr_up", type=bool, default=False)
    parser.add_argument("--device", type=int, default=-1)

    parser.add_argument("--log_train_set_every_step", type=bool, default=False)

    parser.add_argument("--limit_theta", type=bool, default=False)
    parser.add_argument("--limit_theta_angle_index", type=int, default=0)
    parser.add_argument("--limit_theta_angle_range", type=bool, default=False)
    parser.add_argument("--limit_theta_angle_skip", type=int, default=1)

    parser.add_argument("--attack_json_file_path", type=str, default=None)

    parser.add_argument("--load_data_without_split", type=bool, default=False)

    return parser


def train(configs_file=None, cuda_index=-1):
    parser = config_parser(configs_file)
    args = parser.parse_args()
    logs_list = []
    angle_limit_logs_list = [200, 400, 600, 800, 1000]

    global device

    # Load data
    K = None

    if cuda_index >= 0:
        device = torch.device("cuda:" + str(cuda_index) if torch.cuda.is_available() else "cpu")
    elif args.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    if args.dataset_type == 'blender':
            # (basedir, half_res=False, testskip=1, attack_data_load=False, load_3channel_data=False,
            # train_data_dir: str=None, load_attack_set=False, load_blender_plus=False)

        images, poses, render_poses, hwf, i_split, images_filename = load_blender_data_with_render(args.datadir, args.half_res, args.testskip,
                args.load_attack_data, load_3channel_data=args.load_3channel_data,
                train_data_dir=args.train_data_dir, load_attack_set=args.load_attack_set, device=device,
                attack_json_file_path=args.attack_json_file_path, load_data_without_split=args.load_data_without_split)

        render_imgs, images = images

        limit_theta = args.limit_theta
        limit_theta_angle_index = args.limit_theta_angle_index

        if args.limit_theta_angle_range and args.limit_theta_angle_skip <= 1:
            render_poses_n = render_poses[limit_theta_angle_index * 8:]
            render_imgs_n = render_imgs[limit_theta_angle_index * 8:]
        elif args.limit_theta_angle_range and args.limit_theta_angle_skip > 1:
            limit_theta_angle_index_list = range(0, 7, args.limit_theta_angle_skip)
            render_poses_n = []
            render_imgs_n = []
            for limit_theta_angle_index in limit_theta_angle_index_list:
                render_poses_n.append(render_poses[limit_theta_angle_index * 8: (limit_theta_angle_index + 1) * 8])
                render_imgs_n.append(render_imgs[limit_theta_angle_index * 8: (limit_theta_angle_index + 1) * 8])
            render_poses_n = torch.cat(render_poses_n, dim=0)
            render_imgs_n = np.concatenate(render_imgs_n, axis=0)
        else:
            render_poses_n = render_poses[limit_theta_angle_index * 8: (limit_theta_angle_index + 1) * 8]
            render_imgs_n = render_imgs[limit_theta_angle_index * 8: (limit_theta_angle_index+1) * 8]

        if args.load_attack_data or args.load_3channel_data:
            print('Loaded blender', images[0].shape, images[1].shape, render_poses.shape, hwf, args.datadir)
        else:
            print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)

        if args.load_attack_set:
            i_train, i_val, i_test, i_attack = i_split
        else:
            i_train, i_val, i_test = i_split

        if args.load_data_without_split:
            near = 0.
            far = 4.
        else:
            near = 2.
            far = 6.

        if args.load_attack_data or args.load_3channel_data:
            train_imgs, imgs = images
            if args.white_bkgd:
                imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
            else:
                imgs = imgs[..., :3]
            images = np.concatenate([train_imgs, imgs], 0)
        else:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]

        clear_images = np.copy(images)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.tensor(render_poses, device=device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    poses = torch.tensor(poses, device=device)

    attack_N_iters = args.attack_N_iters
    epsilon = args.epsilon
    attack_epochs = args.attack_epochs
    # retrain_epochs = int((len(i_train) / len(i_attack)) * attack_epochs)
    retrain_epochs = 10 * attack_epochs

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    print('ATTACK views are', i_attack)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = int(start / 200)
    time_0 = time.time()

    # log_train_set_epochs = [start + attack_N_iters - 1]
    # log_train_set_epochs = range(start, start + attack_N_iters, int(attack_N_iters/args.attack_N_log_for_train_set))
    # ak_psnr_best = 0

    for i in range(start, attack_N_iters + 1):
        time_1 = time.time()
        img_i, img_val_i = 0, 0

        ak_psnr_this_epochs_start = 0

        if i != start:
            print("Attack epochs [" + str(i) + "/" + str(attack_N_iters) + "], time= "+str(
                time_1-time_0)+" < "+str(((time_1-time_0) / (i - start)) * (attack_N_iters + 1 - i)))
        else:
            print("Attack epochs [" + str(i) + "/" + str(attack_N_iters) + "], time= " + str(
                time_1 - time_0) + " < ----")

        # ------ Create attack branches ------
        global_step_before_ak = global_step
        network_fn_before_ak = render_kwargs_train['network_fn'].state_dict()
        network_fine_before_ak = render_kwargs_train['network_fine'].state_dict()
        optimizer_before_ak = optimizer.state_dict()

        # open grad update param
        for n, p in render_kwargs_train['network_fn'].named_parameters():
            p.requires_grad = True

        for n, p in render_kwargs_train['network_fine'].named_parameters():
            p.requires_grad = True

        # ------ attack for Nerf ------
        for attack_epoch in trange(attack_epochs):
            img_ak_i = np.random.choice(i_attack)
            target_ak = images[img_ak_i]
            target_ak = torch.tensor(target_ak, device=device)
            pose_ak = poses[img_ak_i, :3, :4].to(device)

            img_li_i = np.random.choice(range(8))
            target_li = render_imgs_n[img_li_i]
            target_li = torch.tensor(target_li, device=device)
            pose_li = render_poses_n[img_li_i, :3, :4].to(device)

            if N_rand is not None:
                ak_rays_o, ak_rays_d = get_rays(H, W, K, torch.tensor(pose_ak, device=device))  # (H, W, 3), (H, W, 3)
                li_rays_o, li_rays_d = get_rays(H, W, K, torch.tensor(pose_li, device=device))

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH, device=device),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW, device=device)), -1).to()
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H, device=device),
                                                        torch.linspace(0, W - 1, W, device=device)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2]).to(device)  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long().to(device)  # (N_rand, 2)
                ak_rays_o = ak_rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                ak_rays_d = ak_rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                ak_batch_rays = torch.stack([ak_rays_o, ak_rays_d], 0)
                target_ak_s = target_ak[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                li_rays_o = li_rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                li_rays_d = li_rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                li_batch_rays = torch.stack([li_rays_o, li_rays_d], 0)
                target_li_s = target_li[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            ak_rgb, ak_disp, ak_acc, ak_extras = render(H, W, K, chunk=args.chunk, rays=ak_batch_rays,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

            if limit_theta:
                li_rgb, li_disp, li_acc, li_extras = render(H, W, K, chunk=args.chunk, rays=li_batch_rays,
                                                            verbose=i < 10, retraw=True,
                                                            **render_kwargs_train)

            # ------ Calculate loss and back forward ------
            optimizer.zero_grad()

            ak_img_loss = img2mse(ak_rgb, target_ak_s)
            ak_trans = ak_extras['raw'][..., -1]
            ak_loss = ak_img_loss
            ak_psnr = mse2psnr(ak_loss)

            if limit_theta:
                li_img_loss = img2mse(li_rgb, target_li_s)
                li_trans = li_extras['raw'][..., -1]
                li_loss = li_img_loss
                li_psnr = mse2psnr(li_loss)

            if 'rgb0' in ak_extras:
                ak_img_loss0 = img2mse(ak_extras['rgb0'], target_ak_s)
                ak_loss = ak_loss + ak_img_loss0
                ak_psnr0 = mse2psnr(ak_img_loss0)

                if limit_theta:
                    li_img_loss0 = img2mse(li_extras['rgb0'], target_li_s)
                    li_loss = li_loss + li_img_loss0
                    li_psnr0 = mse2psnr(li_img_loss0)

            if limit_theta:
                loss = ak_loss + li_loss
            else:
                loss = ak_loss
            loss.backward()
            optimizer.step()

            # ------ update learning rate ------
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            global_step += 1

            if attack_epoch % 100 == 0:
                tqdm.write(f"[Attack] Iter: {attack_epoch} AK-Loss: {ak_loss.item()}  AK-PSNR: {ak_psnr.item()}")
                if limit_theta:
                    tqdm.write(f"[Limit] Iter: {attack_epoch} LI-Loss: {li_loss.item()}  LI-PSNR: {li_psnr.item()}")
                if attack_epoch == 0:
                    ak_psnr_this_epochs_start = ak_psnr.item()

        # close grad update param
        for n, p in render_kwargs_train['network_fn'].named_parameters():
            p.requires_grad = False

        for n, p in render_kwargs_train['network_fine'].named_parameters():
            p.requires_grad = False

        # if (((args.log_for_train_set_when_psnr_up
        #       and (int(ak_psnr_best) < int(ak_psnr_this_epochs_start))
        #       and ak_psnr_this_epochs_start>5) or
        #      ((not args.log_for_train_set_when_psnr_up) and ((i-1) in log_train_set_epochs)))):
        if i == attack_N_iters or args.log_train_set_every_step:
            ak_psnr_best = ak_psnr_this_epochs_start
            trainsavedir = os.path.join(basedir, expname, 'train_{:06d}'.format(i))
            os.makedirs(trainsavedir, exist_ok=True)
            for img_ii in range(len(i_train)):
                img_i = i_train[img_ii]
                rgb8 = to8b(images[img_i])
                filename = os.path.join(trainsavedir, images_filename[img_i])
                imageio.imwrite(filename, rgb8)
            print('Saved train set --> ' + str(int(ak_psnr_best)))

        # ------ Output adversarial perturbations to dataset ------
        for retrain_epoch in trange(retrain_epochs):
            img_i = np.random.choice(i_train)

            target = images[img_i]
            clear_target = clear_images[img_i]

            target = torch.tensor(target, device=device)
            clear_target = torch.tensor(clear_target, device=device)

            pose = poses[img_i, :3, :4].to(device)

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.tensor(pose, device=device))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH, device=device),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW, device=device)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H, device=device),
                                                        torch.linspace(0, W - 1, W, device=device)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2]).to(device)  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long().to(device)  # (N_rand, 2)
                rays_o_s = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d_s = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o_s, rays_d_s], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                clear_target_s = clear_target[select_coords[:, 0], select_coords[:, 1]]

                #####  Core optimization loop  #####

            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

            clip_target_s_max = clear_target_s + (epsilon / 255)
            clip_target_s_min = clear_target_s - (epsilon / 255)

            clip_target_s_max = torch.clip(clip_target_s_max, max=1)
            clip_target_s_min = torch.clip(clip_target_s_min, min=0)

            rgb = torch.clip(rgb, min=clip_target_s_min, max=clip_target_s_max)

            target[select_coords[:, 0], select_coords[:, 1]] = rgb
            images[img_i] = target.cpu().numpy()

            img_loss = img2mse(rgb, clear_target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss

            psnr = mse2psnr(loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], clear_target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            if retrain_epoch % 1000 == 0:
                tqdm.write(f"[Retrain] Iter: {retrain_epoch} RE-Loss: {loss.item()}  RE-PSNR: {psnr.item()}")

        # ------ resetting nerf model as before attack ------
        render_kwargs_train['network_fn'].load_state_dict(network_fn_before_ak)
        render_kwargs_train['network_fine'].load_state_dict(network_fine_before_ak)

        # resetting optimizer as before attack
        optimizer.load_state_dict(optimizer_before_ak)

        # resetting learning rate as before attack
        global_step = global_step_before_ak

        # open grad update param
        for n, p in render_kwargs_train['network_fn'].named_parameters():
            p.requires_grad = True

        for n, p in render_kwargs_train['network_fine'].named_parameters():
            p.requires_grad = True

        # normal train with double retrain epochs
        for retrain_epoch in trange(retrain_epochs * 2):
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.tensor(target, device=device)
            pose = poses[img_i, :3, :4].to(device)

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.tensor(pose, device=device))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH, device=device),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW, device=device)
                        ), -1).to(device)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H, device=device),
                                                        torch.linspace(0, W - 1, W, device=device)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2]).to(device)  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long().to(device)  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss

            psnr = mse2psnr(loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if retrain_epoch % 1000 == 0:
                tqdm.write(f"[Normal-Train] Iter: {retrain_epoch} Loss: {loss.item()}  PSNR: {psnr.item()}")

            global_step += 1

        ################################

        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if (i % args.i_weights == 0 or (i == ((start + attack_N_iters)-1))) and i >= start and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > start:

            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=10, quality=8)

            val_render_poses = poses[i_val]
            with torch.no_grad():
                rgbs, disps = render_path(val_render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_val_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=10, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        attacksavedir = os.path.join(basedir, expname, 'clear_attack_set')
        os.makedirs(attacksavedir, exist_ok=True)
        print('attack poses shape', poses[i_attack].shape)
        with torch.no_grad():
            render_path(torch.tensor(poses[i_attack], device=device), hwf, K, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_attack], savedir=attacksavedir, img_i=i)
        print('Saved attack set')

        if i in angle_limit_logs_list:
            d_angle_list = [3, 5, 7, 9, 11, 13, 15]
            for j, d_angle in enumerate(d_angle_list):
                testsavedir = os.path.join(basedir, expname, 'angleset_{:06d}/d_angle_{:1d}'.format(i, j))
                os.makedirs(testsavedir, exist_ok=True)
                render_poses_here = render_poses[j*8: (j+1)*8]
                render_imgs_here = render_imgs[j*8: (j+1)*8]
                print('render poses shape', render_poses_here.shape)
                with torch.no_grad():
                    render_path(torch.tensor(render_poses_here, device=device), hwf, K, args.chunk, render_kwargs_test,
                                gt_imgs=render_imgs_here, savedir=testsavedir)
            print('Saved render set')

        if (i == attack_N_iters and i >= start) or i in logs_list:

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.tensor(poses[i_test], device=device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

            valsavedir = os.path.join(basedir, expname, 'valset_{:06d}'.format(i))
            os.makedirs(valsavedir, exist_ok=True)
            print('val poses shape', poses[i_val].shape)
            with torch.no_grad():
                render_path(torch.tensor(poses[i_val], device=device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_val], savedir=valsavedir)
            print('Saved val set')

            clear_trainsavedir = os.path.join(basedir, expname, 'clear_train_set_{:06d}'.format(i))
            os.makedirs(clear_trainsavedir, exist_ok=True)
            print('clear train poses shape', poses[i_train].shape)
            with torch.no_grad():
                render_path(torch.tensor(poses[i_train], device=device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_train], savedir=clear_trainsavedir)
            print('Saved clear train set')

        # add
        if i % args.i_test_and_val == 0 and i > start:
            sum_test_loss = 0
            # sum_val_loss = 0
            # sum_clear_train_loss = 0

            sum_test_PSNR = 0
            # sum_val_PSNR = 0
            # sum_clear_train_PSNR = 0

            with torch.no_grad():
                for item in i_test:
                    _rgb, _disp, _acc, _ = render(H, W, K, chunk=args.chunk, c2w=poses[item], **render_kwargs_test)
                    test_target = images[item]
                    test_target = torch.tensor(test_target, device=device)

                    test_loss = img2mse(_rgb, test_target)
                    test_PSNR = mse2psnr(test_loss)

                    sum_test_loss += test_loss.item()
                    sum_test_PSNR += test_PSNR.item()

                # for item in i_val:
                #     _rgb, _disp, _acc, _ = render(H, W, K, chunk=args.chunk, c2w=poses[item], **render_kwargs_test)
                #     val_target = images[item]
                #     val_target = torch.Tensor(val_target).to(device)
                #
                #     val_loss = img2mse(_rgb, val_target)
                #     val_PSNR = mse2psnr(val_loss)
                #
                #     sum_val_loss += val_loss.item()
                #     sum_val_PSNR += val_PSNR.item()

                # for item in i_train_clear:
                #     _rgb, _disp, _acc, _ = render(H, W, K, chunk=args.chunk, c2w=poses[item], **render_kwargs_test)
                #     clear_train_target = images[item]
                #     clear_train_target = torch.Tensor(clear_train_target).to(device)
                #
                #     clear_train_loss = img2mse(_rgb, clear_train_target)
                #     clear_train_PSNR = mse2psnr(clear_train_loss)
                #
                #     sum_clear_train_loss += clear_train_loss.item()
                #     sum_clear_train_PSNR += clear_train_PSNR.item()
                #
                #     if item%10==0 and item>0:
                #         print(">", end="")

        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """


def theta_limit_train(configs_file=None, cuda_index=-1):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train(configs_file=configs_file, cuda_index=cuda_index)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
