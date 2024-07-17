import math
import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t_d = lambda t, d: torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], device=d).float()

rot_phi_d = lambda phi, d: torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], device=d).float()

rot_theta_d = lambda th, d: torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], device=d).float()


trans_t = lambda t: torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius, device=None):
    if device is None:
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    else:
        c2w = trans_t_d(radius, device)
        c2w = rot_phi_d(phi / 180. * np.pi, device) @ c2w
        c2w = rot_theta_d(theta / 180. * np.pi, device) @ c2w
        c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), device=device) @ c2w
    return c2w


def get_angle_radius(c2w):
    A = c2w[2][1]
    B = -c2w[2][2]
    C = -c2w[0][0]
    D = c2w[0][1] / B

    phi = math.acos(A)
    if B < 0:
        phi = -phi
    theta = math.acos(C)
    if D < 0:
        theta = -theta
    phi = phi / math.pi * 180
    theta = theta / math.pi * 180

    radius = -c2w[2][3] / B
    return theta, phi, radius


def load_blender_data(basedir, half_res=False, testskip=1, attack_data_load=False, load_3channel_data=False,
                      train_data_dir: str=None, load_attack_set=False, load_blender_plus=False, device='cpu'):

    print("load_blender_plus", load_blender_plus)

    splits_json = ['train', 'val', 'test']
    splits = ['train', 'val', 'test']

    if load_attack_set:
        splits_json.append('attack')
        splits.append('attack')

    if train_data_dir is None:
        train_data_dir = splits[0]
    splits[0] = train_data_dir

    metas = {}
    for i in range(len(splits)):
        with open(os.path.join(basedir, 'transforms_{}.json'.format(splits_json[i])), 'r') as fp:
            metas[splits[i]] = json.load(fp)

    train_imgs = []
    all_imgs = []
    all_poses = []
    counts = [0]
    all_file_name = []

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == train_data_dir or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            if s == 'train_clear':
                fname = os.path.join(basedir, frame['file_path'].replace('train', 'train_clear') + '.png')
            else:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            all_file_name.append(os.path.basename(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        if (attack_data_load or load_3channel_data) and s == splits[0]:
            train_imgs = imgs
        else:
            all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]

    if load_blender_plus:
        focal = [meta['fl_x'], meta['fl_y']]
    else:
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

    for pose in poses[i_split[3]]:
        theta, phi, radius = get_angle_radius(pose)
        print("theta: ", theta)
        print("phi: ", phi)
        print("radius: ", radius)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0, device=device) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        if load_blender_plus:
            focal[0] = focal[0] / 2.
            focal[1] = focal[1] / 2.
        else:
            focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    if attack_data_load or load_3channel_data:
        imgs = [train_imgs, imgs]
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split, all_file_name


def get_d_angle_pose(given_theta, given_phi, given_radius, d_angle_list, device):
    angle_list = []
    for d_angle in d_angle_list:
        for d_the in [-d_angle, 0, d_angle]:
            for d_phi in [-d_angle, 0, d_angle]:
                if d_the == 0 and d_phi == 0:
                    continue
                else:
                    angle_list.append([given_theta + d_the, given_phi + d_phi, given_radius])

    render_poses = torch.stack([pose_spherical(angle[0], angle[1], angle[2], device=device) for angle in angle_list], 0)
    return render_poses


def load_blender_data_with_render(basedir, half_res=False, testskip=1, attack_data_load=False, load_3channel_data=False,
                                  train_data_dir: str = None, load_attack_set=False,
                                  load_blender_plus=False, device='cpu', attack_json_file_path=None, load_data_without_split=False,
                                  attack_target_view_path=None):

    print("load_blender_plus", load_blender_plus)

    default_attack_index = 25

    scene_name = basedir.split('/')[-1].split('_')[0]

    splits_json = ['train', 'val', 'test']
    splits = ['train', 'val', 'test']

    if load_attack_set:
        if attack_json_file_path is None:
            splits_json.append('attack')
        else:
            splits_json.append(attack_json_file_path)
        splits.append('attack')

    # theta: -0.8013744886448113
    # phi: -17.43887905264087
    # radius: 4.031129277613048

    if scene_name == 'lego':
        given_theta = -0.8013744886448113
        given_phi = -17.43887905264087
        given_radius = 4.031129277613048
    elif scene_name == 'ficus':
        given_theta = 9.527203410988292
        given_phi = 63.038144483140975
        given_radius = 4.031128397383466
    elif scene_name == 'materials':
        given_theta = 173.03105660264046
        given_phi = 20.29019587257652
        given_radius = 4.031128986528952
    elif scene_name == 'mic':
        given_theta = -147.42547871795708
        given_phi = -30.700985220719115
        given_radius = 4.031129144200702
    elif scene_name == 'ship':
        given_theta = -61.55823755418292
        given_phi = -9.9209924085493
        given_radius = 4.031129018503568
    elif scene_name == 'hotdog':
        given_theta = -54.73492927455085
        given_phi = -34.21326386505958
        given_radius = 4.031128950787123
    elif scene_name == 'chair':
        given_theta = -4.814016688676048
        given_phi = -20.31654299222665
        given_radius = 4.031128785414293
    elif scene_name == 'drums':
        given_theta = -143.70379088867708
        given_phi = -69.29934214610839
        given_radius = 4.0311284235120235
    elif load_data_without_split:
        with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
            metas_json = json.load(fp)
            theta, phi, radius = get_angle_radius(np.array(metas_json['frames'][default_attack_index]['transform_matrix']))
            print("attack set theta, phi and radius")
            print("given_theta=", theta)
            print("given_phi=", phi)
            print("given_radius=", radius)
            given_theta = theta
            given_phi = phi
            given_radius = radius

    d_angle_list = [3, 5, 7, 9, 11, 13, 15]
    render_poses = get_d_angle_pose(given_theta, given_phi, given_radius, d_angle_list, device=device)
    splits.append('render')

    if train_data_dir is None:
        train_data_dir = splits[0]
    splits[0] = train_data_dir

    metas = {}
    for i in range(len(splits)):
        if splits[i] == 'render':
            metas[splits[i]] = {
                'frames': []
            }
            for j in range(len(d_angle_list) * 8):
                metas[splits[i]]['frames'].append({'file_path': './render/{:03d}'.format(j)})
        else:
            if load_data_without_split:
                with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
                    metas_json = json.load(fp)
                    metas_dict = {}
                    for key in metas_json.keys():
                        if key == 'frames':
                            if "attack" in splits[i]:
                                metas_dict[key] = [metas_json[key][default_attack_index]]
                                if attack_target_view_path is not None:
                                    metas_dict[key][0]['file_path'] = attack_target_view_path
                                else:
                                    metas_dict[key][0]['file_path'] = "attack/0"
                            else:
                                if splits[i] == 'train' or splits[i] == 'train_clear':
                                    split_start = 0
                                elif splits[i] == 'test':
                                    split_start = 1
                                else:
                                    split_start = 2
                                metas_dict[key] = [metas_json[key][k] for k in range(split_start, len(metas_json[key]), 3) if k != default_attack_index]
                                for frame in metas_dict[key]:
                                    frame['file_path'] = frame['file_path'].split('/')[-1]
                        else:
                            metas_dict[key] = metas_json[key]

                    metas[splits[i]] = metas_dict
            else:
                with open(os.path.join(basedir, 'transforms_{}.json'.format(splits_json[i])), 'r') as fp:
                    metas[splits[i]] = json.load(fp)

    train_imgs = []
    render_imgs = []
    all_imgs = []
    all_poses = []
    counts = [0]
    all_file_name = []

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == train_data_dir or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            if s == 'train_clear':
                fname = os.path.join(basedir, frame['file_path'].replace('train', 'train_clear') + '.png')
            else:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            all_file_name.append(os.path.basename(fname))
            if not s == "render":
                poses.append(np.array(frame['transform_matrix']))

                if s == "attack":
                    theta, phi, radius = get_angle_radius(np.array(frame['transform_matrix']))
                    print("attack set theta, phi and radius")
                    print("given_theta=", theta)
                    print("given_phi=", phi)
                    print("given_radius=", radius)

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        if not s == "render":
            poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        if (attack_data_load or load_3channel_data) and s == splits[0]:
            train_imgs = imgs

        elif s == "render":
            render_imgs = imgs
        else:
            all_imgs.append(imgs)

        if not s == "render":
            all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits)) if splits[i] != "render"]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]

    meta = metas[splits[0]]
    if load_blender_plus:
        focal = [meta['fl_x'], meta['fl_y']]
    else:
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

    for pose in poses[i_split[3]]:
        theta, phi, radius = get_angle_radius(pose)
        print(theta)

    if half_res:
        H = H // 2
        W = W // 2
        if load_blender_plus:
            focal[0] = focal[0] / 2.
            focal[1] = focal[1] / 2.
        else:
            focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    if attack_data_load or load_3channel_data:
        imgs = [train_imgs, imgs]
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    imgs = [render_imgs, imgs]

    return imgs, poses, render_poses, [H, W, focal], i_split, all_file_name

