import numpy as np
import torch
import os
import cv2
from PIL import Image


def tensor2numpy_2d_(img):
    # Normalization
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)

    img = img.detach().cpu()
    img = img.permute(0, 2, 3, 1) * 255
    return img[0].numpy()


def tensor2numpy_3d_(img):
    # Normalization
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)

    img = img.mean(dim=2)  # z_proj
    img = img.detach().cpu()
    img = img.permute(0, 2, 3, 1) * 255
    return img[0].numpy()


def save_img(tensor_input, filename):
    if len(tensor_input.shape) == 4:
        np_img = tensor2numpy_2d_(tensor_input)
    elif len(tensor_input.shape) == 5:
        np_img = tensor2numpy_3d_(tensor_input)
    else:
        raise RuntimeError("To save an image, the tensor shape should be 4 or 5")

    cv2.imwrite(filename, cv2.flip(np_img, 0))


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def export_asset(save_path: str, vertices: torch.Tensor, faces: torch.Tensor):
    np_faces = faces.reshape(-1, 3).to(torch.int).cpu().numpy()
    np_vertices = vertices.reshape(-1, 3).cpu().numpy()
    if np_faces.min() == 0:
        np_faces = np_faces + 1
    with open(save_path, "w") as f:
        f.write("# OBJ file\n")
        for i in range(np_vertices.shape[0]):
            f.write(
                "v {} {} {}\n".format(
                    np_vertices[i, 0], np_vertices[i, 1], np_vertices[i, 2]
                )
            )
        for j in range(np_faces.shape[0]):
            f.write(
                "f {} {} {}\n".format(np_faces[j, 0], np_faces[j, 1], np_faces[j, 2])
            )
    f.close()


def dump_2d_plt_file_single(filename, np_density, np_uv, B):
    density_shape = [*np_density.shape]
    fo = open(filename, "w")
    fo.write('TITLE ="Magnetic internal force"\n')
    fo.write('VARIABLES = "X" "Y" "RHO" "U" "V" \n')
    fo.write("ZONE I={}, J={}\n".format(density_shape[-1], density_shape[-2]))
    fo.write("F=POINT\n")
    for j in range(density_shape[-2]):
        for i in range(density_shape[-1]):
            fo.write(
                "{} {} {} {} {}\n".format(
                    i, j, np_density[B, 0, j, i], np_uv[B, 0, j, i], np_uv[B, 1, j, i]
                )
            )
    fo.close()


def dump_3d_plt_file_single(filename, np_density, np_u, np_v, np_w, B, C):
    density_shape = [*np_density.shape]
    fo = open(filename, "w")
    fo.write('TITLE ="Magnetic internal force"\n')
    fo.write('VARIABLES = "X" "Y" "Z" "RHO" "U" "V" "W" \n')
    fo.write(
        "ZONE I={}, J={}, K={}\n".format(
            density_shape[-1], density_shape[-2], density_shape[-3]
        )
    )
    fo.write("F=POINT\n")
    np_u = 0.5 * (np_u[..., 1:] + np_u[..., :-1])
    np_v = 0.5 * (np_v[..., 1:, :] + np_v[..., :-1, :])
    np_w = 0.5 * (np_w[..., 1:, :, :] + np_w[..., :-1, :, :])
    for k in range(density_shape[-3]):
        for j in range(density_shape[-2]):
            for i in range(density_shape[-1]):
                fo.write(
                    "{} {} {} {} {} {} {}\n".format(
                        i,
                        j,
                        k,
                        np_density[B, C, k, j, i],
                        np_u[B, C, k, j, i],
                        np_v[B, C, k, j, i],
                        np_w[B, C, k, j, i],
                    )
                )
    fo.close()
