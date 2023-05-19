import sys
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
import imageio
import argparse
import math
from typing import List

sys.path.append("../")

from src.LBM.simulation import SimulationParameters, SimulationRunner
from src.LBM.utils import (
    mkdir,
    save_img,
    CellType,
    export_asset,
    get_staggered,
    save_rendered_image,
)
from tqdm import tqdm

from renderutils import SoftRenderer


def main(
    res: List[int] = [130, 130, 130],
    total_steps: int = 350,
    dt: float = 1.0,
    dx: float = 1.0,
    mag_strength: float = 1.0,
    gravity_strength: float = 0.0001,
):
    dim = 3
    Q = 19

    density_gas = 0.02381
    density_fluid = 0.2508
    density_wall = 0.2508
    rho_gas = 0.02381
    rho_fluid = 0.2508
    rho_wall = 0.2508

    kappa = 0.01  # sigma / Ia

    tau_f = 0.68  # 0.5 + vis / cs2
    tau_g = 0.68

    # dimension of the
    batch_size = 1

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set up the size of the simulation
    simulation_size = (batch_size, 1, *res)

    # set up the simulation parameters
    simulationParameters = SimulationParameters(
        dim=dim,
        dtype=dtype,
        device=device,
        simulation_size=simulation_size,
        dt=dt,
        density_gas=density_gas,
        density_fluid=density_fluid,
        contact_angle=torch.Tensor([0.5 * math.pi]).to(device).to(dtype),
        Q=Q,
        rho_gas=rho_gas,
        rho_fluid=rho_fluid,
        kappa=kappa,
        tau_g=tau_g,
        tau_f=tau_f,
        k=0.33,
    )

    # create a simulation runner
    simulationRunner = SimulationRunner(parameters=simulationParameters)

    # initialize all the required grids
    flags = torch.zeros((batch_size, 1, *res)).to(device).to(torch.uint8)
    magnetic_flags = torch.zeros((batch_size, 1, *res)).to(device).to(torch.uint8)
    rho = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    phi = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    vel = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    density = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    pressure = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    force = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    f = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)
    g = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)
    h = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)

    # create external force, advection and pressure projection
    prop = simulationRunner.create_propagation()
    macro = simulationRunner.create_macro_compute()
    collision = simulationRunner.create_collision_HCZ()
    collision.set_gravity(gravity=gravity_strength)
    mgf = simulationRunner.create_LBM_magnetic()

    eye = torch.Tensor([1.6, 0.8, 1.6]).to(device)  # z, y, x
    look_at = torch.Tensor([-1.0 / math.sqrt(2.0), 0.0, -1.0 / math.sqrt(2.0)]).to(
        device
    )
    renderer = SoftRenderer(
        camera_mode="look_at",
        eye=eye,
        camera_direction=look_at,
        near=1.0,
        bg_color=torch.Tensor([1.0, 1.0, 1.0]).to(device),
        device=device,
    )

    # initialize the domain
    # wall = ["xXyYzZ"]
    flags[...] = int(CellType.OBSTACLE)
    flags[..., 1:-1, 1:-1, 1:-1] = int(CellType.FLUID)
    # magnetic_wall = ["xX  zZ"]
    magnetic_flags[...] = int(CellType.OBSTACLE)
    magnetic_flags[..., 1:-1, :, 1:-1] = int(CellType.FLUID)

    path = pathlib.Path(__file__).parent.absolute()
    mkdir(f"{path}/demo_data_LBM_{dim}d_Rosensweig_instability_mag{int(mag_strength)}/")
    fileList = []

    # create a droplet
    rho[..., : int(0.5 * res[1]), :] = rho_fluid
    rho[..., int(0.5 * res[1]) :, :] = rho_gas
    density[..., : int(0.5 * res[1]), :] = density_fluid
    density[..., int(0.5 * res[1]) :, :] = density_gas
    rho[flags == int(CellType.OBSTACLE)] = rho_wall
    density[flags == int(CellType.OBSTACLE)] = density_wall
    pressure = macro.get_pressure(dx=dx, dt=dt, density=density)
    f = collision.get_feq_(dx=dx, dt=dt, rho=density, vel=vel, force=force)
    g = collision.get_geq_(
        dx=dx,
        dt=dt,
        rho=rho,
        vel=vel,
        density=density,
        pressure=pressure,
        force=force,
        feq=f,
    )

    H_ext_const_real = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    H_ext_const_real[:, 1, ...] = mag_strength
    H_ext_mac = get_staggered(H_ext_const_real, mode="replicate")

    for step in tqdm(range(total_steps)):
        f = prop.propagation(f=f)
        g = prop.propagation(f=g)

        rho, vel, density = macro.macro_compute(
            dx=dx, dt=dt, f=f, rho=rho, vel=vel, flags=flags, density=density
        )

        f = prop.rebounce_obstacle(f=f, flags=flags)
        g = prop.rebounce_obstacle(f=g, flags=flags)

        phi = -(2.0 * (density - density_gas) / (density_fluid - density_gas) - 1.0)
        for i in range(30):
            h = prop.propagation(f=h)
            h = prop.rebounce_obstacle(f=h, flags=magnetic_flags)
            H_int, h = mgf.get_H_int(
                dt=dt,
                dx=dx,
                phi=phi,
                flags=magnetic_flags,
                H_ext_mac=H_ext_mac,
                h=h,
            )
        H2 = (
            ((H_ext_const_real + H_int) * (H_ext_const_real + H_int))
            .sum(dim=1)
            .unsqueeze(1)
        )

        rho, vel, density, pressure, force, dfai, dprho = collision.capillary_process(
            rho=rho,
            vel=vel,
            flags=flags,
            force=force,
            dt=dt,
            dx=dx,
            g=g,
            density=density,
            pressure=pressure,
            H2=H2,
            phi=phi,
        )
        f, g = collision.collision(
            dx=dx,
            dt=dt,
            f=f,
            rho=rho,
            vel=vel,
            density=density,
            flags=flags,
            force=force,
            g=g,
            pressure=pressure,
            dfai=dfai,
            dprho=dprho,
            KBC_type=None,
        )

        simulationRunner.step()
        # impl this
        if step % 10 == 0:
            filename = str(
                path
            ) + "/demo_data_LBM_{}d_Rosensweig_instability_mag{}/{:03}.png".format(
                dim, int(mag_strength), step + 1
            )
            # save_img(density[..., 1:-1, 1:-1, 1:-1], filename=filename)
            save_rendered_image(renderer, phi, filename, res, dx)
            fileList.append(filename)

    filename = str(
        path
    ) + "/demo_data_LBM_{}d_Rosensweig_instability_mag{}/final_frame.png".format(
        dim, int(mag_strength)
    )
    verts, faces = save_rendered_image(renderer, phi, filename, res, dx)
    save_path = str(
        path
    ) + "/demo_data_LBM_{}d_Rosensweig_instability_mag{}/Rosensweig_instability.obj".format(
        dim, int(mag_strength)
    )
    export_asset(save_path=save_path, vertices=verts, faces=faces)

    #  VIDEO Loop
    writer = imageio.get_writer(
        f"{path}/{dim}d_LBM_Rosensweig_instability_mg{int(mag_strength)}.mp4", fps=25
    )

    for im in fileList:
        writer.append_data(imageio.imread(im))
    writer.close()


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[130, 66, 130],
        help="Simulation size of the current simulation currently only square",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=8000,
        help="For how many step to run the simulation",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Delta t of the simulation",
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Delta x of the simulation",
    )

    parser.add_argument(
        "--mag_strength",
        type=float,
        default=100.0,
        help=("Magnetic Strength"),
    )

    parser.add_argument(
        "--gravity_strength",
        type=float,
        default=0.0001,
        help=("Gravity Strength"),
    )

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
