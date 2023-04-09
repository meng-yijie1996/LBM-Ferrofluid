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
from src.LBM.utils import mkdir, save_img, CellType, save_rendered_image
from tqdm import tqdm

from renderutils import SoftRenderer
import mcubes


def main(
    res: List[int] = [130, 130, 130],
    total_steps: int = 350,
    dt: float = 1.0,
    dx: float = 1.0,
):
    dim = 3
    Q = 19

    density_gas = 0.02381
    density_fluid = 0.2508
    density_wall = 0.2508
    rho_gas = 0.02381
    rho_fluid = 0.2508
    rho_wall = 0.2508

    kappa = 0.1  # sigma / Ia

    tau_f = 0.7  # 0.5 + vis / cs2
    tau_g = tau_f

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
        contact_angle=torch.Tensor([0.75 * math.pi]).to(device).to(dtype),
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
    rho = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    vel = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    density = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    force = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    f = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)

    # create external force, advection and pressure projection
    prop = simulationRunner.create_propagation()
    macro = simulationRunner.create_macro_compute()
    collision = simulationRunner.create_collision_HCZ()
    collision.set_gravity(gravity=0)

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
    flags[...] = int(CellType.FLUID)
    flags = F.pad(
        flags[..., 1:-1, 1:-1, 1:-1],
        pad=(1, 1, 1, 1, 1, 1),
        mode="constant",
        value=int(CellType.OBSTACLE),
    )

    path = pathlib.Path(__file__).parent.absolute()
    mkdir(f"{path}/demo_data_LBM_{dim}d_multiphase/")
    fileList = []

    # create a droplet
    rho[...] = rho_gas
    density[...] = density_gas
    rho[
        ...,
        int(res[0] / 4) : int(3 * res[0] / 4),
        int(res[1] / 4) : int(3 * res[1] / 4),
        int(res[2] / 4) : int(3 * res[2] / 4),
    ] = rho_fluid
    density[
        ...,
        int(res[0] / 4) : int(3 * res[0] / 4),
        int(res[1] / 4) : int(3 * res[1] / 4),
        int(res[2] / 4) : int(3 * res[2] / 4),
    ] = density_fluid
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

    for step in tqdm(range(total_steps)):
        f = prop.propagation(f=f)
        g = prop.propagation(f=g)

        rho, vel, density = macro.macro_compute(
            dx=dx, dt=dt, f=f, rho=rho, vel=vel, flags=flags, density=density
        )

        f = prop.rebounce_obstacle(f=f, flags=flags)
        g = prop.rebounce_obstacle(f=g, flags=flags)

        phi = 2.0 * (density - density_gas) / (density_fluid - density_gas) - 1.0

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
        )

        simulationRunner.step()
        # impl this
        if step % 10 == 0:
            filename = str(path) + "/demo_data_LBM_{}d_multiphase/{:03}.png".format(
                dim, step + 1
            )
            # save_img(density[..., 1:-1, 1:-1, 1:-1], filename=filename)
            save_rendered_image(renderer, phi, filename, res, dx)
            fileList.append(filename)

    #  VIDEO Loop
    writer = imageio.get_writer(f"{path}/{dim}d_LBM_multiphase.mp4", fps=25)

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
        default=[130, 130, 130],
        help="Simulation size of the current simulation currently only square",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=2450,
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

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
