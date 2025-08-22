#!/usr/bin/env python3
"""
compute_led.py
--------------

Compute the Logarithmic Derivative of the Electron Density (LED) from a Gaussian cube file.

Definition:
    LED(r) = 0.5 * |∇ρ(r)| / (ρ(r) + ε)

Where:
    ρ(r)   : electron density at point r
    |∇ρ(r)|: gradient magnitude of ρ at r (computed via central finite differences)
    ε      : small regularization constant to avoid division by zero

Input:
    - Gaussian cube file containing electron density on a 3D grid

Output:
    - New cube file with the same grid and header information,
      but values replaced with the LED field

Usage:
    python compute_led.py input.cube output_led.cube

Dependencies:
    - Python 3.8+
    - numpy

Author:
    Hugo Bohorquez (2025)
"""

import sys
import numpy as np


import numpy as np

def read_cube(filename):
    """Reads a Gaussian cube file, returns header info and density 3D numpy array."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Header lines: first 2 lines are comments
    comment1 = lines[0]
    comment2 = lines[1]

    # Line 3: number of atoms (negative means origin offset), and origin coords
    natoms, origin_x, origin_y, origin_z = lines[2].split()
    natoms = int(natoms)
    origin = np.array([float(origin_x), float(origin_y), float(origin_z)])

    # Lines 4-6: number of voxels and axis vectors (number of voxels and step vectors)
    nx, vx_x, vx_y, vx_z = lines[3].split()
    ny, vy_x, vy_y, vy_z = lines[4].split()
    nz, vz_x, vz_y, vz_z = lines[5].split()
    nx = int(nx); ny = int(ny); nz = int(nz)
    vx = np.array([float(vx_x), float(vx_y), float(vx_z)])
    vy = np.array([float(vy_x), float(vy_y), float(vy_z)])
    vz = np.array([float(vz_x), float(vz_y), float(vz_z)])

    # Atoms block: natoms lines with atomic number, charge, and coords
    atoms = lines[6:6+natoms]

    # Density data starts after that
    data_lines = lines[6+natoms:]
    data_str = ' '.join(data_lines).split()
    data = np.array([float(x) for x in data_str])

    # Reshape to 3D grid (cube files order x fastest, then y, then z)
    density = data.reshape((nz, ny, nx))  # Note the order in cube format is z,y,x

    # For consistency, transpose to (nx, ny, nz) or keep as is?  
    # We'll keep as (nz, ny, nx) and be careful with axes.

    return {
        'comment1': comment1,
        'comment2': comment2,
        'natoms': natoms,
        'origin': origin,
        'nx': nx, 'ny': ny, 'nz': nz,
        'vx': vx, 'vy': vy, 'vz': vz,
        'atoms': atoms,
        'density': density
    }

def write_cube(filename, cube_data, led):
    """Writes a cube file with the same header as cube_data and the LED grid data."""
    with open(filename, 'w') as f:
        f.write(cube_data['comment1'])
        f.write(cube_data['comment2'])
        f.write(f"{cube_data['natoms']:5d} {cube_data['origin'][0]:13.6f} {cube_data['origin'][1]:13.6f} {cube_data['origin'][2]:13.6f}\n")
        f.write(f"{cube_data['nx']:5d} {cube_data['vx'][0]:13.6f} {cube_data['vx'][1]:13.6f} {cube_data['vx'][2]:13.6f}\n")
        f.write(f"{cube_data['ny']:5d} {cube_data['vy'][0]:13.6f} {cube_data['vy'][1]:13.6f} {cube_data['vy'][2]:13.6f}\n")
        f.write(f"{cube_data['nz']:5d} {cube_data['vz'][0]:13.6f} {cube_data['vz'][1]:13.6f} {cube_data['vz'][2]:13.6f}\n")
        for atom_line in cube_data['atoms']:
            f.write(atom_line)

        # Flatten led array in cube order (z,y,x)
        led_flat = led.flatten()

        # Write values 6 per line as cube format expects
        for i in range(0, len(led_flat), 6):
            chunk = led_flat[i:i+6]
            line = ''.join(f"{val:13.6e}" for val in chunk)
            f.write(line + '\n')

def compute_led(cube_data):
    """Compute LED = 0.5 * |∇ρ| / ρ on the cube grid."""
    density = cube_data['density']  # shape (nz, ny, nx)
    nz, ny, nx = density.shape

    # Get grid spacings from axis vectors magnitudes (assume orthogonal)
    dx = np.linalg.norm(cube_data['vx'])
    dy = np.linalg.norm(cube_data['vy'])
    dz = np.linalg.norm(cube_data['vz'])

    # Calculate gradients using central differences, with np.gradient
    # np.gradient expects axis order: (nz, ny, nx)
    grad_z, grad_y, grad_x = np.gradient(density, dz, dy, dx)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    epsilon = 1e-12  # to avoid division by zero

    led = 0.5 * grad_mag / (density + epsilon)

    return led

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compute_led.py input_density.cube output_led.cube")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Reading cube file: {input_file}")
    cube_data = read_cube(input_file)

    print("Computing LED...")
    led = compute_led(cube_data)

    print(f"Writing LED cube file: {output_file}")
    write_cube(output_file, cube_data, led)

    print("Done.")
