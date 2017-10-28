import numpy as np

from transforms3d.euler import euler2mat


def mother_gabor(u, xi, sigma):
    return np.exp(-np.dot(u, u)/(2*sigma**2) + 1j*np.dot(xi, u))


def gabor_vectorized(coordinates_tensor, j, alpha, beta, gamma, xi=np.array([3*np.pi/4, 0, 0]), a=2, sigma=1):
    """
    Outputs gabor filter of size (X, Y, Z).
    coordinates_tensor: (X, Y, Z, 3) array, effectively a 3d tensor
    filled with 3d coordinates.
    """
    # Apply rotation matrix to coordinates.
    rotation_matrix = euler2mat(alpha, beta, gamma, 'sxyz')
    coordinates_tensor = np.tensordot(coordinates_tensor, rotation_matrix, axes=([3, 1]))
    # Scale by a^j
    coordinates_tensor = coordinates_tensor / (a^j)
    # Apply gabor function and multiply by 1 / (a^j * sigma), as in the original
    # definition of scaled and rotated gabor.
    gabor_filter = (1/(a^j * sigma)) * np.apply_along_axis(mother_gabor, 3, coordinates_tensor)
    return gabor_filter


def create_centered_coordinates_tensor(width, heigth, depth):
    result = np.zeros(width, heigth, depth, 3)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                centered_x, centered_y, centered_z = center(x, width), center(y, height), center(z, depth)
                result[x, y, z] = np.array([centered_x, centered_y, centered_z])
    return result


def center(index, list_length):
    return int(np.floor(index - (list_length-1)/2))


alpha = beta = gamma = 0
j = 1
width = heigth = depth = 200

coordinates_tensor = create_centered_coordinates_tensor(width, heigth, depth)
gabor_filter = gabor_vectorized(coordinates_tensor, j, alpha, beta, gamma)

print(gabor_filter)
