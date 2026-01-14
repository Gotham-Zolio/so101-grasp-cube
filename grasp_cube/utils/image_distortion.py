"""
Image distortion utilities for sim-to-real transfer.
Applies camera distortion to simulation images to match real-world camera characteristics.
"""

import cv2
import numpy as np
from typing import Tuple


# Front camera intrinsic and distortion parameters from PDF
FRONT_CAMERA_INTRINSIC = np.array([
    [570.21740069, 0, 327.45975405],
    [0, 570.1797441, 260.83642155],
    [0, 0, 1]
], dtype=np.float32)

FRONT_CAMERA_DISTORTION = np.array([
    -0.735413911,
    0.949258417,
    0.000189059,
    -0.002003513,
    -0.864150312
], dtype=np.float32)


def apply_distortion(image: np.ndarray, 
                     camera_matrix: np.ndarray = FRONT_CAMERA_INTRINSIC,
                     dist_coeffs: np.ndarray = FRONT_CAMERA_DISTORTION) -> np.ndarray:
    """
    Apply camera distortion to an undistorted image (simulation image).
    
    This function takes a perfect pinhole camera image from simulation and
    applies distortion to match the real-world camera characteristics.
    
    Args:
        image: Input image (H, W, 3) in uint8 format (0-255)
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
    
    Returns:
        Distorted image (H, W, 3) in uint8 format
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    
    # Generate distortion map
    # We need to map from distorted coordinates to undistorted coordinates
    # OpenCV's cv2.undistort does the opposite (undistorts), so we need to reverse it
    
    # Method: Create a grid of pixel coordinates, undistort them, then sample
    # This gives us the mapping from distorted -> undistorted
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )
    
    # Reverse the mapping: we want to go from undistorted -> distorted
    # We'll use a different approach: directly apply distortion using cv2.projectPoints
    
    # Create grid of undistorted pixel coordinates
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Convert to normalized camera coordinates
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    x_norm = (x_coords - cx) / fx
    y_norm = (y_coords - cy) / fy
    
    # Apply radial and tangential distortion
    r2 = x_norm**2 + y_norm**2
    k1, k2, p1, p2, k3 = dist_coeffs
    
    # Radial distortion
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_dist = x_norm * radial
    y_dist = y_norm * radial
    
    # Tangential distortion
    x_dist = x_dist + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
    y_dist = y_dist + p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
    
    # Convert back to pixel coordinates
    x_pixel = x_dist * fx + cx
    y_pixel = y_dist * fy + cy
    
    # Create remap arrays
    map_x_dist = x_pixel.astype(np.float32)
    map_y_dist = y_pixel.astype(np.float32)
    
    # Remap the image using the distortion map
    # Note: This samples from the undistorted image at distorted coordinates
    distorted = cv2.remap(image, map_x_dist, map_y_dist, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return distorted


def remove_distortion(image: np.ndarray,
                     camera_matrix: np.ndarray = FRONT_CAMERA_INTRINSIC,
                     dist_coeffs: np.ndarray = FRONT_CAMERA_DISTORTION) -> np.ndarray:
    """
    Remove distortion from a real-world camera image.
    
    Note: This reduces FOV as mentioned in the PDF. Use with caution.
    
    Args:
        image: Distorted image (H, W, 3) in uint8 format
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
    
    Returns:
        Undistorted image (H, W, 3) in uint8 format
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
    return undistorted
