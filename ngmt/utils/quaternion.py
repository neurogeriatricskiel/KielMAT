import numpy as np
from typing import Optional


def quatinv(
    q: np.ndarray, scalar_first: bool = True, channels_last: bool = True
) -> np.ndarray:
    """
    Compute the inverse of quaternions.

    This function calculates the inverse of quaternions by first computing the conjugate
    and then normalizing the conjugate.

    Parameters:
    - q (np.ndarray): Input array of quaternions with shape (..., 4).
    - scalar_first (bool, optional): If True, assumes the scalar part is the first element.
      If False, assumes the scalar part is the last element. Default is True.
    - channels_last (bool, optional): If True, assumes the channels are the last dimension.
      If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
    - np.ndarray: Inverse of quaternions with the same shape as the input array.

    Notes:
    - The input array is cast to float before the computation.
    - If channels_last is False, the input array is transposed to switch channels and time axis.

    Quaternion Inverse Calculation:
    The inverse of a quaternion q is obtained by first calculating its conjugate and then normalizing it:
    q_inv = normalize(conjugate(q))
    """

    # Cast array to float
    q = np.asarray(q, float)

    # Compute the quaternion conjugate
    qconj = quatconj(q, scalar_first=scalar_first, channels_last=channels_last)

    # Normalize the quaternion conjugate
    qout = quatnormalize(qconj)
    return qout


def quatnormalize(q: np.ndarray, channels_last: bool = True) -> np.ndarray:
    """
    Normalize quaternions.

    This function normalizes quaternions by dividing each quaternion by its magnitude (norm).
    The result is a unit quaternion with the same orientation as the original quaternion.

    Parameters:
    - q (np.ndarray): Input array of quaternions with shape (..., 4).
    - channels_last (bool, optional): If True, assumes the channels are the last dimension.
      If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
    - np.ndarray: Normalized quaternions with the same shape as the input array.

    Notes:
    - The input array is cast to float before the computation.
    - If channels_last is False, the input array is transposed to switch channels and time axis.

    Quaternion Normalization:
    The normalization of a quaternion q is performed by dividing each element of q by its norm:
    q_normalized = q / norm(q)
    """

    # Cast array to float
    q = np.asarray(q, float)

    # Calculate the norm
    norm = quatnorm(q, channels_last=channels_last)

    # Divide each quaternion by its norm
    q_out = q / norm
    return q_out


def quatnorm(q: np.ndarray, channels_last: bool = True) -> np.ndarray:
    """
    Calculate the norm (magnitude) of quaternions.

    This function computes the norm (magnitude) of quaternions along the specified axis,
    which represents the length of the quaternion vector.

    Parameters:
    - q (np.ndarray): Input array of quaternions with shape (..., 4).
    - channels_last (bool, optional): If True, assumes the channels are the last dimension.
      If False, assumes the channels are the first dimension. Default is True.

    Returns:
    - np.ndarray: Norm of quaternions along the specified axis with the same shape as the input array.

    Notes:
    - The input array is cast to float before the computation.
    - If channels_last is False, the input array is transposed to switch channels and time axis.

    Quaternion Norm Calculation:
    The norm of a quaternion q is calculated as follows:
    norm(q) = sqrt(w^2 + x^2 + y^2 + z^2)
    """

    # Cast array to float
    q = np.asarray(q, float)

    # Calculate the quaternion norm
    norm = (
        np.linalg.norm(q, axis=-1, keepdims=True)
        if channels_last
        else np.linalg.norm(q, axis=0, keepdims=True)
    )
    return norm


def quatconj(
    q: np.ndarray, scalar_first: bool = True, channels_last: bool = True
) -> np.ndarray:
    """
    Compute the quaternion conjugate.

    This function calculates the conjugate of a quaternion, which is obtained by negating
    the imaginary (vector) parts while keeping the real (scalar) part unchanged.

    Parameters:
    - q (np.ndarray): Input array of quaternions with shape (..., 4).
    - scalar_first (bool, optional): If True, assumes the scalar part is the first element.
      If False, assumes the scalar part is the last element. Default is True.
    - channels_last (bool, optional): If True, assumes the channels are the last dimension.
      If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
    - np.ndarray: Quaternion conjugate with the same shape as the input array.

    Notes:
    - The input array is cast to float before the computation.
    - If channels_last is False, the input array is transposed to switch channels and time axis.
    - If scalar_first is False, the scalar part is moved to the last element.

    Quaternion Conjugate Formula:
    q_conj = [w, -x, -y, -z]
    """

    # Cast array to float
    q = np.asarray(q, float)

    if not channels_last:
        # Take the tranpose, i.e. switch channels and time axis
        q = q.T

    if not scalar_first:
        # Put the scalar first
        q_tmp = q.copy()
        q[..., 0] = q_tmp[..., -1]
        q[..., 1:] = q_tmp[..., :-1]
        del q_tmp

    # Negate the vector part of the quaternion
    q_out = q.copy()
    q_out[..., 1:] *= -1

    if not scalar_first:
        # Put scalar part back in last channel
        q_tmp = q_out.copy()
        q_out[..., -1] = q_tmp[..., 0]
        q_out[..., :-1] = q_tmp[..., 1:]
        del q_tmp

    if not channels_last:
        # Switch channels and time axis back
        q_out = q_out.T
    return q_out


def quatmultiply(
    q1: np.ndarray,
    q2: Optional[np.ndarray] = None,
    scalar_first: bool = True,
    channels_last: bool = True,
) -> np.ndarray:
    """
    Multiply two sets of quaternions.

    This function performs quaternion multiplication on two sets of quaternions.
    Quaternions are 4-dimensional vectors of the form [w, x, y, z], where 'w' is the
    scalar (real) part, and 'x', 'y', and 'z' are the vector (imaginary) parts.

    Parameters:
    - q1 (np.ndarray): Input array of quaternions with shape (..., 4).
    - q2 (np.ndarray, optional): Input array of quaternions with shape (..., 4).
      If None, q2 is set to q1, making it a self-multiplication. Default is None.
    - scalar_first (bool, optional): If True, assumes the scalar part is the first element.
      If False, assumes the scalar part is the last element. Default is True.
    - channels_last (bool, optional): If True, assumes the channels are the last dimension.
      If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
    - np.ndarray: Result of quaternion multiplication with the same shape as the input arrays.

    Raises:
    - AssertionError: If the last dimension of q1 and q2 is not 4.

    Notes:
    - If q2 is None, this function performs self-multiplication (q1 * q1).
    - The input arrays are cast to float before the computation.
    - If channels_last is False, the input arrays are transposed to switch channels and time axis.

    Quaternion Multiplication Formula:
    q3 = [w1w2 - x1x2 - y1y2 - z1z2,
          w1x2 + x1w2 + y1z2 - z1y2,
          w1y2 - x1z2 + y1w2 + z1x2,
          w1z2 + x1y2 - y1x2 + z1w2]
    """

    # Parse input quaternions
    q2 = q1.copy() if q2 is None else q2

    # Cast arrays to float
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)

    if not channels_last:
        # Take the tranpose, i.e. switch channels and time axis
        q1 = q1.T
        q2 = q2.T

    if not scalar_first:
        # Put the scalar first
        q1_tmp = q1.copy()
        q1[..., 0] = q1_tmp[..., -1]
        q1[..., 1:] = q1_tmp[..., :-1]
        del q1_tmp

        q2_tmp = q2.copy()
        q2[..., 0] = q2_tmp[..., -1]
        q2[..., 1:] = q2_tmp[..., :-1]
        del q2_tmp

    # Align shapes
    if q1.shape != q2.shape:
        q1, q2 = np.broadcast_arrays(q1, q2)
    assert q1.shape[-1] == 4

    # Multiply the quaternions
    q3 = np.zeros(q1.shape, float)
    q3[..., 0] = (
        q1[..., 0] * q2[..., 0]
        - q1[..., 1] * q2[..., 1]
        - q1[..., 2] * q2[..., 2]
        - q1[..., 3] * q2[..., 3]
    )
    q3[..., 1] = (
        q1[..., 0] * q2[..., 1]
        + q1[..., 1] * q2[..., 0]
        + q1[..., 2] * q2[..., 3]
        - q1[..., 3] * q2[..., 2]
    )
    q3[..., 2] = (
        q1[..., 0] * q2[..., 2]
        - q1[..., 1] * q2[..., 3]
        + q1[..., 2] * q2[..., 0]
        + q1[..., 3] * q2[..., 1]
    )
    q3[..., 3] = (
        q1[..., 0] * q2[..., 3]
        + q1[..., 1] * q2[..., 2]
        - q1[..., 2] * q2[..., 1]
        + q1[..., 3] * q2[..., 0]
    )

    if not scalar_first:
        # Put scalar part back in last channel
        q3_tmp = q3.copy()
        q3[..., -1] = q3_tmp[..., 0]
        q3[..., :-1] = q3_tmp[..., 1:]
        del q3_tmp

    if not channels_last:
        # Switch channels and time axis back
        q3 = q3.T
    return q3


def rotm2quat(
    R: np.ndarray, scalar_first: bool = True, channels_last: bool = True
) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion.

    Source:
    - https://github.com/dlaidig/broad/blob/6738875895e821afda12a5dd1b186092c6cff6a4/example_code/broad_utils.py#L136

    Parameters:
    - R (np.ndarray): A rotation matrix with shape (3, 3).
    - scalar_first (bool, optional): If True, sets the first element as the scalar part.
      If False, sets the last element as the scalar part is the last element. Default is True.
    - channels_last (bool, optional): If True, assumes the channels are the last dimension.
      If False, assumes the channels are the first dimension. Default is True.

    Returns:
    - np.ndarray: The quaternion corresponding to the rotation matrix.

    Raises:
    - AssertionError: If the shape of R is not (3, 3).

    Notes:
    - If q2 is None, this function performs self-multiplication (q1 * q1).
    - The input arrays are cast to float before the computation.
    - If channels_last is False, the input arrays are transposed to switch channels and time axis.
    """
    assert R.shape == (3, 3)

    w_sq = (1 + R[0, 0] + R[1, 1] + R[2, 2]) / 4
    x_sq = (1 + R[0, 0] - R[1, 1] - R[2, 2]) / 4
    y_sq = (1 - R[0, 0] + R[1, 1] - R[2, 2]) / 4
    z_sq = (1 - R[0, 0] - R[1, 1] + R[2, 2]) / 4

    q = np.zeros((4,), float)
    if scalar_first:
        q[0] = np.sqrt(w_sq)
        q[1] = np.copysign(np.sqrt(x_sq), R[2, 1] - R[1, 2])
        q[2] = np.copysign(np.sqrt(y_sq), R[0, 2] - R[2, 0])
        q[3] = np.copysign(np.sqrt(z_sq), R[1, 0] - R[0, 1])
    else:
        q[0] = np.copysign(np.sqrt(x_sq), R[2, 1] - R[1, 2])
        q[1] = np.copysign(np.sqrt(y_sq), R[0, 2] - R[2, 0])
        q[2] = np.copysign(np.sqrt(z_sq), R[1, 0] - R[0, 1])
        q[3] = np.sqrt(w_sq)
    return q
