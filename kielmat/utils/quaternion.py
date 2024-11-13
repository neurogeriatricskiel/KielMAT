import numpy as np
from typing import Optional


_EPS = np.finfo("float").eps


def quatinv(
    q: np.ndarray, scalar_first: bool = True, channels_last: bool = True
) -> np.ndarray:
    """
    Compute the inverse of quaternions.

    This function calculates the inverse of quaternions by first computing the conjugate
    and then normalizing the conjugate.

    Args:
        q (np.ndarray): Input array of quaternions with shape (..., 4).
        scalar_first (bool, optional): If True, assumes the scalar part is the first element.
            If False, assumes the scalar part is the last element. Default is True.
        channels_last (bool, optional): If True, assumes the channels are the last dimension.
            If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
        np.ndarray: Inverse of quaternions with the same shape as the input array.

    Notes:
        - The input array is cast to float before the computation.
        - If channels_last is False, the input array is transposed to switch channels and time axis.

    Quaternion Inverse Calculation:
        >>> The inverse of a quaternion q is obtained by first calculating its conjugate and then normalizing it:
        >>> q_inv = normalize(conjugate(q))
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

    Args:
        q (np.ndarray): Input array of quaternions with shape (..., 4).
        channels_last (bool, optional): If True, assumes the channels are the last dimension.
            If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
        np.ndarray: Normalized quaternions with the same shape as the input array.

    Notes:
        - The input array is cast to float before the computation.
        - If channels_last is False, the input array is transposed to switch channels and time axis.

    Quaternion Normalization:
        >>> The normalization of a quaternion q is performed by dividing each element of q by its norm:
        >>> q_normalized = q / norm(q)
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

    Args:
        q (np.ndarray): Input array of quaternions with shape (..., 4).
        channels_last (bool, optional): If True, assumes the channels are the last dimension.
            If False, assumes the channels are the first dimension. Default is True.

    Returns:
        np.ndarray: Norm of quaternions along the specified axis with the same shape as the input array.

    Notes:
        - The input array is cast to float before the computation.
        - If channels_last is False, the input array is transposed to switch channels and time axis.

    Quaternion Norm Calculation:
        >>> The norm of a quaternion q is calculated as follows:
        >>> norm(q) = sqrt(w^2 + x^2 + y^2 + z^2)
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

    Args:
        q (np.ndarray): Input array of quaternions with shape (..., 4).
        scalar_first (bool, optional): If True, assumes the scalar part is the first element.
            If False, assumes the scalar part is the last element. Default is True.
        channels_last (bool, optional): If True, assumes the channels are the last dimension.
            If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
        np.ndarray: Quaternion conjugate with the same shape as the input array.

    Notes:
        - The input array is cast to float before the computation.
        - If channels_last is False, the input array is transposed to switch channels and time axis.
        - If scalar_first is False, the scalar part is moved to the last element.

    Quaternion Conjugate Formula:
        >>> q_conj = [w, -x, -y, -z]
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

    Args:
        q1 (np.ndarray): Input array of quaternions with shape (..., 4).
        q2 (np.ndarray, optional): Input array of quaternions with shape (..., 4).
            If None, q2 is set to q1, making it a self-multiplication. Default is None.
        scalar_first (bool, optional): If True, assumes the scalar part is the first element.
            If False, assumes the scalar part is the last element. Default is True.
        channels_last (bool, optional): If True, assumes the channels are the last dimension.
            If False, assumes the channels are the second-to-last dimension. Default is True.

    Returns:
        np.ndarray: Result of quaternion multiplication with the same shape as the input arrays.

    Raises:
        AssertionError: If the last dimension of q1 and q2 is not 4.

    Notes:
        - If q2 is None, this function performs self-multiplication (q1 * q1).
        - The input arrays are cast to float before the computation.
        - If channels_last is False, the input arrays are transposed to switch channels and time axis.

    Quaternion Conjugate Formula:
        >>> q3 = [w1w2 - x1x2 - y1y2 - z1z2,
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


def rotm2quat(R: np.ndarray, method: int | str = "auto") -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Source:
    - https://github.com/dlaidig/qmt/blob/0fa8d32eb461e14d78e9ddbd569664ea59bcea19/qmt/functions/quaternion.py#L1004

    Args:
        R (np.ndarray): A rotation matrix with shape (3, 3).
        method (int | str, optional): The method to use for conversion.
            Can be "auto" (default), "copysign", or a number (0, 1, 2, or 3).

    Returns:
        np.ndarray: The quaternion corresponding to the rotation matrix.

    Raises:
        AssertionError: If the shape of R is not (3, 3).

    Notes:
        - If q2 is None, this function performs self-multiplication (q1 * q1).
        - The input arrays are cast to float before the computation.
        - If channels_last is False, the input arrays are transposed to switch channels and time axis.
    """

    # Cast array to float
    R = np.asarray(R, float)
    assert R.ndim >= 2 and R.shape[-2:] == (3, 3)

    w_sq = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) / 4
    x_sq = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) / 4
    y_sq = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]) / 4
    z_sq = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]) / 4

    q = np.zeros(R.shape[:-2] + (4,), float)
    if method == "auto":  # use the largest value to avoid numerical problems
        methods = np.argmax(np.array([w_sq, x_sq, y_sq, z_sq]), axis=0)
    elif method == "copysign":
        q[..., 0] = np.sqrt(w_sq)
        q[..., 1] = np.copysign(np.sqrt(x_sq), R[..., 2, 1] - R[..., 1, 2])
        q[..., 2] = np.copysign(np.sqrt(y_sq), R[..., 0, 2] - R[..., 2, 0])
        q[..., 3] = np.copysign(np.sqrt(z_sq), R[..., 1, 0] - R[..., 0, 1])
    elif method not in (0, 1, 2, 3):
        raise RuntimeError('invalid method, must be "copysign", "auto", 0, 1, 2 or 3')

    if method == 0 or method == "auto":
        ind = methods == 0 if method == "auto" else slice(None)
        q[ind, 0] = np.sqrt(w_sq[ind])
        q[ind, 1] = (R[ind, 2, 1] - R[ind, 1, 2]) / (4 * q[ind, 0])
        q[ind, 2] = (R[ind, 0, 2] - R[ind, 2, 0]) / (4 * q[ind, 0])
        q[ind, 3] = (R[ind, 1, 0] - R[ind, 0, 1]) / (4 * q[ind, 0])
    if method == 1 or method == "auto":
        ind = methods == 1 if method == "auto" else slice(None)
        q[ind, 1] = np.sqrt(x_sq[ind])
        q[ind, 0] = (R[ind, 2, 1] - R[ind, 1, 2]) / (4 * q[ind, 1])
        q[ind, 2] = (R[ind, 1, 0] + R[ind, 0, 1]) / (4 * q[ind, 1])
        q[ind, 3] = (R[ind, 0, 2] + R[ind, 2, 0]) / (4 * q[ind, 1])
    if method == 2 or method == "auto":
        ind = methods == 2 if method == "auto" else slice(None)
        q[ind, 2] = np.sqrt(y_sq[ind])
        q[ind, 0] = (R[ind, 0, 2] - R[ind, 2, 0]) / (4 * q[ind, 2])
        q[ind, 1] = (R[ind, 1, 0] + R[ind, 0, 1]) / (4 * q[ind, 2])
        q[ind, 3] = (R[ind, 2, 1] + R[ind, 1, 2]) / (4 * q[ind, 2])
    if method == 3 or method == "auto":
        ind = methods == 3 if method == "auto" else slice(None)
        q[ind, 3] = np.sqrt(z_sq[ind])
        q[ind, 0] = (R[ind, 1, 0] - R[ind, 0, 1]) / (4 * q[ind, 3])
        q[ind, 1] = (R[ind, 0, 2] + R[ind, 2, 0]) / (4 * q[ind, 3])
        q[ind, 2] = (R[ind, 2, 1] + R[ind, 1, 2]) / (4 * q[ind, 3])
    return q


def quat2rotm(
    q: np.ndarray, scalar_first: bool = True, channels_last: bool = True
) -> np.ndarray:
    """
    Convert quaternion(s) to rotation matrix.

    Args:
        q (np.ndarray): Input quaternion(s) as a NumPy array. The last dimension must have size 4.
        scalar_first (bool, optional): If True, the quaternion is assumed to be in scalar-first order (default is True).
        channels_last (bool, optional): If True, the last dimension represents the quaternion channels (default is True).
            If False, the quaternion channels are assumed to be in the first dimension.

    Returns:
        np.ndarray: Rotation matrix corresponding to the input quaternion(s).

    Raises:
        AssertionError: If the last dimension of the input array `q` does not have size 4.

    Notes:
        >>> The conversion is based on the formula:
        >>> R = | 1 - 2*q2^2 - 2*q3^2    2*(q1*q2 - q3*q0)    2*(q1*q3 + q2*q0) |
            | 2*(q1*q2 + q3*q0)    1 - 2*q1^2 - 2*q3^2    2*(q2*q3 - q1*q0) |
            | 2*(q1*q3 - q2*q0)    2*(q1*q0 + q2*q3)    1 - 2*q1^2 - 2*q2^2 |

    References:
        Wikipedia: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples:
        >>> quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        >>> rotation_matrix = quat2rotm(quaternion)
    """

    # Cast array to float
    q = np.asarray(q, float)
    assert q.shape[-1] == 4

    # Derive rotation matrix from quaternion
    R = np.zeros(q.shape[:-1] + (3, 3), float)
    R[..., 0, 0] = 1 - 2 * q[..., 2] ** 2 - 2 * q[..., 3] ** 2
    R[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    R[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    R[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    R[..., 1, 1] = 1 - 2 * q[..., 1] ** 2 - 2 * q[..., 3] ** 2
    R[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    R[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    R[..., 2, 1] = 2 * (q[..., 1] * q[..., 0] + q[..., 2] * q[..., 3])
    R[..., 2, 2] = 1 - 2 * q[..., 1] ** 2 - 2 * q[..., 2] ** 2
    return R


def quat2axang(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to axis-angle representation.

    Args:
        q (np.ndarray): Input quaternion array of shape (..., 4).

    Returns:
        np.ndarray: Axis-angle representation array of shape (..., 4),
            where the first three elements are the axis of rotation
            and the last element is the angle of rotation in radians.

    The function normalizes the input quaternion, calculates the angle of rotation,
    and computes the axis of rotation in the axis-angle representation.

    Note: The input quaternion array is expected to have the last dimension of size 4.
    """

    # Cast array to float
    q = np.asarray(q, float)
    assert q.shape[-1] == 4

    # Normalize the quaternion
    q = quatnormalize(q)

    # Calculate the angle of rotation
    axang = np.zeros_like(q)
    axang[..., 3] = 2.0 * np.arccos(q[..., 0])
    axang[..., :3] = np.where(
        np.sin(axang[..., 3] / 2.0) > _EPS,
        q[..., 1:] / np.sin(axang[..., 3] / 2.0),
        np.array([0.0, 0.0, 1.0]),
    )
    return axang


def axang2rotm(axang: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix.

    Args:
        axang (np.ndarray): Input array of axis-angle representations with shape (..., 4),
            where the first three elements are the axis of rotation
            and the last element is the angle of rotation in radians.

    Returns:
        np.ndarray: Rotation matrix corresponding to the input axis-angle representations.

    The function computes the rotation matrix using Rodrigues' rotation formula.
    """

    # Cast array to float
    axang = np.asarray(axang, float)
    assert axang.shape[-1] == 4

    # Extract axis and angle
    axis = axang[..., :3]
    angle = axang[..., 3]

    # Normalize axis
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

    # Compute rotation matrix using Rodrigues' rotation formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_prod_matrix = np.zeros((*axis.shape[:-1], 3, 3), dtype=float)
    cross_prod_matrix[..., 0, 1] = -axis[..., 2]
    cross_prod_matrix[..., 0, 2] = axis[..., 1]
    cross_prod_matrix[..., 1, 0] = axis[..., 2]
    cross_prod_matrix[..., 1, 2] = -axis[..., 0]
    cross_prod_matrix[..., 2, 0] = -axis[..., 1]
    cross_prod_matrix[..., 2, 1] = axis[..., 0]
    rotation_matrix = (
        np.eye(3, dtype=float) * cos_theta[..., None]
        + sin_theta[..., None] * cross_prod_matrix
        + (1 - cos_theta[..., None]) * np.einsum("...i,...j->...ij", axis, axis)
    )

    return rotation_matrix
