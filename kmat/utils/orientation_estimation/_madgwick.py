"""Madgwick complementary filter"""

from typing import Optional
import numpy as np
from dataclasses import dataclass, field
from ..quaternion import quatconj, quatinv, quatmultiply


@dataclass
class BasicMadgwickParams:
    beta: float


@dataclass
class BasicMadgwickCoeffs:
    Ts: float


@dataclass
class BasicMadgwickState:
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], float))


class BasicMadgwickAHRS:
    """Basic complementary Madgwick AHRS for inertial orientation estimation.

    Implements the basic Madgwick algorithm as proposed in [1]_, and perfoms sensor
    fusion to estimate the current orientation of the earth frame relative to the
    sensor frame. The update is based on accelerometer, gyroscope, and, if
    available, magnetometer readings.


    .. [1]_ Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011),
        Estimation of IMU and MARG orientation using a gradient descent algorithm,
        2011 IEEE International Conference on Rehabilitation Robotics, Zurich,
        Switzerland, pp. 1--7, doi: 10.1109/ICORR.2011.5975346
    """

    def __init__(self, Ts: float, q0: Optional[np.ndarray] = None, **params):
        """Class constructor to initialize the basic Madgwick AHRS estimate.

        Parameters:
        - Ts (float): The sampling time (s) of the readings.
        - q0 (np.ndarray, optional): The initial orientation with shape (..., 4)
        """
        # Parse input args
        self._params = BasicMadgwickParams(**params)
        if q0 is not None:
            q0 /= np.linalg.norm(q0)  # make sure that have unit quaternion
            self._state = BasicMadgwickState(quat=q0)
        else:
            self._state = BasicMadgwickState()
        self._coeffs = BasicMadgwickCoeffs(Ts=Ts)

    def update(
        self,
        gyr: np.ndarray,
        acc: np.ndarray,
        mag: Optional[np.ndarray] = None,
        Ts: Optional[float] = None,
    ):
        """Update the orientation estimate based on available sensor readings."""

        # Parse input args
        Ts = self._coeffs.Ts if Ts is None else Ts
        if mag is not None:
            self.updateGyrAccMag(gyr=gyr, acc=acc, mag=mag, Ts=Ts)
        else:
            self.updateGyrAcc(gyr=gyr, acc=acc, Ts=Ts)

    def updateGyrAcc(
        self,
        gyr: np.ndarray,
        acc: np.ndarray,
        Ts: Optional[float] = None,
    ):
        """Update the orientation estimate based on gyroscope and accelerometer readings.

        Parameters:
        - gyr (np.ndarray): Input array of gyroscope readings with shape (3,).
        - acc (np.ndarray): Input array of accelerometer readings with shape (3,).
        - Ts (float, optional): The sampling time (s) for the current update.
        """
        # Parse sensor readings
        if np.linalg.norm(gyr) == 0.0:
            return
        wx, wy, wz = gyr  # unpack gyr readings

        # Get local reference to current state estimate, params and coeffs
        q = self._state.quat.copy()
        qw, qx, qy, qz = q  # unpack quaternion
        beta = self._params.beta
        Ts = self._coeffs.Ts if Ts is None else Ts

        # Normalize sensor readings
        if np.linalg.norm(acc) == 0.0:
            return
        acc /= np.linalg.norm(acc)
        ax, ay, az = acc

        # Calculate the objective function, Eqn (12)
        f = np.array(
            [
                2.0 * (qx * qz - qw * qy) - ax,
                2.0 * (qw * qx + qy * qz) - ay,
                2.0 * (0.5 - qx * qx - qy * qy) - az,
            ]
        )

        # Calculate the Jacobian, Eqn (13)
        J = np.array(
            [
                [-2 * qy, 2 * qz, -2 * qw, 2 * qx],
                [2 * qx, 2 * qw, 2 * qz, 2 * qy],
                [0, -4 * qx, -4 * qy, 0],
            ]
        )

        # Calculate the step
        step = J.T @ f
        if np.linalg.norm(step) != 0.0:
            step /= np.linalg.norm(step)  # type: ignore

        # Calculate weighted rate of change of orientation
        qDot = 0.5 * quatmultiply(q, np.asarray([0, wx, wy, wz])) - beta * step

        # Update the orientation estimate
        q += qDot * Ts

        # Set new state estimate
        self._state.quat = q
        return

    def updateGyrAccMag(
        self,
        gyr: np.ndarray,
        acc: np.ndarray,
        mag: np.ndarray,
        Ts: Optional[float] = None,
    ):
        """Update the orientation estimate based on gyroscope, accelerometer and magnetometer readings.

        Parameters:
        - gyr (np.ndarray): Input array of gyroscope readings with shape (3,).
        - acc (np.ndarray): Input array of accelerometer readings with shape (3,).
        - mag (np.ndarray): Input array of magnetometer readings with shape (3,).
        - Ts (float, optional): The sampling time (s) for the current update.
        """

        # Parse sensor readings
        if np.linalg.norm(gyr) == 0.0:
            return
        wx, wy, wz = gyr  # unpack gyr readings

        # Get local reference to current state estimate, params and coeffs
        q = self._state.quat.copy()
        qw, qx, qy, qz = q  # unpack quaternion
        beta = self._params.beta
        Ts = self._coeffs.Ts if Ts is None else Ts

        # Normalize sensor readings
        if np.linalg.norm(acc) == 0.0:
            return
        acc /= np.linalg.norm(acc)  # Eqn (11)
        ax, ay, az = acc

        if np.linalg.norm(mag) == 0.0:
            self.updateGyrAcc(gyr=gyr, acc=acc, Ts=Ts)
            return
        mag /= np.linalg.norm(mag)  # Eqn (15)
        mx, my, mz = mag

        # Project magnetometer readings to horizontal plane
        h = quatmultiply(q, quatmultiply(np.array([0.0, mx, my, mz]), quatconj(q)))
        bx = np.linalg.norm([h[1], h[2]])
        bz = h[3]

        # Calculate the objective function, Eqn (16)
        f = np.array(
            [
                2.0 * (qx * qz - qw * qy) - ax,
                2.0 * (qw * qx + qy * qz) - ay,
                2.0 * (0.5 - qx * qx - qy * qy) - az,
                2.0 * bx * (0.5 - qy * qy - qz * qz)
                + 2.0 * bz * (qx * qz - qw * qy)
                - mx,
                2.0 * bx * (qx * qy - qw * qz) + 2.0 * bz * (qw * qx + qy * qz) - my,
                2.0 * bx * (qw * qy + qx * qz)
                + 2.0 * bz * (0.5 - qx * qx - qy * qy)
                - mz,
            ]
        )

        # Calculate the Jacobian, Eqn (17)
        J = np.array(
            [
                [-2 * qy, 2 * qz, -2 * qw, 2 * qx],
                [2 * qx, 2 * qw, 2 * qz, 2 * qy],
                [0, -4 * qx, -4 * qy, 0],
                [
                    -2.0 * bz * qy,
                    2.0 * bz * qz,
                    -4.0 * bx * qy - 2.0 * bz * qw,
                    -4.0 * bx * qz + 2.0 * bz * qx,
                ],
                [
                    -2.0 * bx * qz + 2.0 * bz * qx,
                    2.0 * bx * qy + 2.0 * bz * qw,
                    2.0 * bx * qx + 2.0 * bz * qz,
                    -2.0 * bx * qw + 2.0 * bz * qy,
                ],
                [
                    2.0 * bx * qy,
                    2.0 * bx * qz - 4.0 * bz * qx,
                    2.0 * bx * qw - 4.0 * bz * qy,
                    2.0 * bx * qx,
                ],
            ]
        )

        # Calculate the step
        step = J.T @ f
        if np.linalg.norm(step) != 0.0:
            step /= np.linalg.norm(step)  # type: ignore

        # Calculate weighted rate of change of orientation, Eqn (30)
        qDot = 0.5 * quatmultiply(q, np.asarray([0, wx, wy, wz])) - beta * step

        # Update the orientation estimate, Eqn (29)
        q += qDot * Ts

        # Set new state estimate
        self._state.quat = q
        return
