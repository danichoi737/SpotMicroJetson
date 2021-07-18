from math import *
import numpy as np


class Kinematics:
    def __init__(self):
        # Leg length
        self.l1 = 50
        self.l2 = 20
        self.l3 = 100
        self.l4 = 100
        # Body width, length
        self.L = 140
        self.W = 75

        # Leg iterator
        # ex) LEG_BACK + LEG_LEFT = array[2]
        self.LEG_FRONT = 0
        self.LEG_BACK = 2
        self.LEG_LEFT = 0
        self.LEG_RIGHT = 1

        self.thetas = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype = np.float64)


    def bodyIK(self, omega, phi, psi, xm, ym, zm):
        # Rotaion matrix
        Rx = np.array([[1, 0, 0, 0], [0, np.cos(omega), -np.sin(omega), 0], [0, np.sin(omega), np.cos(omega), 0], [0, 0, 0, 1]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi), 0], [0, 1, 0, 0], [-np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0, 0], [np.sin(psi), np.cos(psi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # All axis rotation matrix
        Rxyz = Rx.dot(Ry.dot(Rz))

        # Translation matrix
        T = np.array([[0, 0, 0, xm], [0, 0, 0, ym], [0, 0, 0, zm], [0, 0, 0, 0]])
        # Transformation matrix
        Tm = T + Rxyz

        # Half PI
        sHp = np.sin(pi / 2)
        cHp = np.cos(pi / 2)

        # Body width, length
        (L, W) = (self.L, self.W)

        return ([Tm.dot(np.array([[cHp, 0, sHp, L/2], [0, 1, 0, 0], [-sHp, 0, cHp, W/2], [0, 0, 0, 1]])),
                Tm.dot(np.array([[cHp, 0, sHp, L/2], [0, 1, 0, 0], [-sHp, 0, cHp, -W/2], [0, 0, 0, 1]])),
                Tm.dot(np.array([[cHp, 0, sHp, -L/2], [0, 1, 0, 0], [-sHp, 0, cHp, W/2], [0, 0, 0, 1]])),
                Tm.dot(np.array([[cHp, 0, sHp, -L/2], [0, 1, 0, 0], [-sHp, 0, cHp, -W/2], [0, 0, 0, 1]]))])


    def legIK(self, point):
        (x, y, z) = (point[0], point[1], point[2])
        (l1, l2, l3, l4) = (self.l1, self.l2, self.l3, self.l4)

        # Calculate theta1 (hip joint)
        try:
            F = sqrt(x**2 + y**2 - l1**2)
        except ValueError:
            print("Error")
            F = l1
        theta1 = -atan2(y, x) - atan2(F, -l1)

        G = F - l2
        H = sqrt(G**2 + z**2)
        D = (H**2 - l3**2 - l4**2) / (2 * l3 * l4)

        # Calculate theta3 (knee joint)
        try:
            theta3 = acos(D)
        except ValueError:
            print("Error")
            theta3 = 0

        # Calculate theta2 (shoulder joint)
        theta2 = atan2(z, G) - atan2(l4 * sin(theta3), l3 + l4 * cos(theta3))

        return (theta1, theta2, theta3)


    def legPair(self, Tl, Tr, Ll, Lr, LEG_FR):
        # Identity matrix
        Ix = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # LEG_FR = LEG_FRONT -> 0 + 0 = 0
        self.thetas[LEG_FR + self.LEG_LEFT] = np.array(self.legIK(np.linalg.inv(Tl).dot(Ll)))

        # LEG_FR = LEG_BACK -> 2 + 1 = 3
        self.thetas[LEG_FR + self.LEG_RIGHT] = np.array(self.legIK(Ix.dot(np.linalg.inv(Tr).dot(Lr))))


    # drawRobot
    def robot(self, position, angle, center):
        # Rotation angle
        # omega: roll, phi: pitch, psi: yaw
        (omega, phi, psi) = angle
        # Center of body
        (xm, ym, zm) = center

        # FP = [0, 0, 0, 1]  # FP??

        # Final matrix ??
        # Tlf: LEFT-FRONT, Trf: RIGHT-FRONT, Tlb: LEFT-BACK, Trb: LEFT-BACK
        (Tlf, Trf, Tlb, Trb) = self.bodyIK(omega, phi, psi, xm, ym, zm)
        # CP??
        # CP = [x.dot(FP) for x in [Tlf, Trf, Tlb, Trb]]
        # CPs = [CP[x] for x in [0, 1, 3, 2, 0]]

        # FRONT
        self.legPair(Tlf, Trf, position[0], position[1], self.LEG_FRONT)
        # BACK
        self.legPair(Tlb, Trb, position[2], position[3], self.LEG_BACK)


def initIK(position):
    moduleKinematics = Kinematics()
    moduleKinematics.robot(position, (0, 0, 0), (0, 0, 0))

    return moduleKinematics.thetas


if __name__ == "__main__":
    # [LEFT-FRONT], [RIGNT-FRONT], [LEFT-BACK], [RIGHT-BACK]
    foot_position = np.array([[100, -100, 100, 1], [100, -100, -100, 1], [-100, -100, 100, 1], [-100, -100, -100, 1]])

    # Calculate inverse kinematics
    initIK(foot_position)

