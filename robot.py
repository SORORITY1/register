#!/usr/local/bin/python3.10

__author__ = "Ludovic Mustière", "Apolline de Vaulchier"
__credits__ = ["Ludovic Mustière", "Apolline de Vaulchier"]
__version__ = "1.0.0"
__maintainer__ = "Ludovic Mustière"
__email__ = "ludovic.mustiere@ensta-bretagne.org"
__status__ = "Release"
__doc__ = "This file can be used to create a robot to explore a karst."

import matplotlib.pyplot as plt
import numpy as np
from math import ceil


def tran2H(x : float, y : float) -> np.ndarray:
    """2D homogenous transformation matrix for a translation of (x, y).
    Source: https://www.ensta-bretagne.fr/jaulin/roblib.py

    Args:
        x (float): x coordinate
        y (float): y coordinate

    Returns:
        np.ndarray: Homogenous transformation matrix
    
    Examples:
        >>> tran2H(1, 2)
        array([[1, 0, 1],
               [0, 1, 2],
               [0, 0, 1]])
    """
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

def rot2H(a : float) -> np.ndarray:
    """2D homogenous transformation matrix for a rotation of a radians.
    Source: https://www.ensta-bretagne.fr/jaulin/roblib.py

    Args:
        a (float): Angle of rotation in radian

    Returns:
        np.ndarray: Homogenous transformation matrix
    
    Examples:
        >>> rot2H(pi/2)
        array([[ 6.123234e-17, -1.000000e+00,  0.000000e+00],
               [ 1.000000e+00,  6.123234e-17,  0.000000e+00],
               [ 0.000000e+00,  0.000000e+00,  1.000000e+00]])
    """
    a = float(a)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

def plot2D(M : np.ndarray, col='black', w=1):
    """Draw on a 2D figure a segment with starting point first column and second point second column.
    Source: https://www.ensta-bretagne.fr/jaulin/roblib.py

    Args:
        M (np.ndarray): Matrix containing the coordinates of the points. First line are x coordinates and second line are y coordinates.
        col (str, optional): Color of the plot. Defaults to 'black'.
        w (int, optional): Width of the plot. Defaults to 1.
    
    Examples:
        >>> M = array([[1, 2, 7], [4, 5, 1]])
        >>> plot2D(M, 'red', 2)
    """
    plt.plot(M[0, :], M[1, :], col, linewidth=w)

def add1(M : np.ndarray) -> np.ndarray:
    """Adds to the array M a line of 1 at the bottom and returns it.
    Source: https://www.ensta-bretagne.fr/jaulin/roblib.py

    Args:
        M (np.ndarray): Numpy Array to be transformed

    Returns:
        np.ndarray: M array with one more line of 1 at the bottom
    
    Examples:
        >>> add1(array([[1, 2, 3], [4, 5, 6]]))
        array([[1, 2, 3],
               [4, 5, 6],
               [1, 1, 1]])
    """
    M = np.array(M)
    return np.vstack((M, np.ones(M.shape[1])))

def draw_segment(a : np.ndarray, b : np.ndarray, col='darkblue', w=1):
    """Draw a segment between point a and point b. Both points are represented by vertical Numpy array.
    Source: https://www.ensta-bretagne.fr/jaulin/roblib.py

    Args:
        a (np.ndarray): Vector containing the coordinates of the first point.
        b (np.ndarray): Vector containing the coordinates of the second point.
        col (str, optional): Color of the plot. Defaults to 'darkblue'.
        w (int, optional): Width of the plot. Defaults to 1.
    
    Examples:
        >>> a = array([[1], [2]])
        >>> b = array([[7], [4]])
        >>> draw_segment(a, b, 'red', 2)
    """
    plot2D(np.hstack((a, b)), col, w)


class Robot:
    """This class can be used to create a robot to explore a karst.
    
    Args:
        mx (float): 
            x coordinate of the robot.
        my (float): 
            y coordinate of the robot.
        theta (float): 
            angle of the robot.
        color (str): 
            color of the robot.
    
    Attributes:
        mx (float):
            x coordinate of the robot.
        my (float):
            y coordinate of the robot.
        theta (float):
            angle of the robot.
        color (str):
            color of the robot.
        angle_sonar (float):
            angle of the sonar.
        k (int):
            maximum distance of the sonar.
        dist_right (int):
            distance of the right sonar.
        dist_left (int):
            distance of the left sonar.
        dist_front (int):
            distance of the front sonar.
        u (np.ndarray):
            control vector.
            
    Methods:
        draw_robot(r, w):
            Draw a robot on ax Figure with state vector x.
        intersection(tubes_):
            Compute the intersection between the robot sonars and the tubes.
        compute_intersection(x1, x2, x3, x4):
            Compute the intersection between two line segments.
        distance(xi):
            Compute the distance between the robot and a point.
        reset_sonar():
            Reset the sonar to the maximum distance possible.
        control():
            Compute the control vector.
        move(dt):
            Compute the state vector at the next step.
    
    Examples:
        >>> r1 = Robot(0, 0, 0, 'red')
        >>> r1.draw_robot()
    """
    def __init__(self, mx: float, my: float, theta: float, color='b') -> None:
        self.mx = mx
        self.my = my
        self.theta = theta
        self.color = color
        self.angle_sonar = np.pi/2
        self.k = 250
        self.dist_right = self.k
        self.dist_left = self.k
        self.dist_front = self.k
        self.u = np.array([[0], [0]])

    def draw_robot(self, r=5, w=2) -> None:
        """Draw a robot on ax Figure with state vector x.

        Args:
            r (int, optional): Size of the robot. Defaults to 80.
            w (int, optional): Width of the lines. Defaults to 2.
        
        Examples:
            >>> draw_robot(70, 3)
        """
        M = r * np.array(
            [[0, -0.5, 0.5, 0], [3, -2, -2, 3]])
        M = add1(M)
        plot2D(tran2H(self.mx, self.my) @ rot2H(self.theta-np.pi/2) @ M, self.color, w)
        # SONAR 1 : Right
        draw_segment(np.array([self.mx, self.my]).reshape(2, 1), np.array([self.mx + self.dist_right * np.cos(self.theta - self.angle_sonar), self.my + self.dist_right * np.sin(self.theta - self.angle_sonar)]).reshape(2, 1), col="pink", w=2)
        # SONAR 2 : Left
        draw_segment(np.array([self.mx, self.my]).reshape(2, 1), np.array([self.mx + self.dist_left * np.cos(self.theta + self.angle_sonar), self.my + self.dist_left * np.sin(self.theta + self.angle_sonar)]).reshape(2, 1), col="pink", w=2)
        # SONAR 3 : Front
        draw_segment(np.array([self.mx, self.my]).reshape(2, 1), np.array([self.mx + self.dist_front * np.cos(self.theta), self.my + self.dist_front * np.sin(self.theta)]).reshape(2, 1), col="pink", w=2)

    def intersection(self, tubes_) -> None:
        """Compute the intersection between the robot sonars and the tubes.

        Args:
            tubes_ (list[Tube]): List of tubes.
        """
        r1, r2 = np.array([self.mx, self.my]).reshape(2, 1), np.array([self.mx + self.k * np.cos(self.theta - self.angle_sonar), self.my + self.k * np.sin(self.theta - self.angle_sonar)]).reshape(2, 1)
        l1, l2 = np.array([self.mx, self.my]).reshape(2, 1), np.array([self.mx + self.k * np.cos(self.theta + self.angle_sonar), self.my + self.k * np.sin(self.theta + self.angle_sonar)]).reshape(2, 1)
        f1, f2 = np.array([self.mx, self.my]).reshape(2, 1), np.array([self.mx + self.k * np.cos(self.theta), self.my + self.k * np.sin(self.theta)]).reshape(2, 1)
        for tube_ in tubes_:
            for i in range(max(ceil(self.mx-self.k)+tube_.X, 0), min(ceil(self.mx+self.k)+tube_.X, tube_.xmax-tube_.xmin)-1):
                if tube_.usedi1[i]:
                    a, b = tube_.yi1_cubic[i], tube_.yi1_cubic[i+1]
                    # SONAR 1 : Right
                    if (self.compute_intersection(r1, r2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))) is not None:
                        xi = self.compute_intersection(r1, r2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))
                        dist = self.distance(xi)
                        if dist < self.dist_right:
                            self.dist_right = dist
                    # SONAR 2 : Left
                    if (self.compute_intersection(l1, l2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))) is not None:
                        xi = self.compute_intersection(l1, l2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))
                        dist = self.distance(xi)
                        if dist < self.dist_left:
                            self.dist_left = dist
                    # SONAR 3 : Front
                    if (self.compute_intersection(f1, f2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))) is not None:
                        xi = self.compute_intersection(f1, f2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))
                        dist = self.distance(xi)
                        if dist < self.dist_front:
                            self.dist_front = dist
                if tube_.usedi2[i]:
                    a, b = tube_.yi2_cubic[i], tube_.yi2_cubic[i+1]
                    # SONAR 1 : Right
                    if (self.compute_intersection(r1, r2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))) is not None:
                        xi = self.compute_intersection(r1, r2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))
                        dist = self.distance(xi)
                        if dist < self.dist_right:
                            self.dist_right = dist
                    # SONAR 2 : Left
                    if (self.compute_intersection(l1, l2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))) is not None:
                        xi = self.compute_intersection(l1, l2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))
                        dist = self.distance(xi)
                        if dist < self.dist_left:
                            self.dist_left = dist
                    # SONAR 3 : Front
                    if (self.compute_intersection(f1, f2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))) is not None:
                        xi = self.compute_intersection(f1, f2, np.array([tube_.x[i], a]).reshape(2, 1), np.array([tube_.x[i+1], b]).reshape(2, 1))
                        dist = self.distance(xi)
                        if dist < self.dist_front:
                            self.dist_front = dist
            # Compute Intersection for tube extremities --> TODO: Add tube extremities to a list to check if they can be used

    def compute_intersection(self, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray) -> np.ndarray:
        """Compute the intersection between two line segments.
        Source: https://en.wikipedia.org/wiki/Intersection_(geometry)#Two_line_segments
        
        Args:
            x1 (np.ndarray): First point of the first line segment.
            x2 (np.ndarray): Second point of the first line segment.
            x3 (np.ndarray): First point of the second line segment.
            x4 (np.ndarray): Second point of the second line segment.
        
        Returns:
            np.ndarray: Coordinates of the intersection point.
        """
        a1 = x2[0] - x1[0]
        b1 = x4[0] - x3[0]
        c1 = x3[0] - x1[0]
        a2 = x2[1] - x1[1]
        b2 = x4[1] - x3[1]
        c2 = x3[1] - x1[1]

        if a1*b2 - a2*b1 == 0: # Parallel lines
            return None
        s0 = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
        t0 = (a1*c2 - a2*c1) / (a1*b2 - a2*b1)

        if s0 <= 1 and s0 >= 0 and t0 <= 1 and t0 >= 0:
            return np.array([x1[0] + s0*(x2[0] - x1[0]), x1[1] + s0*(x2[1] - x1[1])])
        else:
            return None

    def distance(self, xi: np.ndarray) -> float:
        """Compute the distance between the robot and a point.

        Args:
            xi (np.ndarray): Coordinates of the point.

        Returns:
            float: Distance between the robot and the point.
        """
        return np.sqrt((self.mx-xi[0])**2+(self.my-xi[1])**2)

    def reset_sonar(self) -> None:
        """Reset the sonar to the maximum distance possible.
        """
        self.dist_front = self.k
        self.dist_left = self.k
        self.dist_right = self.k

    def control(self):
        """Compute the control vector.
        """
        min_dist_front = 35
        min_dist_border = 4
        e = 0
        if self.dist_front < min_dist_front:
            if self.dist_left <= self.dist_right+e :
                u1 = -5
                u2 = -0.5
            else:
                u1 = -5
                u2 = 0.5
        else:
            if self.dist_left < min_dist_border:
                print("Turn right: 2")
                u1 = 15
                u2 = -0.1
            elif self.dist_right < min_dist_border:
                print("Turn left: 2")
                u1 = 15
                u2 = 0.1
            elif self.dist_left < self.dist_right:
                print("Turn left: 1")
                u1 = 25
                u2 = -0.1
            elif self.dist_right < self.dist_left:
                print("Turn right: 1")
                u1 = 25
                u2 = 0.1
            else:
                print("Flat out")
                u1 = 25
                u2 = 0
        self.u = np.array([[u1], [u2]])

    def move(self, dt):
        """Compute the state vector at the next step.
        """
        self.mx = self.mx + self.u[0, 0] * np.cos(self.theta) * dt
        self.my = self.my + self.u[0, 0] * np.sin(self.theta) * dt
        self.theta = self.theta + self.u[1, 0] * dt
