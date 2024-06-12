from dataclasses import dataclass
import numpy as np
from numpy import random
import math
from random import randrange

# Diagonal matrix for Lorentzian metric
M = np.diag([1, 1, 1, -1])

@dataclass
class Vertexer:
    """Class for handling vertex calculations."""
    
    nodes: np.ndarray
    c: float

    def is_cayley_menger_nonzero(self, vertices):
        """Check if the Cayley-Menger determinant is non-zero.
        
        Args:
            vertices (np.ndarray): Array of vertices.
        
        Returns:
            bool: True if the determinant is non-zero, False otherwise.
        """
        n = len(vertices)
        d = np.zeros((n + 2, n + 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                d[i + 2, j + 2] = np.linalg.norm(vertices[i] - vertices[j])**2

        for i in range(2, n + 2):
            for j in range(2, n + 2):
                d[i, j] = d[i, 1] + d[1, j] - 2 * np.dot(vertices[i - 2] - vertices[0], vertices[j - 2] - vertices[0])
        
        return np.linalg.det(d) != 0

    def reconstruct(self, times):
        """Reconstruct the vertex positions given the times.
        
        Args:
            times (np.ndarray): Array of times.
        
        Returns:
            np.ndarray or None: The reconstructed vertex with the least sum of squared residuals, or None if the Cayley-Menger determinant is zero.
        """
        # Normalize times
        times -= times.min()

        if not self.is_cayley_menger_nonzero(self.nodes):
            print("Cayley-Menger determinant is zero.")
            return None

        # Append time column to nodes array, scaled by c
        A = np.append(self.nodes, np.reshape(times, (-1, 1)) * self.c, axis=1)

        def ssr_error(point):
            """Calculate the sum of squared residuals (SSR) error.
            
            Args:
                point (np.ndarray): A point to calculate the SSR error against.
            
            Returns:
                float: The SSR error.
            """
            return np.sum(((np.linalg.norm(self.nodes - point, axis=1) / self.c) - times)**2)

        def lorentz(a, b):
            """Calculate the Lorentzian inner product.
            
            Args:
                a (np.ndarray): First vector.
                b (np.ndarray): Second vector.
            
            Returns:
                np.ndarray: Lorentzian inner product of a and b.
            """
            return np.sum(a * (b @ M), axis=-1)

        # Calculate Lorentzian inner product for each pair of points
        b = lorentz(A, A) * 0.5
        C = np.linalg.solve(A, np.ones(4))
        D = np.linalg.solve(A, b)

        # Find roots of the quadratic equation
        roots = np.roots([lorentz(C, C), (lorentz(C, D) - 1) * 2, lorentz(D, D)])

        solutions = []
        for root in roots:
            # Solve for each root and transform using Lorentzian metric
            X, Y, Z, T = M @ np.linalg.solve(A, root + b)
            solutions.append(np.array([X, Y, Z]))

        # Return the solution with the least SSR error
        return min(solutions, key=ssr_error)
