import accelerator_utils as au
import numpy as np


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def test_pixel_coords():
    np.random.seed(0)
    for _ in range(8):
        # Generate a random bounding box
        upper_left = np.array((-np.random.random(), np.random.random()))
        lower_left = np.array((-np.random.random(), -np.random.random()))
        lower_right = np.array((np.random.random(), -np.random.random()))
        upper_right = lower_right + (upper_left - lower_left)
        edges = (
            (lower_left, lower_right), (lower_right, upper_right), 
            (upper_right, upper_left), (upper_left, lower_left)
        )
        
        # Generate the grid of points
        X, Y = au.pixel_coords_from_bbox(upper_left, lower_left, lower_right, (32, 32))
        
        # Check the boundaries
        np.testing.assert_allclose((X[0, 0], Y[0, 0]), lower_left)
        np.testing.assert_allclose((X[0, -1], Y[0, -1]), lower_right)
        np.testing.assert_allclose((X[-1, -1], Y[-1, -1]), upper_right)
        np.testing.assert_allclose((X[-1, 0], Y[-1, 0]), upper_left)

        # Shrink by a small amount to avoid (literal) edge cases
        scale = 1-1e-8
        X, Y = scale*X + (1-scale)*np.mean(X), scale*Y + (1-scale)*np.mean(Y)
        
        # Check all points lie inside the parallelogram
        for x, y in zip(X.ravel(), Y.ravel()):
            intersections = sum(intersect(*edge, np.array((x, y)), np.array((2, 2))) for edge in edges)
            assert intersections%2 == 1
