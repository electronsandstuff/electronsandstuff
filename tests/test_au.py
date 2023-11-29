import accelerator_utils.file_interfaces as fi
import numpy as np
import tempfile, os


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
        X, Y = fi.pixel_coords_from_bbox(upper_left, lower_left, lower_right, (32, 32))
        
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


def generate_test_subrow():
    row = {k: np.random.random() for k in "nicr"}
    row['image'] = np.random.randint(0, 2**16-1, size=(64, 64)).astype(np.uint16)
    return row


def generate_test_row():
    row = {k: np.random.random() for k in "jwop"}
    row['children'] = [generate_test_subrow() for _ in range(np.random.randint(4, 8))]
    row['image'] = np.random.randint(0, 2**16-1, size=(64, 64)).astype(np.uint16)
    return row


def test_save_load_homogenous_data():
    # Create some fake data
    np.random.seed(0)
    test_data = [generate_test_row() for _ in range(np.random.randint(32, 128))]

    with tempfile.TemporaryDirectory() as dir:
        # Save it to disk
        path = os.path.join(dir, 'test')
        fi.save_homogenous_data(test_data, path)
        
        # Load it back
        data, images = fi.load_data_and_images(path)
        
        # Run some basic checks
        assert data.keys() == {'data', 'children'}  # All tables are here
        assert all(x in images for x in data['data']['image'])  # All images are loaded back
        assert all(x in images for x in data['children']['image'])  # All images are loaded back

        for test, real in zip(data['data'].to_dict('records'), test_data):
            # Check this row
            assert all(np.isclose(test[k], real[k]) for k in real.keys() if len(k) == 1)  # Check the basic keys
            np.testing.assert_allclose(real['image'], images[test['image']])  # Test the image
            
            # Deal with the children
            child_df = data['children'][data['children']['data_idx'] == test['Unnamed: 0']]
            assert len(real['children']) == len(child_df)  # Child count
            for test_child, real_child in zip(child_df.to_dict('records'), real['children']):
                assert all(np.isclose(test_child[k], real_child[k]) for k in real_child.keys() if len(k) == 1)  # Check the basic keys
                np.testing.assert_allclose(real_child['image'], images[test_child['image']])  # Test the image
