import os
import tempfile
import pytest
import numpy as np
from PIL import Image

from electronsandstuff.surrogate_files.images import ImageManager


def create_grayscale_image(size=(10, 10)):
    """Create a random grayscale scientific test image."""
    # Create a random grayscale image (simulating noise in scientific images)
    data = np.random.randint(0, 256, size, dtype=np.uint8)
    return Image.fromarray(data, mode="L")


def create_color_image(size=(10, 10)):
    """Create a random color test image."""
    # Create a random color image
    data = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


@pytest.mark.parametrize("image_type", ["grayscale", "color"])
def test_image_manager(image_type):
    """Test the ImageManager class functionality with different image types."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize the ImageManager with the temporary directory
        manager = ImageManager(temp_dir, depth=2)

        # Create and save multiple images (exceeding max_dir_size=100)
        num_images = 500
        original_images = []
        filenames = []

        # Create and save images
        for i in range(num_images):
            if image_type == "grayscale":
                img = create_grayscale_image()
            else:  # color
                img = create_color_image()

            original_images.append(img)

            # Save the image and get the filename
            filename = manager.add_image(img)
            filenames.append(filename)

            if i == 0:
                assert filename == "00/00.png"

            # Verify the file exists
            assert os.path.exists(os.path.join(temp_dir, filename))

        # Verify the length of the manager
        assert len(manager) == num_images

        # Verify the directory structure
        # With depth=2, we should have nested directories
        dirs = os.listdir(temp_dir)
        assert len(dirs) > 1  # Should have multiple directories

        # Load back all images and verify they match the originals
        for i, filename in enumerate(filenames):
            loaded_img = manager.get_image(filename)

            # Convert both images to arrays for comparison
            original_array = np.array(original_images[i])
            loaded_array = np.array(loaded_img)

            # Verify the images match
            np.testing.assert_array_equal(original_array, loaded_array)


def test_get_filename():
    """Test the _get_filename method of ImageManager."""
    # Initialize the ImageManager with a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ImageManager(temp_dir, depth=3)

        # Test with different indices
        filename0 = manager._get_filename(0)
        assert filename0 == "00/00/00.png"

        filename123 = manager._get_filename(123)
        assert filename123 == "00/01/23.png"

        # Test with a large index that requires all depth levels
        filename12345 = manager._get_filename(12345)
        assert filename12345 == "01/23/45.png"

        # Test with custom depth and extension
        manager_custom = ImageManager(temp_dir, depth=2, extension=".jpg")
        filename_custom = manager_custom._get_filename(42, depth=2, extension=".jpg")
        assert filename_custom == "00/42.jpg"
