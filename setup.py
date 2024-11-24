# setup.py
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_project():
    """
    Setup the project directory structure and verify required files
    """
    # Required directories
    directories = [
        './data_sample',
        './data_sample/data_sample',  # Nested as per original structure
        './models'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Verify labels.csv exists
    if not os.path.exists('labels.csv'):
        logger.error("labels.csv not found in root directory!")
        logger.info("Please place your labels.csv file in the project root directory")
    else:
        logger.info("labels.csv found")
    
    # Check for images
    image_dir = './data_sample/data_sample'
    if os.path.exists(image_dir):
        images = [f for f in os.listdir(image_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"Found {len(images)} images in {image_dir}")
        if len(images) == 0:
            logger.error(f"No images found in {image_dir}")
            logger.info("Please place your image files in the data_sample/data_sample directory")
    
if __name__ == "__main__":
    setup_project()