#!/usr/bin/env python3
"""
Image Processing Script for Wheel of Heaven Data Images

This script processes images according to the configuration in images_to_process.yaml:
- Converts images to WebP format
- Applies compression for web publishing
- Adds subtle salt and pepper grain filter
- Creates thumbnails (optional)
- Backs up originals (optional)

Requirements:
    pip install pillow pyyaml numpy

Usage:
    python process_images.py [--config images_to_process.yaml] [--dry-run]
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_settings = config.get('global_settings', {})
        self.processing_settings = config.get('processing', {})

        # Setup directories
        self.raw_dir = Path('raw')
        self.output_dir = Path(self.global_settings.get('output_directory', 'processed'))
        self.backup_dir = Path(self.global_settings.get('backup_directory', 'backup'))

        # Create output directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        if self.processing_settings.get('backup_originals', True):
            self.backup_dir.mkdir(exist_ok=True)

    def add_grain_filter(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """
        Add subtle salt and pepper grain filter to an image

        Args:
            image: PIL Image object
            intensity: Grain intensity (0.0 to 1.0)

        Returns:
            PIL Image with grain filter applied
        """
        if intensity <= 0:
            return image

        # Convert to numpy array
        img_array = np.array(image)

        # Generate noise
        noise = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)

        # Create mask for salt and pepper
        salt_pepper_mask = np.random.random(img_array.shape[:2]) < intensity

        # Apply salt and pepper effect
        if len(img_array.shape) == 3:  # RGB/RGBA
            # Expand mask to all channels
            salt_pepper_mask = np.stack([salt_pepper_mask] * img_array.shape[2], axis=2)

        # Mix original with noise based on intensity
        grain_strength = intensity * 0.3  # Reduce the effect for subtlety
        noisy_array = img_array.astype(np.float32)
        noise_contribution = (noise.astype(np.float32) - 128) * grain_strength

        # Apply the noise selectively
        noisy_array = noisy_array + noise_contribution * salt_pepper_mask.astype(np.float32)

        # Clip values to valid range
        noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(noisy_array)

    def create_thumbnail(self, image: Image.Image, width: int = 400) -> Image.Image:
        """Create a thumbnail maintaining aspect ratio"""
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def backup_original(self, source_path: Path) -> bool:
        """Backup the original file"""
        try:
            backup_path = self.backup_dir / source_path.name
            if not backup_path.exists():
                import shutil
                shutil.copy2(source_path, backup_path)
                logger.info(f"Backed up: {source_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup {source_path.name}: {e}")
            return False

    def get_output_filename(self, original_name: str, custom_name: Optional[str] = None) -> str:
        """Generate output filename"""
        if custom_name:
            return f"{custom_name}.webp"
        else:
            # Remove extension and add .webp
            name_without_ext = Path(original_name).stem
            return f"{name_without_ext}.webp"

    def process_single_image(self, image_config: Dict[str, Any], dry_run: bool = False) -> bool:
        """
        Process a single image according to its configuration

        Args:
            image_config: Configuration for the specific image
            dry_run: If True, only log what would be done

        Returns:
            True if successful, False otherwise
        """
        filename = image_config['filename']
        source_path = self.raw_dir / filename

        if not source_path.exists():
            logger.error(f"Source file not found: {source_path}")
            return False

        # Get settings with fallbacks
        quality = image_config.get('quality', self.global_settings.get('quality', 80))
        grain_intensity = image_config.get('grain_intensity', self.global_settings.get('grain_intensity', 0.1))
        enabled = image_config.get('enabled', True)

        if not enabled:
            logger.info(f"Skipping disabled image: {filename}")
            return True

        output_filename = self.get_output_filename(filename, image_config.get('output_name'))
        output_path = self.output_dir / output_filename

        logger.info(f"Processing: {filename} -> {output_filename}")
        logger.info(f"  Quality: {quality}, Grain: {grain_intensity}")

        if dry_run:
            logger.info(f"  [DRY RUN] Would process {filename}")
            return True

        try:
            # Backup original if configured
            if self.processing_settings.get('backup_originals', True):
                self.backup_original(source_path)

            # Open and process image
            with Image.open(source_path) as img:
                # Convert to RGB if necessary (WebP doesn't support all modes)
                if img.mode in ('RGBA', 'LA'):
                    # Preserve transparency
                    processed_img = img.convert('RGBA')
                elif img.mode not in ('RGB', 'L'):
                    processed_img = img.convert('RGB')
                else:
                    processed_img = img.copy()

                # Apply grain filter
                if grain_intensity > 0:
                    processed_img = self.add_grain_filter(processed_img, grain_intensity)

                # Save as WebP
                save_kwargs = {
                    'format': 'WebP',
                    'quality': quality,
                    'optimize': self.processing_settings.get('optimize', True),
                    'lossless': self.processing_settings.get('lossless', False)
                }

                # Preserve metadata if configured
                if self.global_settings.get('preserve_metadata', False):
                    save_kwargs['exif'] = img.info.get('exif', b'')

                processed_img.save(output_path, **save_kwargs)

                # Create thumbnail if configured
                if self.processing_settings.get('create_thumbnails', True):
                    thumbnail_width = self.processing_settings.get('thumbnail_width', 400)
                    thumbnail_suffix = self.processing_settings.get('thumbnail_suffix', '_thumb')

                    thumbnail = self.create_thumbnail(processed_img, thumbnail_width)
                    thumb_filename = f"{Path(output_filename).stem}{thumbnail_suffix}.webp"
                    thumb_path = self.output_dir / thumb_filename

                    thumbnail.save(thumb_path, **save_kwargs)
                    logger.info(f"  Created thumbnail: {thumb_filename}")

                # Get file size info
                original_size = source_path.stat().st_size
                new_size = output_path.stat().st_size
                compression_ratio = (1 - new_size / original_size) * 100

                logger.info(f"  Original: {original_size:,} bytes")
                logger.info(f"  Compressed: {new_size:,} bytes ({compression_ratio:.1f}% reduction)")

            return True

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            return False

    def process_all_images(self, dry_run: bool = False) -> Tuple[int, int]:
        """
        Process all images in the configuration

        Args:
            dry_run: If True, only log what would be done

        Returns:
            Tuple of (successful_count, total_count)
        """
        images = self.config.get('images', [])
        successful = 0
        total = len(images)

        logger.info(f"Starting processing of {total} images...")

        for i, image_config in enumerate(images, 1):
            logger.info(f"\n[{i}/{total}] Processing image...")

            if self.process_single_image(image_config, dry_run):
                successful += 1
            else:
                logger.error(f"Failed to process image {i}")

        logger.info(f"\nProcessing complete: {successful}/{total} images processed successfully")

        if not dry_run:
            # Print summary
            logger.info(f"\nSummary:")
            logger.info(f"  Input directory: {self.raw_dir}")
            logger.info(f"  Output directory: {self.output_dir}")
            if self.processing_settings.get('backup_originals', True):
                logger.info(f"  Backup directory: {self.backup_dir}")
            logger.info(f"  Success rate: {successful/total*100:.1f}%")

        return successful, total


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Process images according to YAML configuration')
    parser.add_argument(
        '--config',
        default='manifest.yaml',
        help='Path to configuration file (default: manifest.yaml)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing images'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Initialize processor
    processor = ImageProcessor(config)

    # Process images
    successful, total = processor.process_all_images(dry_run=args.dry_run)

    # Exit with appropriate code
    if successful == total:
        logger.info("All images processed successfully!")
        sys.exit(0)
    else:
        logger.warning(f"Some images failed to process ({total - successful} failures)")
        sys.exit(1)


if __name__ == '__main__':
    main()
