#!/usr/bin/env python3
"""
Image Processing Script for Wheel of Heaven Data Images

This script processes images according to the configuration in manifest.yaml:
- Converts images to modern web formats (AVIF, WebP)
- Applies compression optimized for web publishing
- Adds subtle salt and pepper grain filter
- Creates thumbnails (optional)
- Backs up originals (optional)

Requirements:
    pip install pillow pyyaml numpy pillow-avif-plugin

Usage:
    python process_images.py [--config manifest.yaml] [--dry-run] [--formats avif,webp] [--force]
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import warnings

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

# Suppress PIL warnings about AVIF
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


class ImageProcessor:
    """Handles image processing operations with multi-format support"""

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

        # Check AVIF support
        self.avif_supported = self._check_avif_support()
        if not self.avif_supported:
            logger.warning("AVIF support not available. Install pillow-avif-plugin for AVIF support.")

    def _check_avif_support(self) -> bool:
        """Check if AVIF format is supported"""
        try:
            # Create a small test image and try to save as AVIF
            test_img = Image.new('RGB', (1, 1), color='red')
            test_path = self.output_dir / 'avif_test.avif'
            test_img.save(test_path, 'AVIF')
            test_path.unlink()  # Remove test file
            return True
        except Exception as e:
            logger.debug(f"AVIF not supported: {e}")
            return False

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

    def get_output_filename(self, original_name: str, custom_name: Optional[str] = None,
                          format_ext: str = 'webp') -> str:
        """Generate output filename with proper extension"""
        if custom_name:
            return f"{custom_name}.{format_ext}"
        else:
            # Remove extension and add new format extension
            name_without_ext = Path(original_name).stem
            return f"{name_without_ext}.{format_ext}"

    def get_format_settings(self, fmt: str, base_quality: int) -> Dict[str, Any]:
        """Get format-specific settings"""
        format_settings = self.processing_settings.get(fmt.lower(), {})

        if fmt.lower() == 'avif':
            return {
                'format': 'AVIF',
                'quality': format_settings.get('quality', max(30, base_quality - 30)),  # AVIF can use lower quality
                'speed': format_settings.get('speed', 6),
                'optimize': True
            }
        elif fmt.lower() == 'webp':
            return {
                'format': 'WebP',
                'quality': format_settings.get('quality', base_quality),
                'method': format_settings.get('method', 6),
                'optimize': self.processing_settings.get('optimize', True),
                'lossless': self.processing_settings.get('lossless', False)
            }
        else:
            # Fallback for other formats
            return {
                'format': fmt.upper(),
                'quality': base_quality,
                'optimize': True
            }

    def save_image_format(self, image: Image.Image, output_path: Path, fmt: str,
                         quality: int, preserve_metadata: bool = False,
                         original_image: Optional[Image.Image] = None) -> bool:
        """Save image in specified format with appropriate settings"""
        try:
            if fmt.lower() == 'avif' and not self.avif_supported:
                logger.warning(f"AVIF not supported, skipping {output_path}")
                return False

            save_kwargs = self.get_format_settings(fmt, quality)

            # Handle transparency for different formats
            if image.mode in ('RGBA', 'LA'):
                if fmt.lower() in ['avif', 'webp']:
                    # Both AVIF and WebP support transparency
                    processed_img = image.convert('RGBA')
                else:
                    # For formats that don't support transparency, convert to RGB
                    processed_img = Image.new('RGB', image.size, (255, 255, 255))
                    processed_img.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            elif image.mode not in ('RGB', 'L'):
                processed_img = image.convert('RGB')
            else:
                processed_img = image.copy()

            # Preserve metadata if configured and available
            if preserve_metadata and original_image and hasattr(original_image, 'info'):
                exif = original_image.info.get('exif')
                if exif and fmt.lower() in ['webp']:  # AVIF doesn't preserve EXIF well yet
                    save_kwargs['exif'] = exif

            processed_img.save(output_path, **save_kwargs)
            return True

        except Exception as e:
            logger.error(f"Failed to save {output_path} as {fmt}: {e}")
            return False

    def process_single_image(self, image_config: Dict[str, Any], dry_run: bool = False,
                           force_formats: Optional[List[str]] = None, force: bool = False) -> Dict[str, Any]:
        """
        Process a single image according to its configuration

        Args:
            image_config: Configuration for the specific image
            dry_run: If True, only log what would be done
            force_formats: Override formats from config
            force: If True, overwrite existing files

        Returns:
            Dictionary with processing results
        """
        filename = image_config['filename']
        source_path = self.raw_dir / filename

        if not source_path.exists():
            logger.error(f"Source file not found: {source_path}")
            return {'error': f"Source file not found: {source_path}"}

        # Get settings with fallbacks
        quality = image_config.get('quality', self.global_settings.get('quality', 80))
        grain_intensity = image_config.get('grain_intensity', self.global_settings.get('grain_intensity', 0.1))
        enabled = image_config.get('enabled', True)

        # Determine output formats
        if force_formats:
            formats = force_formats
        else:
            formats = image_config.get('formats',
                                     self.global_settings.get('formats', ['avif', 'webp']))

        if not enabled:
            logger.info(f"Skipping disabled image: {filename}")
            return {'skipped': True, 'reason': 'disabled'}

        logger.info(f"Processing: {filename}")
        logger.info(f"  Quality: {quality}, Grain: {grain_intensity}, Formats: {formats}")

        # Check if files already exist (unless force is True)
        if not force and not dry_run:
            existing_files = []
            missing_files = []

            for fmt in formats:
                output_filename = self.get_output_filename(filename,
                                                         image_config.get('output_name'),
                                                         fmt.lower())
                output_path = self.output_dir / output_filename

                if output_path.exists():
                    existing_files.append(fmt)
                else:
                    missing_files.append(fmt)

            if existing_files and not missing_files:
                logger.info(f"  Skipping {filename} - all formats already exist: {existing_files}")
                logger.info(f"  Use --force to overwrite existing files")
                return {'skipped': True, 'reason': 'already_exists', 'existing_formats': existing_files}
            elif existing_files:
                logger.info(f"  Some formats already exist: {existing_files}, processing missing: {missing_files}")
                formats = missing_files  # Only process missing formats

        if dry_run:
            skip_msg = " (some files exist)" if not force and 'existing_files' in locals() and existing_files else ""
            logger.info(f"  [DRY RUN] Would process {filename} to formats: {formats}{skip_msg}")
            return {'dry_run': True, 'formats': formats}

        results = {
            'source_file': str(source_path),
            'processed_formats': [],
            'thumbnails': [],
            'errors': []
        }

        try:
            # Backup original if configured
            if self.processing_settings.get('backup_originals', True):
                self.backup_original(source_path)

            # Open and process image
            with Image.open(source_path) as original_img:
                # Apply grain filter to a copy
                processed_img = original_img.copy()
                if grain_intensity > 0:
                    processed_img = self.add_grain_filter(processed_img, grain_intensity)

                # Save in each requested format
                for fmt in formats:
                    try:
                        output_filename = self.get_output_filename(filename,
                                                                 image_config.get('output_name'),
                                                                 fmt.lower())
                        output_path = self.output_dir / output_filename

                        success = self.save_image_format(
                            processed_img, output_path, fmt, quality,
                            preserve_metadata=self.global_settings.get('preserve_metadata', False),
                            original_image=original_img
                        )

                        if success:
                            file_size = output_path.stat().st_size
                            results['processed_formats'].append({
                                'format': fmt,
                                'filename': output_filename,
                                'size_bytes': file_size,
                                'size_mb': round(file_size / (1024 * 1024), 2)
                            })
                            logger.info(f"  Created {fmt.upper()}: {output_filename} ({file_size:,} bytes)")

                            # Create thumbnail if configured
                            if self.processing_settings.get('create_thumbnails', True):
                                thumbnail_width = self.processing_settings.get('thumbnail_width', 400)
                                thumbnail_suffix = self.processing_settings.get('thumbnail_suffix', '_thumb')

                                thumbnail = self.create_thumbnail(processed_img, thumbnail_width)
                                thumb_filename = f"{Path(output_filename).stem}{thumbnail_suffix}.{fmt.lower()}"
                                thumb_path = self.output_dir / thumb_filename

                                thumb_success = self.save_image_format(
                                    thumbnail, thumb_path, fmt, quality,
                                    preserve_metadata=False
                                )

                                if thumb_success:
                                    thumb_size = thumb_path.stat().st_size
                                    results['thumbnails'].append({
                                        'format': fmt,
                                        'filename': thumb_filename,
                                        'size_bytes': thumb_size,
                                        'size_mb': round(thumb_size / (1024 * 1024), 2)
                                    })
                                    logger.info(f"  Created thumbnail: {thumb_filename}")
                        else:
                            results['errors'].append(f"Failed to save {fmt} format")

                    except Exception as e:
                        error_msg = f"Error processing {fmt} format: {e}"
                        logger.error(f"  {error_msg}")
                        results['errors'].append(error_msg)

                # Calculate compression statistics
                original_size = source_path.stat().st_size
                results['original_size_bytes'] = original_size
                results['original_size_mb'] = round(original_size / (1024 * 1024), 2)

                if results['processed_formats']:
                    # Show compression stats for each format
                    for fmt_result in results['processed_formats']:
                        new_size = fmt_result['size_bytes']
                        compression_ratio = (1 - new_size / original_size) * 100
                        fmt_result['compression_ratio'] = round(compression_ratio, 1)

                        logger.info(f"  {fmt_result['format'].upper()} compression: "
                                  f"{compression_ratio:.1f}% reduction")

            return results

        except Exception as e:
            error_msg = f"Failed to process {filename}: {e}"
            logger.error(error_msg)
            return {'error': error_msg}

    def process_all_images(self, dry_run: bool = False,
                          force_formats: Optional[List[str]] = None, force: bool = False) -> Tuple[int, int, int, Dict[str, Any]]:
        """
        Process all images in the configuration

        Args:
            dry_run: If True, only log what would be done
            force_formats: Override formats from config
            force: If True, overwrite existing files

        Returns:
            Tuple of (successful_count, skipped_count, total_count, summary_stats)
        """
        images = self.config.get('images', [])
        successful = 0
        skipped = 0
        errors = 0
        total = len(images)
        all_results = []

        logger.info(f"Starting processing of {total} images...")
        if force_formats:
            logger.info(f"Forcing formats: {force_formats}")
        if force:
            logger.info(f"Force mode: will overwrite existing files")

        for i, image_config in enumerate(images, 1):
            logger.info(f"\n[{i}/{total}] Processing image...")

            result = self.process_single_image(image_config, dry_run, force_formats, force)
            all_results.append(result)

            if 'error' in result:
                errors += 1
                logger.error(f"Failed to process image {i}: {result['error']}")
            elif result.get('skipped', False):
                skipped += 1
                logger.debug(f"Skipped image {i}: {result.get('reason', 'unknown')}")
            else:
                successful += 1

        # Generate summary statistics
        summary = self._generate_summary(all_results, dry_run)

        logger.info(f"\nProcessing complete: {successful} processed, {skipped} skipped, {errors} errors ({total} total)")

        if not dry_run and successful > 0:
            self._print_summary(summary)

        return successful, skipped, total, summary

    def _generate_summary(self, results: List[Dict[str, Any]], dry_run: bool) -> Dict[str, Any]:
        """Generate summary statistics from processing results"""
        if dry_run:
            return {'dry_run': True}

        summary = {
            'total_processed': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'formats': {},
            'total_original_size': 0,
            'total_processed_size': 0,
            'thumbnails_created': 0
        }

        for result in results:
            if result.get('skipped'):
                summary['total_skipped'] += 1
            elif 'error' in result:
                summary['total_errors'] += 1
            else:
                summary['total_processed'] += 1
                summary['total_original_size'] += result.get('original_size_bytes', 0)

                for fmt_result in result.get('processed_formats', []):
                    fmt = fmt_result['format']
                    if fmt not in summary['formats']:
                        summary['formats'][fmt] = {
                            'count': 0,
                            'total_size': 0,
                            'total_compression': 0
                        }

                    summary['formats'][fmt]['count'] += 1
                    summary['formats'][fmt]['total_size'] += fmt_result['size_bytes']
                    summary['formats'][fmt]['total_compression'] += fmt_result.get('compression_ratio', 0)
                    summary['total_processed_size'] += fmt_result['size_bytes']

                summary['thumbnails_created'] += len(result.get('thumbnails', []))

        # Calculate averages
        for fmt_data in summary['formats'].values():
            if fmt_data['count'] > 0:
                fmt_data['average_compression'] = fmt_data['total_compression'] / fmt_data['count']

        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """Print processing summary"""
        logger.info(f"\nSUMMARY:")
        logger.info(f"  Images processed: {summary['total_processed']}")
        logger.info(f"  Images skipped: {summary['total_skipped']}")
        logger.info(f"  Errors: {summary['total_errors']}")
        logger.info(f"  Thumbnails created: {summary['thumbnails_created']}")

        original_mb = summary['total_original_size'] / (1024 * 1024)
        processed_mb = summary['total_processed_size'] / (1024 * 1024)
        saved_mb = original_mb - processed_mb

        logger.info(f"  Original total size: {original_mb:.2f} MB")
        logger.info(f"  Processed total size: {processed_mb:.2f} MB")
        logger.info(f"  Space saved: {saved_mb:.2f} MB")

        logger.info(f"\nFORMAT BREAKDOWN:")
        for fmt, data in summary['formats'].items():
            fmt_mb = data['total_size'] / (1024 * 1024)
            logger.info(f"  {fmt.upper()}: {data['count']} files, "
                       f"{fmt_mb:.2f} MB, "
                       f"{data['average_compression']:.1f}% avg compression")

        logger.info(f"\nOUTPUT DIRECTORY: {self.output_dir}")
        if self.processing_settings.get('backup_originals', True):
            logger.info(f"BACKUP DIRECTORY: {self.backup_dir}")


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
    parser.add_argument(
        '--formats',
        help='Comma-separated list of formats to generate (e.g., avif,webp)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files (default: skip already processed files)'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse formats if provided
    force_formats = None
    if args.formats:
        force_formats = [fmt.strip().lower() for fmt in args.formats.split(',')]
        logger.info(f"Override formats: {force_formats}")

    # Load configuration
    config = load_config(args.config)

    # Initialize processor
    processor = ImageProcessor(config)

    # Process images
    successful, skipped, total, summary = processor.process_all_images(
        dry_run=args.dry_run,
        force_formats=force_formats,
        force=args.force
    )

    # Exit with appropriate code
    if args.dry_run:
        logger.info("Dry run completed!")
        sys.exit(0)
    elif successful + skipped == total:
        logger.info("All images processed successfully!")
        sys.exit(0)
    else:
        failed = total - successful - skipped
        logger.warning(f"Some images failed to process ({failed} failures)")
        sys.exit(1)


if __name__ == '__main__':
    main()
