#!/usr/bin/env python3
"""
Processed Images Analysis Script

This script analyzes all processed WebP images and provides detailed information including:
- File sizes and compression ratios
- Image dimensions and metadata
- Quality analysis
- Processing statistics

Usage:
    python analyze_processed.py [--output-format table|json|csv] [--sort-by size|compression|name]
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime

class ProcessedImageAnalyzer:
    """Analyzes processed WebP images and provides detailed statistics"""

    def __init__(self, processed_dir: str = "processed", raw_dir: str = "raw", manifest_file: str = "manifest.yaml"):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.manifest_file = Path(manifest_file)
        self.manifest_data = self.load_manifest()

    def load_manifest(self) -> Dict[str, Any]:
        """Load the manifest configuration"""
        try:
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Manifest file {self.manifest_file} not found")
            return {}
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing manifest file: {e}")
            return {}

    def get_original_filename(self, webp_filename: str) -> Optional[str]:
        """Find the original filename from manifest for a given WebP file"""
        webp_stem = Path(webp_filename).stem

        # Handle thumbnail files
        if webp_stem.endswith('_thumb'):
            webp_stem = webp_stem[:-6]  # Remove '_thumb' suffix

        images = self.manifest_data.get('images', [])
        for image_config in images:
            original_name = image_config['filename']
            output_name = image_config.get('output_name')

            # Check if this matches our processed file
            if output_name and output_name == webp_stem:
                return original_name
            elif Path(original_name).stem == webp_stem:
                return original_name

        return None

    def get_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract metadata from an image file"""
        metadata = {}
        try:
            with Image.open(image_path) as img:
                metadata.update({
                    'format': img.format,
                    'mode': img.mode,
                    'width': img.width,
                    'height': img.height,
                    'megapixels': round((img.width * img.height) / 1_000_000, 2),
                    'aspect_ratio': round(img.width / img.height, 2) if img.height > 0 else 0,
                })

                # Get EXIF data if available
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        metadata[f'exif_{tag}'] = str(value)

                # Get additional info from PIL
                if hasattr(img, 'info'):
                    for key, value in img.info.items():
                        if key not in ['exif']:  # Skip exif as we handle it above
                            metadata[f'info_{key}'] = str(value)

        except Exception as e:
            metadata['error'] = str(e)

        return metadata

    def analyze_single_image(self, processed_path: Path) -> Dict[str, Any]:
        """Analyze a single processed image"""
        result = {
            'processed_filename': processed_path.name,
            'processed_path': str(processed_path),
            'processed_size_bytes': processed_path.stat().st_size,
            'processed_size_mb': round(processed_path.stat().st_size / (1024 * 1024), 2),
            'is_thumbnail': '_thumb' in processed_path.stem,
        }

        # Get processed image metadata
        processed_metadata = self.get_image_metadata(processed_path)
        result.update({f'processed_{k}': v for k, v in processed_metadata.items()})

        # Find original file
        original_filename = self.get_original_filename(processed_path.name)
        if original_filename:
            original_path = self.raw_dir / original_filename
            result['original_filename'] = original_filename

            if original_path.exists():
                result['original_path'] = str(original_path)
                result['original_size_bytes'] = original_path.stat().st_size
                result['original_size_mb'] = round(original_path.stat().st_size / (1024 * 1024), 2)

                # Calculate compression ratio
                compression_ratio = (1 - result['processed_size_bytes'] / result['original_size_bytes']) * 100
                result['compression_ratio_percent'] = round(compression_ratio, 1)
                result['size_reduction_bytes'] = result['original_size_bytes'] - result['processed_size_bytes']
                result['size_reduction_mb'] = round(result['size_reduction_bytes'] / (1024 * 1024), 2)

                # Get original image metadata
                original_metadata = self.get_image_metadata(original_path)
                result.update({f'original_{k}': v for k, v in original_metadata.items()})

                # Calculate dimension changes
                if 'processed_width' in result and 'original_width' in result:
                    width_change = ((result['processed_width'] - result['original_width']) / result['original_width']) * 100
                    height_change = ((result['processed_height'] - result['original_height']) / result['original_height']) * 100
                    result['width_change_percent'] = round(width_change, 1)
                    result['height_change_percent'] = round(height_change, 1)
            else:
                result['original_found'] = False
        else:
            result['original_matched'] = False

        # Get manifest settings for this image
        if original_filename and self.manifest_data:
            images = self.manifest_data.get('images', [])
            for image_config in images:
                if image_config['filename'] == original_filename:
                    result['manifest_quality'] = image_config.get('quality', 'default')
                    result['manifest_grain_intensity'] = image_config.get('grain_intensity', 'default')
                    break

        return result

    def analyze_all_images(self) -> List[Dict[str, Any]]:
        """Analyze all processed images"""
        if not self.processed_dir.exists():
            print(f"Processed directory '{self.processed_dir}' not found.")
            return []

        results = []
        webp_files = list(self.processed_dir.glob("*.webp"))

        print(f"Analyzing {len(webp_files)} processed images...")

        for webp_file in sorted(webp_files):
            try:
                analysis = self.analyze_single_image(webp_file)
                results.append(analysis)
            except Exception as e:
                print(f"Error analyzing {webp_file}: {e}")
                results.append({
                    'processed_filename': webp_file.name,
                    'error': str(e)
                })

        return results

    def generate_summary_stats(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from all analyses"""
        if not analyses:
            return {}

        # Filter out error entries and thumbnails for main stats
        valid_analyses = [a for a in analyses if 'error' not in a and not a.get('is_thumbnail', False)]
        thumbnail_analyses = [a for a in analyses if 'error' not in a and a.get('is_thumbnail', False)]

        if not valid_analyses:
            return {'error': 'No valid analyses found'}

        # Calculate basic stats
        total_processed_size = sum(a['processed_size_bytes'] for a in valid_analyses)
        total_original_size = sum(a.get('original_size_bytes', 0) for a in valid_analyses if 'original_size_bytes' in a)

        compression_ratios = [a['compression_ratio_percent'] for a in valid_analyses if 'compression_ratio_percent' in a]
        processed_sizes = [a['processed_size_bytes'] for a in valid_analyses]

        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_files_analyzed': len(analyses),
            'main_images': len(valid_analyses),
            'thumbnails': len(thumbnail_analyses),
            'files_with_errors': len([a for a in analyses if 'error' in a]),

            # Size statistics
            'total_processed_size_mb': round(total_processed_size / (1024 * 1024), 2),
            'total_original_size_mb': round(total_original_size / (1024 * 1024), 2) if total_original_size > 0 else 0,
            'total_space_saved_mb': round((total_original_size - total_processed_size) / (1024 * 1024), 2) if total_original_size > 0 else 0,

            # Compression statistics
            'average_compression_ratio_percent': round(sum(compression_ratios) / len(compression_ratios), 1) if compression_ratios else 0,
            'best_compression_ratio_percent': max(compression_ratios) if compression_ratios else 0,
            'worst_compression_ratio_percent': min(compression_ratios) if compression_ratios else 0,

            # File size statistics
            'average_processed_size_bytes': round(sum(processed_sizes) / len(processed_sizes), 0) if processed_sizes else 0,
            'largest_processed_size_bytes': max(processed_sizes) if processed_sizes else 0,
            'smallest_processed_size_bytes': min(processed_sizes) if processed_sizes else 0,

            # Format distribution
            'original_formats': {},
            'quality_settings': {},
        }

        # Count original formats
        for analysis in valid_analyses:
            original_format = analysis.get('original_format', 'unknown')
            summary['original_formats'][original_format] = summary['original_formats'].get(original_format, 0) + 1

        # Count quality settings
        for analysis in valid_analyses:
            quality = analysis.get('manifest_quality', 'unknown')
            summary['quality_settings'][str(quality)] = summary['quality_settings'].get(str(quality), 0) + 1

        return summary

    def print_table_format(self, analyses: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Print results in table format"""
        print("\n" + "="*120)
        print("PROCESSED IMAGES ANALYSIS REPORT")
        print("="*120)

        # Summary section
        print(f"\nSUMMARY:")
        print(f"  Analysis Date: {summary.get('analysis_timestamp', 'N/A')}")
        print(f"  Total Files: {summary.get('total_files_analyzed', 0)}")
        print(f"  Main Images: {summary.get('main_images', 0)}")
        print(f"  Thumbnails: {summary.get('thumbnails', 0)}")
        print(f"  Files with Errors: {summary.get('files_with_errors', 0)}")

        print(f"\nSIZE STATISTICS:")
        print(f"  Total Processed Size: {summary.get('total_processed_size_mb', 0):.2f} MB")
        print(f"  Total Original Size: {summary.get('total_original_size_mb', 0):.2f} MB")
        print(f"  Total Space Saved: {summary.get('total_space_saved_mb', 0):.2f} MB")

        print(f"\nCOMPRESSION STATISTICS:")
        print(f"  Average Compression: {summary.get('average_compression_ratio_percent', 0):.1f}%")
        print(f"  Best Compression: {summary.get('best_compression_ratio_percent', 0):.1f}%")
        print(f"  Worst Compression: {summary.get('worst_compression_ratio_percent', 0):.1f}%")

        # Format distribution
        if summary.get('original_formats'):
            print(f"\nORIGINAL FORMATS:")
            for fmt, count in summary['original_formats'].items():
                print(f"  {fmt}: {count} files")

        # Detailed file listing
        print(f"\nDETAILED FILE ANALYSIS:")
        print("-"*120)
        print(f"{'Filename':<30} {'Size':<8} {'Orig':<8} {'Comp%':<6} {'Dims':<12} {'Quality':<7} {'Type':<4}")
        print("-"*120)

        for analysis in sorted(analyses, key=lambda x: x.get('processed_size_bytes', 0), reverse=True):
            if 'error' in analysis:
                print(f"{analysis['processed_filename']:<30} ERROR: {analysis['error']}")
                continue

            filename = analysis['processed_filename'][:29]
            size_mb = f"{analysis.get('processed_size_mb', 0):.1f}MB"
            orig_mb = f"{analysis.get('original_size_mb', 0):.1f}MB" if 'original_size_mb' in analysis else "N/A"
            comp_ratio = f"{analysis.get('compression_ratio_percent', 0):.1f}%" if 'compression_ratio_percent' in analysis else "N/A"
            dims = f"{analysis.get('processed_width', 0)}x{analysis.get('processed_height', 0)}"
            quality = str(analysis.get('manifest_quality', 'N/A'))[:6]
            img_type = "THUMB" if analysis.get('is_thumbnail') else "MAIN"

            print(f"{filename:<30} {size_mb:<8} {orig_mb:<8} {comp_ratio:<6} {dims:<12} {quality:<7} {img_type:<4}")

    def save_json_format(self, analyses: List[Dict[str, Any]], summary: Dict[str, Any], filename: str):
        """Save results in JSON format"""
        output = {
            'summary': summary,
            'detailed_analysis': analyses
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to {filename}")

    def save_csv_format(self, analyses: List[Dict[str, Any]], filename: str):
        """Save results in CSV format"""
        if not analyses:
            print("No data to save")
            return

        # Get all possible fieldnames
        fieldnames = set()
        for analysis in analyses:
            fieldnames.update(analysis.keys())
        fieldnames = sorted(list(fieldnames))

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analyses)
        print(f"Analysis saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyze processed WebP images')
    parser.add_argument(
        '--output-format',
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format (default: table)'
    )
    parser.add_argument(
        '--sort-by',
        choices=['size', 'compression', 'name'],
        default='size',
        help='Sort results by (default: size)'
    )
    parser.add_argument(
        '--output-file',
        help='Save results to file (for json/csv formats)'
    )
    parser.add_argument(
        '--processed-dir',
        default='processed',
        help='Directory containing processed images (default: processed)'
    )
    parser.add_argument(
        '--raw-dir',
        default='raw',
        help='Directory containing original images (default: raw)'
    )
    parser.add_argument(
        '--manifest',
        default='manifest.yaml',
        help='Path to manifest file (default: manifest.yaml)'
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ProcessedImageAnalyzer(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        manifest_file=args.manifest
    )

    # Analyze images
    analyses = analyzer.analyze_all_images()
    summary = analyzer.generate_summary_stats(analyses)

    # Sort results
    if args.sort_by == 'size':
        analyses.sort(key=lambda x: x.get('processed_size_bytes', 0), reverse=True)
    elif args.sort_by == 'compression':
        analyses.sort(key=lambda x: x.get('compression_ratio_percent', 0), reverse=True)
    elif args.sort_by == 'name':
        analyses.sort(key=lambda x: x.get('processed_filename', ''))

    # Output results
    if args.output_format == 'table':
        analyzer.print_table_format(analyses, summary)
    elif args.output_format == 'json':
        output_file = args.output_file or f"image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.save_json_format(analyses, summary, output_file)
    elif args.output_format == 'csv':
        output_file = args.output_file or f"image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        analyzer.save_csv_format(analyses, output_file)

if __name__ == '__main__':
    main()
