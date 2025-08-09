#!/bin/bash

# Image Processing Script for Wheel of Heaven Data Images
# This script provides an easy interface to the Python image processor

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup Python environment
setup_python() {
    print_info "Setting up Python environment..."

    if ! command_exists python3; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate

    # Install/upgrade pip
    python -m pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_info "Installing Python dependencies..."
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing core dependencies..."
        pip install Pillow PyYAML numpy
    fi

    print_success "Python environment ready!"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if raw directory exists
    if [ ! -d "raw" ]; then
        print_error "Raw directory not found. Please ensure you have a 'raw' directory with images."
        exit 1
    fi

    # Check if config file exists
    if [ ! -f "manifest.yaml" ]; then
        print_error "Configuration file 'manifest.yaml' not found."
        exit 1
    fi

    # Count images in raw directory
    image_count=$(find raw -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)
    print_info "Found $image_count image(s) in raw directory"

    if [ "$image_count" -eq 0 ]; then
        print_warning "No images found in raw directory"
    fi

    print_success "Prerequisites check complete!"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup           Setup Python environment and dependencies"
    echo "  check           Check prerequisites"
    echo "  dry-run         Show what would be processed without doing it"
    echo "  process         Process all images according to configuration"
    echo "  clean           Clean up generated files (processed images, backups)"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup       # First time setup"
    echo "  $0 dry-run     # See what would be processed"
    echo "  $0 process     # Process all images"
    echo ""
}

# Function to run dry-run
run_dry_run() {
    print_info "Running dry-run (no files will be modified)..."
    source venv/bin/activate
    python scripts/process_images.py --dry-run
}

# Function to process images
process_images() {
    print_info "Starting image processing..."
    source venv/bin/activate
    python scripts/process_images.py

    if [ $? -eq 0 ]; then
        print_success "Image processing completed successfully!"

        # Show output directory info
        if [ -d "processed" ]; then
            processed_count=$(find processed -name "*.webp" | wc -l)
            print_info "Generated $processed_count WebP files in 'processed' directory"
        fi
    else
        print_error "Image processing failed. Check the logs for details."
        exit 1
    fi
}

# Function to clean up
clean_up() {
    print_warning "This will remove all processed images and backups. Continue? (y/N)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."

        if [ -d "processed" ]; then
            rm -rf processed
            print_info "Removed processed directory"
        fi

        if [ -d "backup" ]; then
            rm -rf backup
            print_info "Removed backup directory"
        fi

        if [ -f "image_processing.log" ]; then
            rm image_processing.log
            print_info "Removed log file"
        fi

        print_success "Cleanup complete!"
    else
        print_info "Cleanup cancelled"
    fi
}

# Main script logic
case "${1:-}" in
    setup)
        setup_python
        ;;
    check)
        check_prerequisites
        ;;
    dry-run)
        check_prerequisites
        if [ ! -d "venv" ]; then
            print_info "Virtual environment not found. Running setup first..."
            setup_python
        fi
        run_dry_run
        ;;
    process)
        check_prerequisites
        if [ ! -d "venv" ]; then
            print_info "Virtual environment not found. Running setup first..."
            setup_python
        fi
        process_images
        ;;
    clean)
        clean_up
        ;;
    help|--help|-h)
        show_usage
        ;;
    "")
        print_info "Wheel of Heaven Image Processor"
        echo ""
        show_usage
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
