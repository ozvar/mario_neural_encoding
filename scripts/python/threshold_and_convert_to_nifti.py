"""
Threshold CIFTI file and convert to NIFTI format.

Usage:
    python threshold_and_convert_cifti.py <input_cifti> <threshold>
"""
import sys
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np


def threshold_cifti(input_path: Path, threshold: float, output_path: Path):
    """Set all values below threshold to 0 and save."""
    print(f"Loading: {input_path}")
    img = nib.load(input_path)
    data = img.get_fdata()

    print(f"Original - min: {data.min():.6f}, max: {data.max():.6f}, mean: {data.mean():.6f}")
    n_below = (data < threshold).sum()
    print(f"Values below {threshold}: {n_below} ({100*n_below/data.size:.2f}%)")

    data[data < threshold] = 0

    print(f"Thresholded - min: {data.min():.6f}, max: {data.max():.6f}, mean: {data.mean():.6f}")

    new_img = nib.Cifti2Image(dataobj=data, header=img.header)
    nib.save(new_img, output_path)
    print(f"Saved thresholded CIFTI: {output_path}")

    return output_path


def convert_to_nifti(cifti_path: Path, nifti_path: Path):
    """Convert CIFTI to NIFTI using wb_command."""
    cmd = [
        'wb_command',
        '-cifti-convert',
        '-to-nifti',
        str(cifti_path),
        str(nifti_path)
    ]

    print(f"Converting to NIFTI: {nifti_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    print(f"Saved NIFTI: {nifti_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python threshold_and_convert_cifti.py <input_cifti> <threshold>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    threshold = float(sys.argv[2])

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Create output paths
    stem = input_path.stem.replace('.dscalar', '')
    parent = input_path.parent

    thresholded_cifti = parent / f"{stem}_threshold_{threshold}.dscalar.nii"
    output_nifti = parent / f"{stem}_threshold_{threshold}_nifti.nii.gz"

    print(f"Input: {input_path}")
    print(f"Threshold: {threshold}")
    print(f"Output CIFTI: {thresholded_cifti}")
    print(f"Output NIFTI: {output_nifti}")
    print()

    # Step 1: Threshold
    threshold_cifti(input_path, threshold, thresholded_cifti)
    print()

    # Step 2: Convert
    convert_to_nifti(thresholded_cifti, output_nifti)
    print()
    print("Done.")


if __name__ == '__main__':
    main()
