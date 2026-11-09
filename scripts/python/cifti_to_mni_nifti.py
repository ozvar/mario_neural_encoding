"""
Convert CIFTI to NIFTI for nimare decoding.

Usage:
    python cifti_to_nifti_simple.py <input_cifti> <output_nifti>
"""
import sys
import subprocess
from pathlib import Path


def cifti_to_nifti(cifti_path: Path, nifti_path: Path):
    """Convert CIFTI to NIFTI."""
    cmd = [
        'wb_command',
        '-cifti-convert',
        '-to-nifti',
        str(cifti_path),
        str(nifti_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    print(f"Converted: {nifti_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python cifti_to_nifti_simple.py <input_cifti> <output_nifti>")
        sys.exit(1)

    input_cifti = Path(sys.argv[1])
    output_nifti = Path(sys.argv[2])

    if not input_cifti.exists():
        print(f"ERROR: File not found: {input_cifti}")
        sys.exit(1)

    print(f"Input:  {input_cifti}")
    print(f"Output: {output_nifti}")

    cifti_to_nifti(input_cifti, output_nifti)
    print("Done.")


if __name__ == '__main__':
    main()
