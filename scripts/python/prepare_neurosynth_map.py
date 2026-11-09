"""
Convert shared VP CIFTI to continuous MNI NIfTI for Neurosynth CorrelationDecoder.

Cortical grayordinates are projected to the 2mm MNI volume via nearest-neighbour
vertex-to-voxel mapping using midthickness surface coordinates. Subcortical
grayordinates are already stored as 2mm MNI ijk voxels and are mapped directly.
"""
import logging
import numpy as np
import nibabel as nib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

CIFTI_PATH = (
    PROJECT_DIR / 'results' / 'figures' / 'ciftis'
    / 'group_n5_primary_analysis' / 'vp_category_shared_tau-0.6.dscalar.nii'
)
OUT_DIR = PROJECT_DIR / 'results' / 'neurosynth_decoding'
OUT_NII = OUT_DIR / 'shared_vp_mni.nii.gz'

# Update paths below if running on cluster
MNI_2MM   = Path('/home/ozvar/Atlases/HCPpipelines/global/templates/MNI152_T1_2mm.nii.gz')
ATLAS_DIR = Path('/home/ozvar/Atlases/HCP_S1200_Atlas')

SURF_FILES = {
    'CIFTI_STRUCTURE_CORTEX_LEFT':  ATLAS_DIR / 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii',
    'CIFTI_STRUCTURE_CORTEX_RIGHT': ATLAS_DIR / 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii',
}


def project_mm_to_voxel(coords_mm: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    hom = np.column_stack([coords_mm, np.ones(len(coords_mm))])
    return np.round((inv_affine @ hom.T).T[:, :3]).astype(int)


def accumulate(ijk: np.ndarray, values: np.ndarray,
               vol: np.ndarray, counts: np.ndarray) -> None:
    shape = np.array(vol.shape)
    valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
    i, j, k = ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]
    np.add.at(vol, (i, j, k), values[valid])
    np.add.at(counts, (i, j, k), 1)


def cifti_to_mni(cifti_path: Path, mni_ref_path: Path, out_path: Path) -> None:
    log.info(f'Loading CIFTI: {cifti_path.name}')
    cifti_img = nib.load(str(cifti_path))
    data = cifti_img.get_fdata(dtype=np.float32).squeeze()
    log.info(f'  {data.shape[0]} grayordinates, {(data != 0).sum()} nonzero')

    mni_img = nib.load(str(mni_ref_path))
    vol_shape = mni_img.shape[:3]
    inv_affine = np.linalg.inv(mni_img.affine)

    vol = np.zeros(vol_shape, dtype=np.float32)
    counts = np.zeros(vol_shape, dtype=np.int32)

    brain_axis = cifti_img.header.get_axis(1)

    for name, slc, bm in brain_axis.iter_structures():
        struct_data = data[slc]
        n_nonzero = int((struct_data != 0).sum())

        if bm.surface_mask.any():
            if name not in SURF_FILES:
                log.warning(f'  {name}: no surface file, skipping ({n_nonzero} nonzero)')
                continue
            surf = nib.load(str(SURF_FILES[name]))
            coords_mm = surf.darrays[0].data[bm.vertex]
            ijk = project_mm_to_voxel(coords_mm, inv_affine)
            accumulate(ijk, struct_data, vol, counts)
            log.info(f'  {name}: {len(bm.vertex)} vertices, {n_nonzero} nonzero -> projected')

        elif bm.volume_mask.any():
            ijk = bm.voxel  # already in 2mm MNI ijk space (affine matches MNI_2MM)
            accumulate(ijk, struct_data, vol, counts)
            log.info(f'  {name}: {len(bm.voxel)} voxels, {n_nonzero} nonzero -> mapped')

    hits = counts > 0
    vol[hits] /= counts[hits]
    log.info(f'Output: {hits.sum()} nonzero MNI voxels')

    nib.save(nib.Nifti1Image(vol, mni_img.affine, mni_img.header), str(out_path))
    log.info(f'Saved: {out_path}')


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cifti_to_mni(CIFTI_PATH, MNI_2MM, OUT_NII)
