"""
Neurosynth CorrelationDecoder on the shared VP brain map.
Compute-intensive — run on cluster via run_neurosynth_decoding.sbatch.

On cluster: update NS_RAW_DIR and NS_CACHE to the paths you sftp'd the data to.
"""
import logging
from pathlib import Path

from nimare.dataset import Dataset
from nimare.decode.continuous import CorrelationDecoder
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Update these two paths on the cluster before running
NS_RAW_DIR = Path(
    '/home1/ozvar/Toolboxes/nimare_neurosynth_decoding/fresh_old/neurosynth'
)
NS_CACHE = Path(
    '/home1/ozvar/Git/cneuromod/mario_neural_encoding/inputs/neurosynth/neurosynth_dataset_v7'
)

INPUT_NII = PROJECT_DIR / 'results' / 'neurosynth_decoding' / 'shared_vp_mni.nii.gz'
OUT_TSV   = PROJECT_DIR / 'results' / 'neurosynth_decoding' / 'decoding_results.tsv'

N_CORES = -1  # all available cores


def load_or_build_dataset(raw_dir: Path, cache_path: Path) -> Dataset:
    if cache_path.exists():
        log.info(f'Loading cached dataset from {cache_path}...')
        try:
            dset = Dataset.load(str(cache_path))
            log.info(f'  {len(dset.ids)} studies')
            return dset
        except Exception as exc:
            log.warning(f'  Cache load failed ({exc}), rebuilding from raw...')

    log.info('Building Neurosynth dataset from raw files...')
    annotations_files = [{
        'features':   str(raw_dir / 'data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz'),
        'vocabulary': str(raw_dir / 'data-neurosynth_version-7_vocab-terms_vocabulary.txt'),
    }]
    dset = convert_neurosynth_to_dataset(
        coordinates_file=str(raw_dir / 'data-neurosynth_version-7_coordinates.tsv.gz'),
        metadata_file=str(raw_dir / 'data-neurosynth_version-7_metadata.tsv.gz'),
        annotations_files=annotations_files,
    )
    log.info(f'  {len(dset.ids)} studies loaded')
    local_cache = PROJECT_DIR / 'results' / 'neurosynth_decoding' / 'neurosynth_dataset_v7.pkl.gz'
    dset.save(str(local_cache))
    log.info(f'  Cached to {local_cache}')
    return dset


if __name__ == '__main__':
    if not INPUT_NII.exists():
        raise FileNotFoundError(
            f'NIfTI not found: {INPUT_NII} -- run prepare_neurosynth_map.py first'
        )

    dset = load_or_build_dataset(NS_RAW_DIR, NS_CACHE)

    log.info('Fitting CorrelationDecoder (runs MKDA per term -- may take hours)...')
    decoder = CorrelationDecoder(
        meta_estimator=MKDAChi2,
        frequency_threshold=0.001,
        target_image='z_desc-association',
        n_cores=N_CORES,
    )
    decoder.fit(dset)
    log.info('Transform...')

    results = decoder.transform(str(INPUT_NII))
    results = results.reset_index()
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(str(OUT_TSV), sep='\t', index=False)
    log.info(f'Saved {len(results)} terms -> {OUT_TSV.name}')
    log.info('Top 5:\n' + results.nlargest(5, 'r')[['feature', 'r']].to_string(index=False))
