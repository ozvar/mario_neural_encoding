"""
Generate a word cloud from Neurosynth decoding results.

Usage:
    python scripts/python/plot_neurosynth_wordcloud.py [--n-terms N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

from mario_encoding.config import PATHS
from mario_encoding.utils import sns_styleset, save_figure


EXCLUDE_TERMS = [
    # Anatomical
    'parietal', 'frontal', 'temporal', 'occipital',
    'cortex', 'cortical', 'cortices',
    'premotor', 'motor cortex', 'primary motor', 'supplementary motor',
    'inferior', 'superior', 'gyrus', 'sulcus', 'lobe',
    'visual', 'motor', 'somatosensory', 'auditory',
    'network', 'networks', 'connectivity',
    'activations', 'recruitment', 'recruited',
    'supplementary', 'movements', 'eye', 'eye field', 'spl', 'ips',
    'parieto', 'ppc', 'field',
    # Generic
    'involved', 'produced', 'processes',
    'conditions', 'overlapping', 'overt',
    'common', 'general', 'modality', 'demand',
]


def _base_term(term: str) -> str:
    clean = term.replace('terms_abstract_tfidf__', '')
    if clean.endswith('s') and len(clean) > 3:
        return clean[:-1]
    return clean


def filter_terms(df: pd.DataFrame) -> pd.DataFrame:
    pattern = '|'.join(EXCLUDE_TERMS)
    df = df[~df['feature'].str.lower().str.contains(pattern)].copy()
    df['base_term'] = df['feature'].apply(_base_term)
    df = df.sort_values('r', ascending=False).drop_duplicates('base_term', keep='first')
    return df


def build_wordcloud(word_freq: dict) -> WordCloud:
    cmap = LinearSegmentedColormap.from_list(
        'dark_purples', plt.cm.Purples(np.linspace(0.4, 1.0, 256))
    )
    return WordCloud(
        width=2000,
        height=2000,
        background_color='white',
        colormap=cmap,
        relative_scaling=0.5,
        min_font_size=40,
        max_font_size=300,
        prefer_horizontal=1.0,
    ).generate_from_frequencies(word_freq)


def plot_wordcloud(wc: WordCloud, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'png'):
        fig, ax = plt.subplots(figsize=(7, 10))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        save_figure(fig, output_dir, f'neurosynth_wordcloud.{ext}', dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-terms', type=int, default=20)
    args = parser.parse_args()

    sns_styleset()

    df = pd.read_csv(PATHS['neurosynth'] / 'decoding_results.tsv', sep='\t')
    df_filtered = filter_terms(df)
    print(f"Filtered: {len(df_filtered)} terms after deduplication")
    print(df_filtered[['base_term', 'r']].head(20).to_string(index=False))

    word_freq = dict(zip(
        df_filtered.head(args.n_terms)['base_term'],
        df_filtered.head(args.n_terms)['r'],
    ))
    wc = build_wordcloud(word_freq)
    plot_wordcloud(wc, PATHS['figures'] / 'neurosynth')
    print("Done.")


if __name__ == '__main__':
    main()
