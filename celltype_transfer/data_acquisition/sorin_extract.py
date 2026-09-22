"""
Sorin 2023 lung adenocarcinoma (IMC) - build a per-cell table.

Unlike the other cohorts, Sorin does NOT ship a cell table. LungData.zip contains, per sample:
  LUAD_IMC_MaskTif/<s>.tif                     18-frame marker image (uint8)
  LUAD_IMC_Segmentation/<s>/nuclei_multiscale.mat   `nucleiOccupancyIndexed` = labelled cell mask
  LUAD_IMC_CellType/<s>.mat                    `cellTypes` = one label per cell, in label order

So we run the feature extraction ourselves: for every cell label, the centroid, the pixel area,
and the mean of each of the 18 channels inside that cell. Verified on LUAD_D001 - the mask's
max label equals the number of cellTypes exactly, and the mask and TIFF have identical shape;
both are asserted per sample below rather than assumed.

Caveat worth carrying forward: the channel images are **uint8 (0-255)**, so they are already
quantised for display. Real IMC counts are 16-bit. The dynamic range is coarser than the other
cohorts - Stage 1's rank transform absorbs this, but it is a real limitation, not a detail.

Output: Datasets/Sorin/sorin_cells.csv   (then read by the normal spec in config.py)
"""
import os, re, sys, zipfile, io, time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import ndimage as ndi
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ZIP = os.path.join(ROOT, "Datasets", "Sorin", "LungData.zip")
OUT = os.path.join(ROOT, "Datasets", "Sorin", "sorin_cells.csv")

# From the Zenodo record's "Channel index names" table, in channel order.
CHANNELS = ['CD117', 'CD11c', 'CD14', 'CD163', 'CD16', 'CD20', 'CD31', 'CD3', 'CD4',
            'CD68', 'CD8a', 'CD94', 'DNA1', 'FoxP3', 'HLA-DR', 'MPO', 'Pancytokeratin', 'TTF1']


def samples(z):
    names = [i.filename for i in z.infolist() if not i.is_dir() and '__MACOSX' not in i.filename]
    ct = {re.search(r'CellType/(.+)\.mat$', n).group(1)
          for n in names if '/LUAD_IMC_CellType/' in n and n.endswith('.mat')}
    sg = {re.search(r'Segmentation/([^/]+)/nuclei_multiscale\.mat$', n).group(1)
          for n in names if n.endswith('nuclei_multiscale.mat')}
    tf = {re.search(r'MaskTif/(.+)\.tif$', n).group(1)
          for n in names if '/LUAD_IMC_MaskTif/' in n and n.endswith('.tif')}
    return sorted(ct & sg & tf), dict(cellType=len(ct), segmentation=len(sg), maskTif=len(tf))


def one(z, s):
    ct = sio.loadmat(io.BytesIO(z.read(f'LungData/LUAD_IMC_CellType/{s}.mat')))
    types = np.array([str(x[0][0]) if len(x[0]) else '' for x in ct['cellTypes']])

    sg = sio.loadmat(io.BytesIO(z.read(
        f'LungData/LUAD_IMC_Segmentation/{s}/nuclei_multiscale.mat')))
    lab = sg['nucleiOccupancyIndexed']

    im = Image.open(io.BytesIO(z.read(f'LungData/LUAD_IMC_MaskTif/{s}.tif')))
    assert im.n_frames == len(CHANNELS), f"{s}: {im.n_frames} frames, expected {len(CHANNELS)}"
    ch = np.stack([np.array(im.seek(k) or im) for k in range(im.n_frames)])

    assert lab.shape == ch.shape[1:], f"{s}: mask {lab.shape} != image {ch.shape[1:]}"
    n = int(lab.max())
    assert n == len(types), f"{s}: {n} mask labels but {len(types)} cellTypes"
    if n == 0:
        return None

    idx = np.arange(1, n + 1)
    cy, cx = np.array(ndi.center_of_mass(np.ones(lab.shape, bool), lab, idx)).T
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]

    d = {'image_id': s, 'cell_label': idx, 'x_px': cx, 'y_px': cy,
         'area_px2': area.astype('float32'), 'cellType': types}
    for k, name in enumerate(CHANNELS):
        d[name] = ndi.mean(ch[k].astype('float32'), lab, idx).astype('float32')
    return pd.DataFrame(d)


def main():
    z = zipfile.ZipFile(ZIP)
    ss, counts = samples(z)
    print(f"samples: {counts} -> {len(ss)} usable", flush=True)
    out, t0, bad = [], time.time(), []
    for i, s in enumerate(ss, 1):
        try:
            df = one(z, s)
            if df is not None:
                out.append(df)
        except AssertionError as e:
            bad.append(str(e))
        if i % 50 == 0 or i == len(ss):
            done = sum(len(x) for x in out)
            print(f"  {i}/{len(ss)} images | {done:,} cells | "
                  f"{time.time() - t0:.0f}s | {len(bad)} skipped", flush=True)
    D = pd.concat(out, ignore_index=True)
    # Sample ids look like LUAD_D001 and LUAD_V16B - the trailing A/B is a second core from the
    # same patient, so the patient is the id without it. Derived, not shipped; flagged in the audit.
    D['patient'] = D.image_id.str.replace(r'[AB]$', '', regex=True)
    D.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(f"{len(D):,} cells | {D.image_id.nunique()} images | {D.patient.nunique()} patients | "
          f"{D.cellType.nunique()} cell types")
    if bad:
        print(f"\nSKIPPED {len(bad)}:")
        for b in bad[:10]:
            print('  ', b)


if __name__ == "__main__":
    main()
