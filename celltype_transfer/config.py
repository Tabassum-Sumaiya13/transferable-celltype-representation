"""
Cohort registry for the dynamic cross-cohort pipeline.

Everything cohort-specific lives HERE, declaratively. No `if cohort == "X"` anywhere else in
the codebase: `loaders.py` is a single generic reader driven by these specs, so adding a new
dataset means adding a dict, not editing code.

A spec describes three things:
  base   - the table with one row per cell (coordinates, label)
  joins  - extra tables merged onto it by key (patient ids, cell sizes)
  expr   - the marker matrix, merged by key (or already inside `base`)

Marker columns are declared by RULE, not by hand-listing:
  {'kind': 'pattern', 'regex': ...}   columns whose name matches
  {'kind': 'file',    'path': ...}    one marker name per line
  {'kind': 'exclude', 'cols': [...]}  everything except these

Stage 0 deliberately keeps EVERY column and EVERY cell. It does not decide what is a real
protein (DNA stains, elemental channels) or a real cell (`dirt`, `undefined`) - Stage 0b's
gene resolver and Stage 1b's coherence check make those calls on evidence.
"""
import os
import zlib

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "Datasets")
# CT_WORK lets a test point every stage at a scratch copy of the artifacts. Unset, nothing changes.
WORK     = os.environ.get("CT_WORK", os.path.join(ROOT, "work"))
RAW      = os.path.join(WORK, "raw")
VALUES   = os.path.join(WORK, "values")
PANEL    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "declared")
REPORTS  = os.environ.get("CT_REPORTS", os.path.join(ROOT, "reports"))
FIGURES  = os.path.join(REPORTS, "figures")

for _d in (WORK, RAW, VALUES, REPORTS, FIGURES):
    os.makedirs(_d, exist_ok=True)

SEED = 20260810

# The standard table every loader must produce, in this order.
STANDARD = ['cell_id', 'cohort', 'image_id', 'patient_id',
            'x_px', 'y_px', 'area_px2', 'native_label', 'label_confidence']

# ----------------------------------------------------------------------------- cohort registry
SPECS = {

'CRC': dict(
    tech='CODEX', tissue='colorectal', disease='colorectal cancer', role='train',
    px_um=0.37744,
    px_um_source='published - TCIA collection page states 377.44 nm/pixel (Keyence BZ-X710)',
    arrival='raw',
    citation='Schurch et al., Cell 2020, 10.1016/j.cell.2020.07.005',
    base=dict(
        file='CRC/CRC_clusters_neighborhoods_markers.csv', format='csv',
        cols={'image_id': 'File Name', 'patient_id': 'patients',
              'x_px': 'X:X', 'y_px': 'Y:Y', 'native_label': 'ClusterName'},
        # size:size is a 3-D voxel count (Z ranges 2..16), NOT a 2-D area -> divide by Z.
        area=dict(kind='divide', numerator='size:size', denominator='Z:Z'),
    ),
    joins=[],
    # every marker column carries its imaging cycle/channel: 'CD8 - cytotoxic T cells:Cyc_3_ch_2'
    expr=dict(markers=dict(kind='pattern', regex=r':Cyc_\d+_ch_\d+$')),
),

'UPMC': dict(
    tech='CODEX', tissue='head and neck', disease='head & neck squamous cell carcinoma',
    role='train',
    px_um=0.3774,
    px_um_source='ASSUMED - not published anywhere; same CODEX vendor stack as CRC',
    arrival='arcsinh',
    citation='Wu et al., Nat Biomed Eng 2022, 10.1038/s41551-022-00951-w',
    base=dict(
        file='UPMC/dataset_info/cell_locations_and_labels.csv', format='csv',
        cols={'image_id': 'ACQUISITION_ID', 'x_px': 'X', 'y_px': 'Y',
              'native_label': 'CLUSTER_LABEL', 'label_confidence': 'kNN.prob'},
        area=dict(kind='column', column='SIZE'),
        keep=['CELL_ID'],                      # needed as a join key below
    ),
    joins=[dict(
        file='UPMC/dataset_info/sample_metadata.csv', format='csv',
        left_on=['ACQUISITION_ID'], right_on=['acquisition_id'],
        cols={'patient_id': 'patient_id'},
    )],
    expr=dict(
        file='UPMC/dataset_info/labeled_arcsinh_norm_data.parquet', format='parquet',
        left_on=['ACQUISITION_ID', 'CELL_ID'], right_on=['sample_id', 'cell_id'],
        # bare one-name-per-line list, NO header row - say so or the first marker is eaten
        markers=dict(kind='file', path='UPMC/dataset_info/marker_names.csv', header=None),
    ),
),

'Keren': dict(
    tech='MIBI-TOF', tissue='breast', disease='triple-negative breast cancer', role='train',
    px_um=0.39063,
    px_um_source='published - paper states 800 um field of view / 2048 pixels',
    arrival='zscore',
    citation='Keren et al., Cell 2018, 10.1016/j.cell.2018.08.039',
    base=dict(
        file='Keren/cell_locations.csv', format='csv',
        cols={'image_id': 'SampleID', 'x_px': 'X', 'y_px': 'Y',
              'native_label': 'cluster_label'},
        keep=['cellLabelInImage'],
    ),
    joins=[
        dict(file='Keren/sample_metadata.csv', format='csv',
             left_on=['SampleID'], right_on=['SampleID'],
             cols={'patient_id': 'patient_id'}),
        dict(file='Keren/cellData.csv', format='csv',
             left_on=['SampleID', 'cellLabelInImage'],
             right_on=['SampleID', 'cellLabelInImage'],
             cols={'_area': 'cellSize'}),
    ],
    base_area=dict(kind='column', column='_area'),     # arrives via a join, applied after merge
    expr=dict(
        file='Keren/cell_expression.csv', format='csv',
        left_on=['SampleID', 'cellLabelInImage'], right_on=['SampleID', 'cellLabelInImage'],
        markers=dict(kind='exclude', cols=['SampleID', 'cellLabelInImage']),
    ),
),

'ferguson': dict(
    tech='IMC', tissue='skin', disease='cutaneous squamous cell carcinoma',
    # WAS role='holdout' - the frozen test-only cohort, 'one final number, once'.
    # Retired 2026-09-06: the protocol is now 7-fold leave-one-cohort-out with NO frozen
    # holdout, so ferguson is trained on in the six folds where it is not held out. This
    # dissolves H10 (the label space could not be blind to a holdout that no longer
    # exists) and makes the roster comparable to DeepCell Types' leave-one-dataset-out.
    # COST, stated in files/02: no cohort is now untouched by design decisions, and the
    # old zero-shot 0.3309 must be relabelled as an old-protocol number.
    role='train',
    px_um=1.0,
    px_um_source='hardware fact - IMC laser ablation spot size is 1 um by construction',
    arrival='raw',
    citation='Ferguson et al., Clin Cancer Res 2022, 10.1158/1078-0432.CCR-22-1332',
    base=dict(
        file='ferguson/csv_export/ferguson_cells_counts.csv', format='csv',
        cols={'image_id': 'imageID', 'patient_id': 'patientID',
              'x_px': 'x', 'y_px': 'y', 'native_label': 'cellType'},
        area=dict(kind='column', column='area'),
    ),
    joins=[],
    expr=dict(markers=dict(kind='file', path='ferguson/csv_export/marker_panel.csv',
                           column='marker')),
),

'Phillips': dict(
    tech='CODEX', tissue='skin', disease='cutaneous T cell lymphoma', role='train',
    px_um=0.3774,
    px_um_source='ASSUMED - not published; same Nolan-lab CODEX + Keyence stack as CRC',
    arrival='raw',
    citation='Phillips et al., Nat Commun 2021, 10.1038/s41467-021-26974-6',
    base=dict(
        file='Phillips/Raw_df_CODEX.csv', format='csv',
        cols={'image_id': 'FileName', 'patient_id': 'Patients',
              'x_px': 'X', 'y_px': 'Y', 'native_label': 'ClusterName'},
        # same Nolan CODEX export shape as CRC: `size` is a 3-D voxel count, Z is the depth
        area=dict(kind='divide', numerator='size', denominator='Z'),
    ),
    joins=[],
    # The marker block sits between the metadata columns and a wall of one-hot cell-type and
    # functional-gate columns ('B cells', 'PD-1+CD4+', ...). No pattern separates them, so the
    # block is named by its first and last member.
    expr=dict(markers=dict(kind='range', start='FOXP3', end='DRAQ5')),
),

'Danenberg': dict(
    tech='IMC', tissue='breast', disease='breast cancer (METABRIC)', role='train',
    px_um=1.0,
    px_um_source='hardware fact - IMC laser ablation spot size is 1 um by construction',
    arrival='raw',
    citation='Danenberg et al., Nat Genet 2022, 10.1038/s41588-022-01041-y (Zenodo 6036188)',
    # SingleCells.csv is extracted from MBTMEStrIMCPublic.zip (6.65 GB) by acquire/download.sh.
    # The zip also carries CellNeighbours.csv (802 MB), which Stage 4 would use - not needed
    # while Stage 4 is deferred, and the graph is rebuilt from coordinates anyway.
    base=dict(
        file='Danenberg/SingleCells.csv', format='csv',
        cols={'image_id': 'ImageNumber', 'patient_id': 'metabric_id',
              'x_px': 'Location_Center_X', 'y_px': 'Location_Center_Y',
              'native_label': 'cellPhenotype'},
        area=dict(kind='column', column='AreaShape_Area'),
    ),
    joins=[],
    # 39 marker columns sit in one contiguous block between the is_* region flags and the
    # Location_/AreaShape_ geometry columns, so the block is named by its ends.
    expr=dict(markers=dict(kind='range', start='Histone H3', end='DNA2')),
),

'Sorin': dict(
    tech='IMC', tissue='lung', disease='lung adenocarcinoma', role='train',
    px_um=1.0,
    px_um_source='hardware fact - IMC laser ablation spot size is 1 um by construction',
    arrival='raw-uint8',
    citation='Sorin et al., Nature 2023, 10.1038/s41586-022-05672-3',
    # NOTE: this cohort ships no cell table. celltype_transfer/data_acquisition/sorin_extract.py runs the
    # feature extraction from the shipped masks + 18-frame marker TIFFs and writes this CSV.
    base=dict(
        file='Sorin/sorin_cells.csv', format='csv',
        cols={'image_id': 'image_id', 'patient_id': 'patient',
              'x_px': 'x_px', 'y_px': 'y_px', 'native_label': 'cellType'},
        area=dict(kind='column', column='area_px2'),
    ),
    joins=[],
    expr=dict(markers=dict(kind='range', start='CD117', end='TTF1')),
),

}

# ------------------------------------------------------------------ rejected during Stage 0
# Kept as a written record so the decision is auditable and not silently repeated.
REJECTED = {
 'Risom': dict(
   citation='Risom et al., Cell 2022, 10.1016/j.cell.2021.12.023 (Mendeley 10.17632/d87vg86zd8.2)',
   reason=("Single_Cell_Data.csv (69,672 cells x 136 cols) ships NO X/Y centroid - only derived "
           "Neighbor_dist_* features. Risom_Mendeley_Image_Data.zip (298 MB, 82 Points) carries "
           "marker TIFs and four REGION masks (duct/epi/myoep/stroma) but no per-cell label mask, "
           "so centroids cannot be recovered. Unusable for a spatial graph."),
   note=("Zenodo 5945388, which several sources cite as this dataset, is a different study - "
         "the MIBI-TOF reproducibility paper on TONSIL (345,490 cells, 165 FOVs, 16 markers). "
         "It has centroids but is healthy tissue, so it fails the cancer-only rule."),
 ),
}

TRAIN   = [c for c, s in SPECS.items() if s['role'] == 'train']
HOLDOUT = [c for c, s in SPECS.items() if s['role'] == 'holdout']


def path(rel):
    return os.path.join(DATASETS, rel)


def raw_table(cohort):
    return os.path.join(RAW, f"{cohort}.parquet")


def value_table(cohort):
    return os.path.join(VALUES, f"{cohort}.parquet")


def full_table(cohort):
    """work/values/{cohort}_full.parquet - the wide value table every model stage reads
    (built by build_marker_vocabulary.py)."""
    return os.path.join(VALUES, f'{cohort}_full.parquet')


def rng(*parts):
    """Reproducible per-purpose RNG. crc32, not hash() - hash() is salted per process.

    One definition for every stage (it used to be copied into four files, byte-identical):
    the stream depends only on SEED and the purpose key, e.g. rng('split', 'CRC').
    """
    import numpy as np
    h = zlib.crc32('|'.join(map(str, parts)).encode()) & 0xffffffff
    return np.random.default_rng(SEED ^ h)
