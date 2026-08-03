# Data setup

Phase 0 uses deterministic synthetic complex ellipse phantoms and downloads nothing.

The real-data phase will support official fastMRI knee single-coil HDF5 volumes. Users must accept
the fastMRI data-use agreement independently. The loader will read `FASTMRI_ROOT` or an explicit
configuration field, split by volume, and never commit data, credentials, or patient-derived files.

On Colab, store data and run artifacts in Google Drive and expose the dataset root through the
environment. Do not copy the dataset into the Git repository.

