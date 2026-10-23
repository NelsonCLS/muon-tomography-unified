# References and local reading copies

Scientific references are linked to their publisher or author archive. Downloaded
PDFs under `docs/` are local reading copies, excluded from new Git snapshots and
source distributions. They are not required to install or run the software.
Earlier Git history may still contain the former copies; this cleanup does not
rewrite published history.

- Lim and Qiu, *μ-Net: ConvNext-Based U-Nets for Cosmic Muon Tomography* (2023):
  [arXiv:2312.17265](https://arxiv.org/abs/2312.17265).
- Schultz et al., *Statistical Reconstruction for Cosmic Ray Muon Tomography*
  (2007): [DOI:10.1109/TIP.2007.901239](https://doi.org/10.1109/TIP.2007.901239).
- Sehgal et al., *Voxelization based PoCA point cloud filtration algorithm for
  image reconstruction for Muon Tomography* (2020):
  [DOI:10.1088/1748-0221/15/09/P09012](https://doi.org/10.1088/1748-0221/15/09/P09012).
- *Cosmic Ray Energy Spectrum from Measurements of Air Showers* (2013):
  [arXiv:1303.3565](https://arxiv.org/abs/1303.3565).
- *A parametrization of the cosmic-ray muon flux at sea-level* (2015):
  [arXiv:1509.06176](https://arxiv.org/abs/1509.06176).

These references do not establish that every repository variant implements the
published estimator exactly. In particular, the pooled PoCA estimator is a local
experimental adaptation; its numerator/exposure pooling and development choices
are documented in the source and [scientific contracts](scientific_contracts.md).
Upstream code and model attribution are recorded separately in [NOTICE](../NOTICE)
and [CITATION.cff](../CITATION.cff).
