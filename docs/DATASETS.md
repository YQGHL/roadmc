# Development dataset attribution and evidence boundaries

Reviewed 2026-10-20. This is an attribution ledger for data actually inspected during development, not a claim that all sources were used for training or that any real-domain acceptance test passed. The current paper's state/sampling results are controlled synthetic fixtures; the datasets below are development context and must not be described as validation for those material-history mechanisms.

## Sources used in bounded investigations

| Source / citation key | Original record | Use and unresolved evidence | License evidence |
|---|---|---|---|
| Pan et al. (2023), `Pan2023Texture` | [Paper and supplementary data](https://doi.org/10.1098/rsta.2022.0166) | DAC/MS/OGFC/SMA matrices; file-level normalized PSD/ACF/distribution comparisons. XY spacing supported; Z units, release processing and independent-road identity unresolved. Historical TEST exposed. | Retain the original supplementary-data terms; do not infer a separate dataset license solely from article access. |
| Fan et al., `Fan2019Pothole`, `Fan2018Reconstruction` | [Author dataset repository](https://github.com/ruirangerfan/stereo_pothole_datasets) | Dataset3 Frame01–03 investigated; repeated observations, XYZ gaps, coordinate/scale/reference-surface admission limits remain. Author README requests both papers. | Upstream repository displays MIT; preserve its notice and verify applicable asset terms before redistribution. |
| Staat, Bühler and Giessler (2026), `Staat2026Concrete` | [Data DOI](https://doi.org/10.35097/ku8qs1jt347zxyeq), [KIT institutional record](https://publikationen.bibliothek.kit.edu/1000197578) | ELA rings and normal concrete PLY/repeats; road exposure and inferred PLY scale prevent independent acceptance. Exposed aggregate is a normal finish, not a raveling label. | KIT institutional record: CC BY-NC-SA 4.0. |
| SUT-Crack (2023), `SUTCrack2023` | [Mendeley v1](https://data.mendeley.com/datasets/gsbmknrhkv/1) | Locally 130 masks but only 3 originals; same-ID pairs are candidates, pairing quality not independently established. Physical measurement claims need local calibration evidence. | Record: CC BY 4.0. |
| Qiu et al. (2026), `Qiu2026SUDRoad` | [Author repository](https://github.com/msqiu/SUD-Road), [requested paper](https://doi.org/10.1016/j.compeleceng.2026.110993) | Native LAS/local geometry. Author's seven class names do not establish a mapping to the observed byte values. LAS scale factors do not establish physical coordinate units. | Repository displays Apache-2.0; original SUD and LAS asset licensing must also be checked. |
| Ranieri, Moscoso Thompson and Biasotti (2022), `PotholeMix2022` | [Pothole Mix v2](https://doi.org/10.17632/kfth5g2xk3.2) | Selected RGB-D clips, disparity and automatic video masks. Local red-color mask extraction is not manual GT; capture groups are not independent roads; metric Z unresolved. | Record: CC BY-NC 3.0; consult component-dataset terms too. |
| Kane (2023), `Kane2023Texture` | [Laboratory texture/friction v1](https://doi.org/10.17632/kkcztnxzph.1) | Selected SUR/TXT and metadata inspection; no paired particle-phase/retention truth for the proposed material evolution. | Record: CC BY 4.0. |
| Kováč, Brna and Pisca (2026), `Kovac2026Field` | [Raw field scans v1](https://doi.org/10.5281/zenodo.20606458) | Selected scan strips only, not all 45 sections. Local extra-column semantics and coordinate/processing admission limits remain. | Preserve record terms; historical intake records CC BY 4.0, not relicensed by this project. |
| `MLS2026Paired` | [Paired MLS dataset](https://doi.org/10.5281/zenodo.19267189), [author institution record](https://research.aalto.fi/fi/datasets/mobile-laser-scanning-for-road-pavement-distress-detection-datase/) | Selected PPS-plus/Roamer-R4DW support/format observations; not an independently calibrated full benchmark here. | Institution: CC BY-NC-SA 4.0. |
| `M2SRoAD` | [Author repository](https://github.com/chinitaberrio/M2S-RoAD), [linked data release](https://huggingface.co/datasets/DarrylT/M2S-RoAD) | Single exposed subsample/weak ROI for dimensionless observation-support sensitivity; not the complete dataset, not road-level train/test or sensor calibration. | Preserve data-release terms; historical intake records CC BY-NC-ND 4.0; not independently relicensed. |
| `OSF98vkz` | [Original OSF project](https://osf.io/98vkz/), [associated article](https://doi.org/10.1371/journal.pone.0318188) | Selected surface files/format inspection only. No disease labels, road independence or material evolution claim inferred. | Project access returned 403 during this check; confirm original terms before reuse or redistribution. |

DEF JPEG references and other incompletely identified visual samples remain provenance-unresolved. Do not invent a dataset name, license, author or DOI for them; do not use them as quantitative real-domain evidence until traced.

## How to cite and use these sources

Use [datasets.bib](datasets.bib) as a starting point. Cite the actual version and asset used and retain the provider's notice. A paper citation does not replace a dataset citation or license. SUT's 130 masks are not 130 complete image pairs; repeated frames, sampling seeds, configurations and points are not independent roads. No completed synthetic-training-to-independent-real-segmentation benefit is established by this attribution ledger.

Raw data and large outputs stay outside Git. The repository's MIT license does not grant rights to third-party data, and noncommercial/share-alike/no-derivatives terms are not removed by generating statistics. This ledger records source terms, not a determination of rights for every derivative.

## Verification scope

Author repositories and original/institutional records were checked on 2026-10-20 where accessible. DOI endpoints for the Royal Society supplement and KIT RADAR, the direct MLS Zenodo page and OSF were not all readable in this run; author/institution records and existing intake receipts supplement those limitations. No full-literature review, new data download or renewed scientific evaluation was performed.
