# kpms-zoomcheck

Diagnostics and normalization helpers for zoom mismatches in DLC/SLEAP keypoints used with keypoint-MoSeq (KPMS).

## Current results (before normalization)

| Video                  | Median bodylen (px) | CV    |
|-------------------------|--------------------:|------:|
| bma5_oft_dlc_results.h5 | 74.8                | 0.120 |
| bma6_oft_dlc_results.h5 | 44.7                | 0.122 |

- KS(body length) = 0.931 (p~0) -> large difference
- Zoom-classifier acc = 0.592 (chance ~ 0.50)
- ROC AUC = 0.561

These confirm zoom differences are detectable and need correction.
