"""HP agent capture submodule.

Contains:
- ``rssi_iw``: parse ``iw scan``, compute SNR from per-BSS signal + ath9k noise floor.
- ``csi_atheros``: parse Atheros CSI Tool binary frames, provide live + fixture sources.
- ``csi_features``: per-window feature aggregation over CSIFrame lists.
- ``webcam_stream``: JPEG producer.
"""
