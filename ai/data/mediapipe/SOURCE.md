# canonical_face_model.obj

Official MediaPipe canonical face mesh (468 vertices, same indexing as the Face Landmarker output).

- Source: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
- License: Apache License 2.0 (Google LLC)
- Coordinates: +x = the subject's left (MediaPipe's "LEFT_*" landmark sets), +y = up, units ≈ cm.

Used by `tests/test_face_regions.py` to check that the skin-region landmark indices in
`ai/colorimetry/face_color.py` sit where their names say, and avoid the eyes, brows, lips and nose.
