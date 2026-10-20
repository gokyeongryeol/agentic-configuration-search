"""Leaf utility layer — imports no other layer of the package.

COCO data model, matching/metrics, tiling, prompt text, HF data conversion,
shared artifact loaders (io). Contract: anything here can be imported from
anywhere without creating a cycle (enforced by tests/test_architecture.py).
"""
