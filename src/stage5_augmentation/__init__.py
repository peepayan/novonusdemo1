"""Stage 5 — physics-validated demonstration augmentation.

Re-simulates each Stage 4 baseline demo under randomized physical conditions
(grape pose, mass, friction, stiffness, gripper approach angle, lighting,
table friction) and keeps only the scenarios that pass a six-point
verification filter. Output is a unified dataset (Stage 4 demos + Stage 5
augmentations) ready for Diffusion Policy training in Stage 7.
"""
