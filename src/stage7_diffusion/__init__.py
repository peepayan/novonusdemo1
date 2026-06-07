"""Stage 7 — multimodal Diffusion Policy with EMG conditioning.

The policy takes a current observation (camera frame + EMG window + robot
state) and generates the next 16-step joint-velocity + gripper action
sequence by iteratively denoising random noise, conditioned on the fused
multimodal observation. The conditioning vector includes the Stage 2 LSTM
hidden state, which carries the EMG force-intent encoding — this is the
core differentiator from a vision-only baseline.
"""
