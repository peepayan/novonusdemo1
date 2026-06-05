"""Stage 2 add-on: dedicated EMG-to-force regressor trained on DB2 block E3.

Trains directly on the real measured force-sensor data in E3 (not the
EMG-amplitude proxy used by the main Stage 2 intent classifier). The
intent classifier in src/stage2_lstm/ is NOT touched by this module.
"""
