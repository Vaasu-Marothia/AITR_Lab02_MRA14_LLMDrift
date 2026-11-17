"""
src/__init__.py
Exposes primary module components.
"""
from .data_loader import load_triviaqa, partition_dataset, create_drift_stream, prepare_and_save_partitions
from .evaluator import compute_exact_match, compute_f1, normalize_answer, score_batch
from .drift_detector import AccuracyMonitor, JSDMonitor, MMDMonitor, calibrate_threshold
from .visualizer import plot_roc_curves, plot_detection_timeline, plot_ablation
