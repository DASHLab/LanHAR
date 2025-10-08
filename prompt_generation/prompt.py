from data_processing import preprocess_acc_segment,_rotate_gyro_to_aligned
from data_analysis import *
import numpy as np
import pandas as pd


def generate_promt(acc_raw, gyro_raw, dataset_name, label, fs=50.0, mode="auto"):

  pre = preprocess_acc_segment(acc_raw, fs=fs, mode=mode)

  acc_raw = pre["aligned"]["acc_xyz"]    
  gyro_raw = _rotate_gyro_to_aligned(gyro_raw, pre["gravity"]["rot_R"])

  ACC_DATA_HERE = run_eda_slim(acc_raw, fs=fs)            
  GYRO_FEATURES_JSON_HERE = extract_gyro_features(gyro_raw, fs=fs)  
  GYRO_ACC_SUMMARY_JSON_HERE = gait_sync_and_impact(acc_raw, gyro_raw, fs=fs)


  if dataset_name == "uci":
    device_location = "waist"
  elif dataset_name == "motion":
    device_location = "front pockets"
  elif dataset_name == "hhar":
    device_location = "waists"
  elif dataset_name == "shoaib":
    position_label= [
            "left_pocket", "right_pocket", "wrist", "upper_arm", "belt"
        ]
    device_location = position_label[label]


  prompt = f"""

You are a professional wearable-device motion analysis expert, specializing in identifying motion patterns and gait characteristics from **short-window** accelerometer and gyroscope data. 
────────────────────────────────
【Data Introduction】
The data you need to anlyzez is a segment of IMU sensor reading. 
- It include (i) accelerometer data: coordinates are **gravity-aligned**, with +Z pointing **upward (vertical)**; Vert (vertical) and Horiz (horizontal) components can be used to distinguish **up–down oscillation vs. horizontal swinging**.
(ii) gyroscope data: raw angular velocity (rad/s), **no gravity alignment needed**.
- Sampling rate: 50 Hz
- Device location: {device_location}
- The context is that the IMU sensor reading may be in one of the following states: [walking, biking, jogging, going upstairs, walking downstairs, stilling]


────────────────────────────────
【Data Analysis】
All summary values for the current window; directly reference numbers with units
1) Accelerometer concise analysis (gravity-aligned):
{ACC_DATA_HERE}

Field description (Accelerometer):
- meta: N, fs
- stats: statistics of each axis and SVM (mean, std, min, max, p2p)
- body_summary: dynamic statistics after gravity separation (vert_rms, horiz_rms, vert_p2p, horiz_p2p)
- freq_summary: dominant frequency / step frequency and spectrum structure (dom_freq_hz, dom_freq_spm, low_high_energy_ratio, harmonic_2x_ratio)
- peaks_summary: peaks / rhythm (peak_count, ibi_mean_s, ibi_std_s, ibi_cv)
- jerk_summary: jerk intensity (rms / p95 / vert_rms / vert_p95, in m/s³)

2) Gyroscope features (angular velocity):
{GYRO_FEATURES_JSON_HERE}
Field description (Gyroscope):
- time_stats: mean_xyz, rms_xyz, p2p_xyz, wmag_mean/rms/p2p, zcr_xyz
- energy: energy_frac_xyz (distribution of ∑ω²), net_angle_xyz_rad, abs_angle_xyz_rad
- spectral: welch_wmag / welch_wx (dom_freq_hz, top_freqs), peak_rate_hz, step_freq_est_hz

3) Gyro–Acc synchronization & vertical impact (cross-sensor alignment):
{GYRO_ACC_SUMMARY_JSON_HERE}

Field description (Sync / Impact):
Coordinates: gravity aligned with +Z vertical upward; Vert vs Horiz can be used for up–down vs sideways motion
- step_metrics: step_rate_acc_hz, step_rate_gyro_hz, step_rate_diff_hz, n_steps_acc/gyro
- sync_metrics: n_matched_pairs, mean_lag_s, median_abs_lag_s, phase_consistency_0to1
- vertical_impact: per_step_peak_to_valley_mps2 list, impact_median_mps2, impact_p95_mps2

────────────────────────────────
【Knowledge】

1. Vertical impact–related features are indicators of high-impact tasks. 
Stairsdown usually produces the largest vertical impact (and correspondingly higher jerk) compared with ascent and level walking. 
Ascent/fast walking generally show moderate–strong vertical oscillation with a continuous rhythm; 
level walking shows medium amplitude with symmetric periodicity; still presents low amplitude and low jerk. 
Gyroscope ‘rotational intensity’ can help identify arm swing/trunk rotation typical of walking, but this depends on sensor placement and behavior

2. When the sensor is positioned near the body’s center of mass (e.g., waist or front pocket) and the axes are gravity-aligned, the relative contributions of vertical and horizontal acceleration, together with the energy distribution across gyroscope axes, can help differentiate dominant movement mechanisms:
Vertical-dominant patterns often appear in movements with clear vertical displacement or impact.
Horizontal-dominant patterns are more typical of level walking or low-intensity motion.
Note that gait speed, handrail use, restricted arm swing, and sensor placement can all influence these tendencies.

3. During coordinated gait such as level walking, accelerometer and gyroscope signals often show similar step-related periodicity in rate and timing. 
When arm motion is limited or movement becomes asymmetric (e.g., using a handrail on stairs or carrying objects), accelerometer rhythms may persist while gyroscope activity weakens or becomes less synchronized, leading to partial decoupling.

────────────────────────────────
【Task Introduction】
You task is analyzing the pattern of the above IMU data segment. 
We summarize the analysis into the following categories. 
Please respond strictly following the 7-point output format (numbers → direct verbal explanation; full units; mark data origin as [ACC] / [GYRO] / [SYNC]).
For each category, answer the most fitting option if one applies; if none applies, it is okay to leave it unselected.
If you think there is a pattern that particularly fits, you are also welcome to add.


• Category 1 **Intensity characteristics (overall magnitude and whether clearly non-still)**
- [ACC] stats.SVM mean, std / p2p (m/s²)
- [ACC] jerk_summary rms / p95 (m/s³)
- [GYRO] wmag_rms / wmag_p2p (rad/s)
→ Conclusion: intensity is low / medium / high? Clearly non-still?

• Category 2 **Directional characteristics (vertical vs horizontal; dominance of arm-swing / pitch / yaw)**
- [ACC] body_summary: vert_rms / horiz_rms, vert_p2p / horiz_p2p (m/s²)
- [GYRO] energy_frac_xyz (distribution of ∑ω²; X=roll/arm-swing, Y=pitch, Z=yaw/torso-twist)
→ Conclusion: is vertical (Vert) dominant? Is roll dominant (high X) / pitch-biased (high Y) / low yaw (low Z)?

• Category 3 **Rhythm characteristics (final step frequency & stability)**
- [ACC] peaks_summary: ibi_mean_s / ibi_std_s / ibi_cv; peaks_freq_spm = 60 / ibi_mean_s
- [ACC] freq_summary: dom_freq_hz / dom_freq_spm, harmonic_2x_ratio, low_high_energy_ratio
- [GYRO] spectral: step_freq_est_hz (≈ peak_rate/2), welch_wmag.top_freqs (check for integer harmonic structure)
- (Conflict resolution: if [ACC] dom_freq_spm and peaks_freq_spm disagree, follow established rules – prefer peaks or 2×dom_freq_spm)
→ Conclusion: final step frequency (Hz / spm) and stability (IBI CV)? Is spectrum harmonically regular?

• Category 4 **Waveform shape (impact-like vs smooth; rising/falling symmetry)**
- [ACC] jerk_summary (rms / p95 / vert_rms / vert_p95) + qualitative interpretation of peak sharpness
- [ACC] or [GYRO] (after bandpass) describe symmetry of rise vs fall (e.g., half-height width ratio)
→ Conclusion: more impact-type or smooth transition? Symmetric or asymmetric?

• Category 5 **Postural drift / slow orientation bias (only if relevant)**
- [GYRO] time_stats.mean_xyz (rad/s) and energy.net_angle_xyz_rad / abs_angle_xyz_rad (rad)
→ Conclusion: is there continuous forward-tilt / inward-rotation (e.g., mean_y<0, Δθ_y<0) or gradual drift instead of isolated wrist turns?

• Category 6 **Cross-sensor synchronization (decoupling or not)**
- [SYNC] step_rate_acc_hz vs step_rate_gyro_hz (Hz) and difference
- [SYNC] median_abs_lag_s, mean_lag_s (s)
- [SYNC] phase_consistency_0to1 (0–1)
→ Conclusion: are step rates consistent but phase misaligned (decoupling, common in stair ascent using handrail)? Or well synchronized (typical in flat walking)?

• Category 7 **Vertical impact (cross-sensor: accelerometer vertical component)**
- [SYNC]/[ACC] vertical_impact: impact_median_mps2, impact_p95_mps2 (m/s²), and per-step dispersion
→ Conclusion: impact strength is “high and scattered” or “moderate and stable” or others

────────────────────────────────
【Output template】
We provide an example of a final output (STRICTLY reuse this structure):
*Pattern Summary
- Strength:
- Axis dominance:
- Rhythm:
- Shape:
- Posture / drift:
- Gyro–Acc sync:
- Vertical impact:

"""

  return prompt



if __name__ == "__main__":
    
    data = np.load("uci/data_20_120.npy")
    acc  = data[0, :, :3].astype(float)   
    gyro = data[0, :, 3:6].astype(float)   
    dataset_name = "uci"
    prompt = generate_promt(acc, gyro, dataset_name, '')
    print(prompt)

    
