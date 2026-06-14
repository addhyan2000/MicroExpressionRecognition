import os
import json
import pandas as pd
from pathlib import Path

results_dir = Path("c:/Users/Addhyan/Desktop/Thesis/Thesis3/Ablation_Study/results")
configs = [
    "config_1_pure_base__no_evm__no_simam__no_3dcnn__no_transformer",
    "config_2_temporal_only__no_evm__no_simam__no_3dcnn__WITH_transformer",
    "config_3_spatial_only__no_evm__no_simam__WITH_3dcnn__no_transformer",
    "config_4_motion_amp_base__WITH_evm__no_simam__no_3dcnn__no_transformer",
    "config_5_attention_base__no_evm__WITH_simam__WITH_3dcnn__no_transformer",
    "config_6_full_stage2_noevm__no_evm__WITH_simam__WITH_3dcnn__WITH_transformer",
    "config_7_full_no_attention__WITH_evm__no_simam__WITH_3dcnn__WITH_transformer",
    "config_8_proposed_unified__WITH_evm__WITH_simam__WITH_3dcnn__WITH_transformer",
    "config_9_permutation__no_evm__no_simam__WITH_3dcnn__WITH_transformer",
    "config_12_permutation__WITH_evm__no_simam__no_3dcnn__WITH_transformer",
    "config_13_permutation__WITH_evm__no_simam__WITH_3dcnn__no_transformer",
    "config_16_permutation__WITH_evm__WITH_simam__WITH_3dcnn__no_transformer"
]

rows = []
for c in configs:
    json_path = results_dir / c / "final_results.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        row = {
            "name": data["config_name"],
            "phase": data["extra"]["phase"],
            "EVM": data["toggles"]["use_evm"],
            "SimAM": data["toggles"]["use_simam"],
            "CNN3D": data["toggles"]["use_cnn"],
            "SLSTT": data["toggles"]["use_transformer"],
            "accuracy": data["metrics"]["accuracy"],
            "macro_f1": data["metrics"]["macro_f1"],
            "Negative_F1": data["metrics"]["per_class_f1"][0],
            "Positive_F1": data["metrics"]["per_class_f1"][1],
            "Surprise_F1": data["metrics"]["per_class_f1"][2]
        }
        rows.append(row)
    else:
        print(f"Missing: {json_path}")

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("ablation_clean_summary.csv", index=False)
