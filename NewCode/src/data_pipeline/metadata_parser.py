# -*- coding: utf-8 -*-
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any

class CASME2_MetadataParser:
    def __init__(self, excel_path: Path, dataset_root: Path):
        self.excel_path = Path(excel_path)
        self.dataset_root = Path(dataset_root)
        
    def get_micro_clips(self) -> List[Dict[str, Any]]:
        df = pd.read_excel(self.excel_path, header=None)
        clips = []

        for i, row in df.iterrows():
            try:
                # 1. Subject (Col 0)
                sub_id = int(row[0])
                subject = f"s{sub_id}"
                subject_path = self.dataset_root / subject
                
                if not subject_path.exists(): continue

                # 2. Extract digits from Excel Filename (Col 1)
                # Example: 'anger1_2' -> ['1', '2']
                excel_vid_name = str(row[1])
                digits = re.findall(r'\d+', excel_vid_name)
                if not digits: continue
                
                # In CAS(ME)2: First digit is Stimulus, Second is Clip
                stim_id = f"{int(digits[0]):02d}"
                clip_id = f"{int(digits[1]):02d}" if len(digits) > 1 else ""
                
                # 3. Fuzzy Search in Subject Folder
                # We look for ANY avi that contains the subject ID and Stimulus ID
                # Example: look for "15_0102" inside "s15"
                match_pattern = f"{sub_id}_{stim_id}{clip_id}"
                
                target_video = None
                for file in subject_path.glob("*.avi"):
                    if match_pattern in file.name:
                        target_video = file
                        break
                
                # Final fallback: just match by Subject_Stimulus (e.g. 15_01)
                if not target_video:
                    short_pattern = f"{sub_id}_{stim_id}"
                    for file in subject_path.glob("*.avi"):
                        if file.name.startswith(short_pattern):
                            target_video = file
                            break

                if not target_video: continue

                # 4. Metadata
                onset = int(row[2])
                offset = int(row[4])
                if offset <= onset: offset = onset + 45 # Window for missing offsets
                
                expr_type = str(row[7]).lower().replace(" ", "_")
                emotion = str(row[8]) if len(row) > 8 else "unknown"

                clips.append({
                    "video_id": f"{subject}_{excel_vid_name}_{expr_type}_{onset}_{offset}",
                    "path": target_video,
                    "onset": onset,
                    "offset": offset,
                    "label": emotion
                })
            except:
                continue
                
        print(f"--- Discovery Result: Found {len(clips)} / 102 potential videos ---")
        return clips