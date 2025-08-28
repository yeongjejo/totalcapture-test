import csv
import numpy as np
from typing import List, Dict

def parse_quat_accel_csv(file_path: str):
    """
    output_quat_accel.csv 형식 파싱
    반환: frames 리스트
      frames[i][joint] = {
          'quat': np.array([x,y,z,w]),
          'accel': np.array([ax,ay,az])
      }
    """
    frames: List[Dict[str, Dict[str, np.ndarray]]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # DictReader는 각 셀을 "헤더: 문자열"로 줌
        for row in reader:
            frame: Dict[str, Dict[str, np.ndarray]] = {}
            for col in row:
                if row[col].strip() == "":
                    continue
                if "_quat" in col:
                    joint = col.replace("_quat", "")
                    quat_vals = [float(x) for x in row[col].split()]
                    if len(quat_vals) == 4:
                        frame.setdefault(joint, {})["quat"] = np.array(quat_vals, dtype=float)
                elif "_accel" in col:
                    joint = col.replace("_accel", "")
                    accel_vals = [float(x) for x in row[col].split()]
                    if len(accel_vals) == 3:
                        frame.setdefault(joint, {})["accel"] = np.array(accel_vals, dtype=float)
            frames.append(frame)

    return frames

# ===================== 사용 예시 =====================
if __name__ == "__main__":
    file_path = "output_quat_accel.csv"  # 저장된 csv 경로
    frames = parse_quat_accel_csv(file_path)
    for f in frames:
        print(f.keys())

    print("총 프레임 수:", len(frames))
    print("첫 번째 프레임의 Hips 데이터:")
    print("Quat:", frames[0]["Hips"]["quat"])
    print("Accel:", frames[0]["Hips"]["accel"])
