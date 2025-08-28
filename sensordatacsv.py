# import re
# import csv
# import numpy as np
# from typing import List, Dict, Tuple
#
# # ===== IMU → Bone 매핑 =====
# IMU_TO_BONE = {
#     "Head": "Head",
#     "Sternum": "Spine3",
#     "Pelvis": "Hips",
#     "L_UpArm": "LeftArm",
#     "R_UpArm": "RightArm",
#     "L_LowArm": "LeftForeArm",
#     "R_LowArm": "RightForeArm",
#     "L_UpLeg": "LeftUpLeg",
#     "R_UpLeg": "RightUpLeg",
#     "L_LowLeg": "LeftLeg",
#     "R_LowLeg": "RightLeg",
#     "L_Foot": "LeftFoot",
#     "R_Foot": "RightFoot",
# }
#
# # ===== 출력할 관절 순서 (예시 헤더) =====
# TARGET_JOINTS = [
#     "Hips","Spine","Spine1","Spine2","Spine3","Neck","Head",
#     "RightShoulder","RightArm","RightForeArm","RightHand",
#     "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
#     "RightUpLeg","RightLeg","RightFoot","LeftUpLeg","LeftLeg","LeftFoot"
# ]
#
# FRAME_MARKER_RE = re.compile(r"^\s*\d+\s*$")
# HEADER_TWO_INTS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
#
# def _maybe_two_int_header(line: str) -> bool:
#     return HEADER_TWO_INTS_RE.match(line.replace("\t"," ")) is not None
#
# def _is_frame_marker(line: str) -> bool:
#     return FRAME_MARKER_RE.match(line) is not None
#
# def _parse_sensor_line(line: str) -> Tuple[str, np.ndarray]:
#     parts = line.strip().split()
#     if len(parts) < 14:
#         raise ValueError(f"센서 라인 형식 오류: {line[:80]}")
#     name = parts[0]
#     vals = np.array([float(x) for x in parts[1:]], dtype=float)
#     return name, vals
#
# def load_sensors_quat_only(file_path: str, encoding="utf-8") -> List[Dict[str,np.ndarray]]:
#     """ .sensors 파일을 읽어 프레임별 quaternion(x,y,z,w) dict 반환 """
#     with open(file_path,"r",encoding=encoding) as f:
#         lines = [ln.strip() for ln in f if ln.strip()]
#
#     if lines and _maybe_two_int_header(lines[0]):
#         lines = lines[1:]
#
#     frames = []
#     cur_rows = []
#
#     def flush():
#         nonlocal cur_rows, frames
#         if not cur_rows: return
#         bones = {}
#         for sname, vals in cur_rows:
#             qw,qx,qy,qz = vals[0:4]
#             bone = IMU_TO_BONE.get(sname)
#             if bone:
#                 bones[bone] = np.array([qx,qy,qz,qw])
#         frames.append(bones)
#         cur_rows = []
#
#     for line in lines:
#         if _is_frame_marker(line):
#             flush()
#             continue
#         sname, vals = _parse_sensor_line(line)
#         cur_rows.append((sname, vals))
#     flush()
#     return frames
#
# def save_frames_to_csv(frames: List[Dict[str,np.ndarray]], out_path: str):
#     """ CSV 파일로 저장 (열=조인트, 각 셀="x y z w") """
#     with open(out_path,"w",newline="",encoding="utf-8") as f:
#         writer = csv.writer(f)
#         # 헤더
#         writer.writerow(TARGET_JOINTS)
#         # 데이터
#         nan4 = "NaN NaN NaN NaN"
#         for fr in frames:
#             row = []
#             for j in TARGET_JOINTS:
#                 if j in fr:
#                     q = fr[j]
#                     cell = f"{q[0]} {q[1]} {q[2]} {q[3]}"
#                 else:
#                     cell = nan4
#                 row.append(cell)
#             writer.writerow(row)
#
# # ================= 사용 예시 =================
# if __name__ == "__main__":
#
#     in_path = r"C:\Users\Ipop_Dev\Downloads\totalcapture\imu\s5\acting3_Xsens_AuxFields.sensors"
#     out_path = "output_quat.csv" # csv 저장 경로
#
#     frames = load_sensors_quat_only(in_path)
#     save_frames_to_csv(frames, out_path)
#     print("CSV 저장 완료:", out_path)




import re
import csv
import numpy as np
from typing import List, Dict, Tuple

# ===== IMU → Bone 매핑 =====
IMU_TO_BONE = {
    "Head": "Head",
    "Sternum": "Spine3",
    "Pelvis": "Hips",
    "L_UpArm": "LeftArm",
    "R_UpArm": "RightArm",
    "L_LowArm": "LeftForeArm",
    "R_LowArm": "RightForeArm",
    "L_UpLeg": "LeftUpLeg",
    "R_UpLeg": "RightUpLeg",
    "L_LowLeg": "LeftLeg",
    "R_LowLeg": "RightLeg",
    "L_Foot": "LeftFoot",
    "R_Foot": "RightFoot",
}

# ===== 출력 관절 순서(예시와 동일) =====
TARGET_JOINTS = [
    "Hips","Spine","Spine1","Spine2","Spine3","Neck","Head",
    "RightShoulder","RightArm","RightForeArm","RightHand",
    "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
    "RightUpLeg","RightLeg","RightFoot","LeftUpLeg","LeftLeg","LeftFoot"
]

FRAME_MARKER_RE = re.compile(r"^\s*\d+\s*$")
HEADER_TWO_INTS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

def _maybe_two_int_header(line: str) -> bool:
    return HEADER_TWO_INTS_RE.match(line.replace("\t"," ")) is not None

def _is_frame_marker(line: str) -> bool:
    return FRAME_MARKER_RE.match(line) is not None

def _parse_sensor_line(line: str) -> Tuple[str, np.ndarray]:
    # vals = [qw,qx,qy,qz, ax,ay,az, gx,gy,gz, mx,my,mz]
    parts = line.strip().split()
    if len(parts) < 14:
        raise ValueError(f"센서 라인 형식 오류: {line[:100]}")
    name = parts[0]
    vals = np.array([float(x) for x in parts[1:]], dtype=float)
    if vals.size != 13:
        raise ValueError(f"수치 개수는 13이어야 합니다. (현재 {vals.size})")
    return name, vals

def load_sensors_quat_accel(file_path: str, encoding="utf-8") -> List[Dict[str, Dict[str, np.ndarray]]]:
    """
    .sensors 파일을 읽어 프레임 리스트 반환.
    각 프레임: { bone: { 'quat_xyzw': (4,), 'accel_xyz': (3,) } }
    - quaternion은 입력(WXYZ)에서 출력(XYZW)로 재배치.
    """
    with open(file_path, "r", encoding=encoding) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if lines and _maybe_two_int_header(lines[0]):
        lines = lines[1:]

    frames: List[Dict[str, Dict[str, np.ndarray]]] = []
    cur_rows: List[Tuple[str, np.ndarray]] = []

    def flush():
        nonlocal cur_rows, frames
        if not cur_rows:
            return
        frame: Dict[str, Dict[str, np.ndarray]] = {}
        for sname, vals in cur_rows:
            qw, qx, qy, qz = vals[0:4]
            ax, ay, az = vals[4:7]
            bone = IMU_TO_BONE.get(sname)
            if not bone:
                continue
            frame[bone] = {
                "quat_xyzw": np.array([qx, qy, qz, qw], dtype=float),
                "accel_xyz": np.array([ax, ay, az], dtype=float),
            }
        frames.append(frame)
        cur_rows = []

    for line in lines:
        if _is_frame_marker(line):
            flush()
            continue
        sname, vals = _parse_sensor_line(line)
        cur_rows.append((sname, vals))
    flush()

    if not frames:
        raise ValueError("프레임을 파싱하지 못했습니다. 파일 포맷을 확인하세요.")
    return frames

def save_frames_quat_accel_csv(frames: List[Dict[str, Dict[str, np.ndarray]]], out_path: str):
    """
    CSV 저장:
      열: 각 관절마다 '<Joint>_quat', '<Joint>_accel'
      값: quat은 'x y z w', accel은 'ax ay az' (공백으로 묶인 문자열)
    """
    # 헤더 구성
    header: List[str] = []
    for j in TARGET_JOINTS:
        header.append(f"{j}_quat")
        header.append(f"{j}_accel")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        nan_quat = "NaN NaN NaN NaN"
        nan_acc  = "NaN NaN NaN"

        for fr in frames:
            row: List[str] = []
            for j in TARGET_JOINTS:
                if j in fr:
                    q = fr[j]["quat_xyzw"]   # (x,y,z,w)
                    a = fr[j]["accel_xyz"]   # (ax,ay,az)
                    row.append(f"{q[0]} {q[1]} {q[2]} {q[3]}")
                    row.append(f"{a[0]} {a[1]} {a[2]}")
                else:
                    row.append(nan_quat)
                    row.append(nan_acc)
            writer.writerow(row)

# ================= 사용 예시 =================
if __name__ == "__main__":
    in_path = r"C:\Users\Ipop_Dev\Downloads\totalcapture\imu\s5\acting3_Xsens_AuxFields.sensors"     # 실제 .sensors 파일 경로
    out_path = "output_quat_accel.csv"

    frames = load_sensors_quat_accel(in_path)
    save_frames_quat_accel_csv(frames, out_path)
    print("CSV 저장 완료:", out_path)
