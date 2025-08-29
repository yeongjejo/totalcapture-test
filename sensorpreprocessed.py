import csv
import numpy as np
from typing import List, Dict

import socket, struct, time
import json



def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_inverse(q):
    w, x, y, z = q
    norm2 = np.dot(q,q)
    return np.array([w, -x, -y, -z], dtype=float) / norm2


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


if __name__ == "__main__":
    TARGET_IP = "127.0.0.1"
    TARGET_PORT = 5005

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    file_path = "output_quat_accel.csv"  # 저장된 csv 경로
    frames = parse_quat_accel_csv(file_path)

    bone_seq = ["Hips", "Spine3", "Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]

    first_q = []
    for idx, f in enumerate(frames):
        time.sleep(0.016)
        send_data = []
        # print(f["RightHand"]["quat"][0] == "NaN")
        for i, bone in enumerate(bone_seq):
            q = f[bone]["quat"].tolist()
            q = np.array([q[0], q[1], q[2], q[3]])
            if bone in ["LeftHand", "RightHand"]:
                q = np.array([1.0, 0.0, 0.0, 0.0])

            if idx == 0:
                first_q.append(quat_inverse(q))

            frame_bone_data = {
                "time": "1",
                "name": "test",
                "position": [0.0, 0.0, 0.0],
                "rotation": quat_mul(q, first_q[i]).tolist(),
                # "rotation": [bone[1].w, -bone[1].x, -bone[1].z, bone[1].y],
                "acc": [0.0, 0.0, 0.0]
            }
            send_data.append(frame_bone_data)
        print(send_data)
        data = json.dumps(send_data).encode("utf-8")
        sock.sendto(data, (TARGET_IP, TARGET_PORT))

    print("총 프레임 수:", len(frames))
    print("첫 번째 프레임의 Hips 데이터:")
    print("Quat:", frames[0]["Hips"]["quat"])
    print("Accel:", frames[0]["Hips"]["accel"])
