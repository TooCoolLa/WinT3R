import os
import glob
import numpy as np
import cv2
import viser
import viser.transforms as tf
import argparse
import time
import threading
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="WinT3R 推理结果交互式流式展示")
    parser.add_argument("--save_dir", "-d", type=str, required=True, help="recon.py 的 --save_dir 路径")
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="原始图像路径")
    parser.add_argument("--max_points", type=int, default=5000, help="每帧显示的最大点数")
    parser.add_argument("--conf_thresh", type=float, default=0.01, help="置信度阈值")
    args = parser.parse_args()

    # 数据准备
    local_frame_dir = os.path.join(args.save_dir, "local_frames")
    pose_path = os.path.join(args.save_dir, "global_poses.npz")
    
    if not os.path.exists(pose_path) or not os.path.exists(local_frame_dir):
        print("错误: 找不到结果文件。请确认推理已完成。")
        return

    poses = np.load(pose_path)['poses']
    npz_files = sorted(glob.glob(os.path.join(local_frame_dir, "*.npz")))

    # 启动 Viser
    server = viser.ViserServer()
    server.scene.set_up_direction("-y")

    # GUI 元素
    gui_info = server.gui.add_text("Status", initial_value="Waiting for client...", disabled=True)
    gui_progress = server.gui.add_text("Progress", initial_value="0/0", disabled=True)
    btn_start = server.gui.add_button("Start Streaming")
    btn_reset = server.gui.add_button("Clear Scene")
    
    with server.gui.add_folder("Filter Settings"):
        conf_slider = server.gui.add_slider("Conf Thresh", min=0.0, max=100.0, step=0.1, initial_value=5.0)
        depth_slider = server.gui.add_slider("Max Depth", min=1.0, max=100.0, step=1.0, initial_value=20.0)
        speed_slider = server.gui.add_slider("Streaming Speed (s)", min=0.0, max=1.0, step=0.05, initial_value=0.05)
    
    with server.gui.add_folder("Visual Settings"):
        point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005)
        point_shape = server.gui.add_dropdown("Point Shape", options=["square", "circle", "sparkle"], initial_value="circle")

    state = {"is_streaming": False, "stop_requested": False}

    def load_image(full_name):
        for ext in ['.jpg', '.png', '.jpeg', '.webp']:
            p = os.path.join(args.image_dir, full_name + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def streaming_task():
        state["is_streaming"] = True
        state["stop_requested"] = False
        gui_info.value = "Streaming..."
        
        for i, p in enumerate(npz_files):
            if state["stop_requested"]: break
            
            try:
                f_name = os.path.basename(p)
                f_idx = int(f_name.split('_')[1].split('.')[0])
                data = np.load(p)
                pts_local, conf, cam_idx = data['pts_local'], data['conf'], int(data['cam_idx'])
                
                if cam_idx >= len(poses): continue

                # 矩阵处理
                w2c = poses[cam_idx]
                if w2c.shape == (3, 4):
                    w2c_tmp = np.eye(4); w2c_tmp[:3, :4] = w2c; w2c = w2c_tmp
                c2w = np.linalg.inv(w2c)

                # 点云过滤逻辑
                # 1. 置信度过滤
                mask = conf > conf_slider.value
                # 2. 局部深度过滤 (通常 z 轴代表深度)
                mask &= (pts_local[..., 2] < depth_slider.value) 
                
                sel_pts = pts_local[mask]
                if len(sel_pts) == 0: continue
                
                img = load_image(f_name.replace(".npz", ""))
                if img is not None:
                    sel_cols = cv2.resize(img, (pts_local.shape[1], pts_local.shape[0]))[mask]
                else:
                    sel_cols = np.ones_like(sel_pts) * 255

                if len(sel_pts) > args.max_points:
                    idx = np.random.choice(len(sel_pts), args.max_points, replace=False)
                    sel_pts, sel_cols = sel_pts[idx], sel_cols[idx]

                pts_world = (c2w[:3, :3] @ sel_pts.T).T + c2w[:3, 3]

                # 更新场景
                server.scene.add_camera_frustum(
                    f"/cameras/frame_{f_idx}",
                    fov=2 * np.arctan(320 / (2 * 500)),
                    aspect=1.0, scale=0.05,
                    wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                    position=c2w[:3, 3]
                )
                server.scene.add_point_cloud(
                    f"/points/frame_{f_idx}",
                    points=pts_world, colors=sel_cols,
                    point_size=point_size_slider.value,
                    point_shape=point_shape.value
                )

                gui_progress.value = f"{i+1}/{len(npz_files)}"
                time.sleep(speed_slider.value)

            except Exception as e:
                print(f"Error at {f_idx}: {e}")

        gui_info.value = "Finished"
        state["is_streaming"] = False

    @btn_start.on_click
    def _(_):
        if not state["is_streaming"]:
            threading.Thread(target=streaming_task, daemon=True).start()

    @btn_reset.on_click
    def _(_):
        state["stop_requested"] = True
        time.sleep(0.2)
        server.scene.reset()
        gui_progress.value = "0/0"
        gui_info.value = "Scene Cleared"

    print(f"Viser server running at http://localhost:8080")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
