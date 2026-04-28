import torch
from PIL import Image
import math
import os
import argparse
import numpy as np
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict, List, Optional

# 导入 WinT3R 核心组件
from dust3r.wint3r import WinT3R, load_model
from layers.pose_enc import pose_encoding_to_extri
from layers.geometry import compute_relative_poses

def save_npz_async(frame_idx: int, data: Dict, save_dir: str):
    """异步将单帧结果保存为 npz"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.npz")
    # 转换为 numpy 格式
    np_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            np_data[k] = v.detach().cpu().numpy()
        else:
            np_data[k] = v
    np.savez_compressed(save_path, **np_data)

class ImageLoaderThread(threading.Thread):
    """生产者：多线程加载和预处理图像"""
    def __init__(self, data_path, load_queue, interval=1, img_size=512):
        super().__init__()
        self.data_path = data_path
        self.load_queue = load_queue
        self.interval = interval
        self.img_size = img_size
        self.stopped = False
        self.daemon = True

    def run(self):
        from dust3r.utils.image import _load_from_dir, _load_from_video, img_unnormalized
        
        # 获取文件列表
        if os.path.isdir(self.data_path):
            images_list = sorted(_load_from_dir(self.data_path))
        elif os.path.isfile(self.data_path):
            images_list = _load_from_video(self.data_path, self.interval)
        else:
            print(f"Error: 路径 {self.data_path} 不存在")
            return

        for i, img_path in enumerate(images_list):
            if self.stopped: break
            try:
                # 加载并 Resize 到 patch_size(16) 的倍数
                img = Image.open(img_path).convert('RGB')
                W, H = img.size
                scale = self.img_size / max(W, H)
                new_W, new_H = int(W * scale), int(H * scale)
                new_W, new_H = (new_W // 16) * 16, (new_H // 16) * 16
                img = img.resize((new_W, new_H), resample=Image.LANCZOS)
                
                # 预处理为 Tensor
                img_tensor = img_unnormalized(img)[None] # (1, 3, H, W)
                true_shape = torch.tensor([[new_H, new_W]])
                
                item = {
                    'img': img_tensor,
                    'true_shape': true_shape,
                    'idx': i,
                    'path': img_path
                }
                # 放入队列，若队列满则阻塞，实现背压（Backpressure）控制内存
                self.load_queue.put(item, block=True)
            except Exception as e:
                print(f"Warning: 加载帧 {i} 失败: {e}")
                continue
        
        # 放入结束标记
        self.load_queue.put(None)

def main():
    parser = argparse.ArgumentParser(description="WinT3R 高性能流式推理管道")
    parser.add_argument("--data_path", type=str, default='examples/001', help="输入图像目录或视频文件")
    parser.add_argument("--save_dir", type=str, default='output/recon_stream', help="结果保存目录")
    parser.add_argument("--ckpt", type=str, default='checkpoints/pytorch_model.bin', help="模型权重路径")
    parser.add_argument("--device", type=str, default='cuda', help="运行设备 (cuda/cpu)")
    parser.add_argument("--window_size", type=int, default=16, help="窗口大小")
    parser.add_argument("--interval", type=int, default=1, help="视频采样间隔 (60km/h建议为1)")
    
    # 高级性能参数
    parser.add_argument("--load_queue_size", type=int, default=512, help="加载队列长度，控制内存占用")
    parser.add_argument("--save_threads", type=int, default=max(1, os.cpu_count()-2), help="保存线程数")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    local_frame_dir = os.path.join(args.save_dir, "local_frames")
    os.makedirs(local_frame_dir, exist_ok=True)

    # 1. 加载模型
    print(f"正在加载模型: {args.ckpt}")
    model = load_model(args.ckpt, args.device)
    model.eval()
    model.window_size = args.window_size
    stride = args.window_size // 2

    # 2. 启动加载线程 (生产者)
    load_queue = queue.Queue(maxsize=args.load_queue_size)
    loader = ImageLoaderThread(args.data_path, load_queue, args.interval, img_size=512)
    loader.start()

    # 3. 启动保存线程池 (消费者)
    save_executor = ThreadPoolExecutor(max_workers=args.save_threads)
    
    # 状态与记忆变量
    state_feat, state_pos = None, None
    camera_embed_list = []
    frame_buffer = {} # 暂存尚未确定最终结果的帧
    active_views = [] # 当前推理窗口内的帧
    
    processed_count = 0 # 已滑出窗口并确认的帧数
    total_loaded = 0
    start_time = time.time()

    print(f"开始推理管道... 结果将实时存入 {local_frame_dir}")

    try:
        while True:
            # 从加载队列获取数据
            item = load_queue.get()
            
            if item is None: # 信号：输入流已结束
                if not active_views: break
                
                # 末尾 Padding：模仿原版 online_inference 逻辑
                last_item = active_views[-1]
                padding_len = 0
                if len(active_views) < args.window_size:
                    padding_len = args.window_size - len(active_views)
                elif len(active_views) % stride != 0:
                    padding_len = stride - (len(active_views) % stride)
                
                if padding_len > 0:
                    active_views.extend([last_item] * padding_len)
                is_last_batch = True
            else:
                # 搬运到 GPU
                item['img'] = item['img'].to(args.device)
                item['true_shape'] = item['true_shape'].to(args.device)
                active_views.append(item)
                total_loaded += 1
                is_last_batch = False
            
            # 当累积够一个 window_size 时，触发推理
            if len(active_views) == args.window_size:
                with torch.no_grad():
                    # A. 视图编码
                    _, feat_ls, pos = model._encode_views(active_views)
                    feat = feat_ls[-1]
                    
                    # B. 初始化或接力记忆状态 (Recurrent State)
                    if state_feat is None:
                        bs = pos[0].shape[0]
                        state_feat, state_pos = model._init_state(feat[0], pos[0])
                        # 相机 Token 位置编码
                        camera_pos = torch.zeros(bs, args.window_size, 1, pos[0].shape[2]).to(state_pos)

                    # C. 递归解码过程 (The Core Math)
                    feat_i = torch.stack(feat, dim=1)
                    feat_i = model.decoder_embed(feat_i)
                    pos_i = torch.stack(pos, dim=1) + 1
                    cam_token = torch.stack([model.cam_token]*args.window_size, dim=1).expand(bs, -1, -1, -1)
                    
                    feat_i = torch.cat([cam_token, feat_i], dim=2)
                    pos_i = torch.cat([camera_pos, pos_i], dim=2)
                    
                    # rollout 会产生 new_state_feat，用于接力下一个窗口
                    new_state_feat, dec, f_img_local = model._recurrent_rollout(state_feat, state_pos, feat_i, pos_i)
                    state_feat = new_state_feat # 更新记忆
                    
                    # D. 提取头部的输出结果
                    B, S, P, C = dec[-1].shape
                    dec_final = dec[-1].reshape(B, S, P, C)
                    f_img_local_final = f_img_local.reshape(B, S, P, C)
                    
                    # 相机特征池 (用于后期全局位姿优化)
                    camera_token = torch.cat([dec_final[:, :, 0], f_img_local_final[:, :, 0]], dim=-1)
                    camera_embed_list.append(camera_token)
                    
                    # 获取局部点云和置信度 (PtsHead)
                    head_input = [
                        f_img_local_final[:, :, 1:].float().reshape(-1, P-1, C),
                        dec_final[:, :, 1:].float().reshape(-1, P-1, C)
                    ]
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        pts_res = model.pts_head(head_input, active_views[0]["img"])
                    
                    current_pts = pts_res['pts_local'].reshape(B, S, *pts_res['pts_local'].shape[1:])
                    current_conf = pts_res['conf'].reshape(B, S, *pts_res['conf'].shape[1:])
                    
                    # E. 置信度竞争与缓存更新
                    # 每一窗口的 stride 为 window_size // 2，意味着每帧至少参与 2 个窗口
                    window_start_idx = (processed_count // stride) * stride
                    for j in range(args.window_size):
                        view = active_views[j]
                        f_idx = view['idx']
                        
                        this_conf = current_conf[0, j].cpu()
                        this_pts = current_pts[0, j].cpu()
                        this_conf_sum = this_conf.sum().item()
                        
                        # 在全局位姿池中的相对位置
                        global_cam_idx = (processed_count // stride) * args.window_size + j
                        
                        # 如果是新帧，或当前窗口给出的结果置信度更高，则更新
                        if f_idx not in frame_buffer or this_conf_sum > frame_buffer[f_idx]['conf_sum']:
                            frame_buffer[f_idx] = {
                                'pts_local': this_pts,
                                'conf': this_conf,
                                'conf_sum': this_conf_sum,
                                'cam_idx': global_cam_idx
                            }
                
                # F. 窗口滑动：确认并保存已经“过时”的帧
                # 一旦帧离开 active_views，且后续窗口不再包含它，就可以安全落盘
                if not is_last_batch:
                    # 步长为 stride，所以前 stride 个结果可以确定了
                    safe_to_save_range = range(processed_count, processed_count + stride)
                    for f_idx_to_save in safe_to_save_range:
                        if f_idx_to_save in frame_buffer:
                            data = frame_buffer.pop(f_idx_to_save)
                            save_executor.submit(save_npz_async, f_idx_to_save, data, local_frame_dir)
                    
                    # 真正滑出窗口
                    for _ in range(stride):
                        active_views.pop(0)
                    processed_count += stride
                
                # 定期维护
                if total_loaded % 100 == 0:
                    torch.cuda.empty_cache()
                    elapsed = time.time() - start_time
                    print(f"已加载: {total_loaded} 帧 | 已确认: {processed_count} 帧 | 速度: {total_loaded/elapsed:.2f} FPS")
                
                if is_last_batch: break

        # 结束处理：保存最后残留在 buffer 中的帧
        for f_idx in list(frame_buffer.keys()):
            data = frame_buffer.pop(f_idx)
            save_executor.submit(save_npz_async, f_idx, data, local_frame_dir)

        print("--- 步骤 1：局部推理完成。正在计算全局位姿... ---")
        
        # 4. 全局位姿计算 (Global Pose Refinement)
        with torch.no_grad():
            full_camera_tokens = torch.cat(camera_embed_list, dim=1).float()
            # 30,000 个 tokens 在 4090 上计算很快
            all_poses_enc = model.cam_head(full_camera_tokens)
            # 转换为第一帧坐标系
            rel_poses_enc = [compute_relative_poses(p) for p in all_poses_enc]
            # 提取 4x4 矩阵
            final_extrinsics = pose_encoding_to_extri(rel_poses_enc[-1])[0].cpu().numpy()

        # 保存全局位姿文件
        pose_save_path = os.path.join(args.save_dir, "global_poses.npz")
        np.savez_compressed(pose_save_path, poses=final_extrinsics)
        
        print(f"--- 步骤 2：全局位姿已保存至 {pose_save_path} ---")
        print(f"总处理时长: {time.time()-start_time:.2f} 秒。")
        print("提示：您可以使用 local_frames/ 中的局部点云结合 global_poses.npz 还原出完整世界坐标点云。")

    except KeyboardInterrupt:
        print("\n用户中断。正在安全退出并保存当前结果...")
    finally:
        loader.stopped = True
        save_executor.shutdown(wait=True)

if __name__ == '__main__':
    main()
