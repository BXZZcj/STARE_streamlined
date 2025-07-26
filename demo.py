import sys
import os
from pathlib import Path
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import time


rvt_project_root = Path("/mnt/f1590153-780c-408d-b394-7b3b56082548/ESOT500/RVT/RVT")
if rvt_project_root.exists() and str(rvt_project_root) not in sys.path:
    print(f"INFO: Adding RVT project root to sys.path: {rvt_project_root}")
    sys.path.insert(0, str(rvt_project_root))

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_model_module
from models.detection.yolox.utils.boxes import postprocess

def main():
    # ------------------------------------------------------------------------ #
    # 1. 加载配置和模型 (与之前相同)
    # ------------------------------------------------------------------------ #
    root_path = Path('/mnt/f1590153-780c-408d-b394-7b3b56082548/ESOT500/RVT/RVT')
    config_path = root_path / 'config'
    checkpoint_path = root_path / '../checkpoints/rvt-t.ckpt' 

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    overrides = [
        'dataset=gen1',
        f'checkpoint={str(checkpoint_path)}',
        '+experiment/gen1=tiny',
    ]
    
    print("INFO: 正在加载配置...")
    with hydra.initialize_config_dir(config_dir=str(config_path.absolute()), version_base='1.2'):
        config = hydra.compose(config_name='val', overrides=overrides)

    esot500_resolution_hw = (240, 304)    
    print(f"\nINFO: 尝试将分辨率修改为: {esot500_resolution_hw}")

    # 【步骤 1】: 手动更新 config 中的 data_loading_hw
    with open_dict(config.dataset): # 使用 open_dict 来允许修改
        config.dataset.data_loading_hw = list(esot500_resolution_hw)

    # 【步骤 2】: 再次调用动态配置函数，让它根据新分辨率调整所有模型参数
    print("INFO: 重新运行动态配置函数以适配新分辨率...")
    dynamically_modify_train_config(config)
    
    final_model_resolution = tuple(config.model.backbone.in_res_hw)
    print(f"INFO: 模型已成功适配！最终有效输入分辨率为: {final_model_resolution}")
    
    # ======================================================================== #

    print("INFO: 正在加载模型...")
    model_class = fetch_model_module(config=config)
    pl_module = model_class.load_from_checkpoint(str(checkpoint_path), **{'full_config': config})
    
    pl_module.eval()
    pl_module.to(device)
    print(f"INFO: 模型已成功加载到 {device}")
    
    # ------------------------------------------------------------------------ #
    # 2. 根据“新的”分辨率创建随机输入
    # ------------------------------------------------------------------------ #
    print("\nINFO: 正在根据新分辨率创建随机输入序列...")

    sequence_length = 100
    channels = 10 * 2
    # 【关键】从最终的、经过适配的配置中读取 H 和 W
    height, width = final_model_resolution

    sequence_shape = (sequence_length, channels, height, width)
    print(f"INFO: 输入序列形状 (S, C, H, W): {sequence_shape}")
    dummy_sequence = torch.randn(sequence_shape, device=device)
    
    # ------------------------------------------------------------------------ #
    # 3. 模拟 _val_test_step_impl 的核心循环逻辑
    # ------------------------------------------------------------------------ #
    
    hidden_state = None
    
    print("\nINFO: 正在以循环方式执行模型的前向传播 (模拟真实推理)...")
    start_time = time.time()
    with torch.inference_mode():
        for t in range(sequence_length):
            current_frame = dummy_sequence[t].unsqueeze(0)
            
            # 调用 pl_module 的 forward, 它会代理调用 self.mdl.forward
            predictions, _, hidden_state = pl_module(current_frame, hidden_state)
            
            # pl_module.mdl.forward 返回的 predictions 是一个列表，每个元素对应batch中的一个样本
            # postprocess 函数也期望接收这个列表
            pred_processed = postprocess(
                prediction=predictions,
                num_classes=config.model.head.num_classes,
                conf_thre=config.model.postprocess.confidence_threshold,
                nms_thre=config.model.postprocess.nms_threshold
            )
            # postprocess 的输出也是一个列表，我们取出我们这个batch中唯一一个样本的结果
            final_dets = pred_processed[0]

            print(f"--- Timestep {t+1}/{sequence_length} ---")
            if final_dets is not None and final_dets.shape[0] > 0:
                print(f"  检测到 {final_dets.shape[0]} 个目标。")
                det = final_dets[0]
                box = det[:4].cpu().numpy().astype(int)
                conf = det[4].item()
                class_idx = int(det[5].item())
                print(f"  示例目标 -> Class: {class_idx}, Conf: {conf:.3f}, Box: {box.tolist()}")
            else:
                print("  未检测到目标。")

    end_time = time.time()
    print("\nINFO: 循环前向传播完成！")
    print(f"INFO: 处理 {sequence_length} 帧耗时: {end_time - start_time:.4f} 秒")
    
if __name__ == '__main__':
    main()