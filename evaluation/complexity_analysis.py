"""
模型复杂度分析
计算参数量、FLOPs、推理速度等
"""
import os
import sys
import argparse
import yaml
import torch
import time
import numpy as np
from thop import profile, clever_format

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_vit_base, create_vit_small, DynamicViT, LightweightViT


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def measure_inference_time(model, input_size, device, num_runs=100, warmup=10):
    """测量推理时间"""
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 同步CUDA（如果使用GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


def compute_flops(model, input_size, device):
    """计算FLOPs"""
    dummy_input = torch.randn(1, *input_size).to(device)
    
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    return flops, params


def analyze_model(model_type, config, device):
    """分析单个模型"""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_type}")
    print(f"{'='*60}")
    
    # 创建模型
    img_size = config.get('img_size', 32)
    
    if model_type == 'vit_base':
        model = create_vit_base(num_classes=10, img_size=img_size)
    elif model_type == 'vit_small':
        model = create_vit_small(num_classes=10, img_size=img_size)
    elif model_type == 'dynamic_vit':
        model = DynamicViT(
            img_size=img_size,
            num_classes=10,
            embed_dim=config.get('embed_dim', 384),
            depth=config.get('depth', 6),
            num_heads=config.get('num_heads', 6)
        )
    elif model_type == 'lightweight_vit':
        model = LightweightViT(
            img_size=img_size,
            num_classes=10,
            embed_dim=config.get('embed_dim', 384),
            depth=config.get('depth', 6)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    # 参数量
    total_params, trainable_params = count_parameters(model)
    
    # FLOPs
    input_size = (3, img_size, img_size)
    try:
        flops, params_str = compute_flops(model, input_size, device)
    except:
        flops = "N/A"
        params_str = "N/A"
    
    # 推理时间
    avg_time, std_time = measure_inference_time(model, input_size, device)
    throughput = 1.0 / avg_time
    
    # 模型大小（MB）
    param_size = total_params * 4 / (1024 ** 2)  # 假设float32
    
    results = {
        'model_type': model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'param_size_mb': param_size,
        'avg_inference_time_ms': avg_time * 1000,
        'std_inference_time_ms': std_time * 1000,
        'throughput_fps': throughput
    }
    
    # 打印结果
    print(f"\nModel: {model_type}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"FLOPs: {flops}")
    print(f"Model Size: {param_size:.2f} MB")
    print(f"Inference Time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")
    
    return results


def main(args):
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 分析的模型列表
    models_to_analyze = args.models if args.models else ['vit_base', 'vit_small', 'dynamic_vit', 'lightweight_vit']
    
    # 分析每个模型
    all_results = []
    for model_type in models_to_analyze:
        try:
            results = analyze_model(model_type, config, device)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {model_type}: {e}")
    
    # 生成对比表格
    print("\n" + "="*80)
    print("MODEL COMPLEXITY COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Params (M)':<15} {'FLOPs':<15} {'Size (MB)':<12} {'Time (ms)':<15} {'FPS':<10}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['model_type']:<20} "
              f"{result['total_params']/1e6:<15.2f} "
              f"{result['flops']:<15} "
              f"{result['param_size_mb']:<12.2f} "
              f"{result['avg_inference_time_ms']:<15.2f} "
              f"{result['throughput_fps']:<10.2f}")
    
    # 保存结果
    save_dir = config.get('save_dir', './experiments/results')
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, 'complexity_analysis.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPLEXITY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Device: {device}\n\n")
        
        for result in all_results:
            f.write(f"\nModel: {result['model_type']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Parameters: {result['total_params']:,} ({result['total_params']/1e6:.2f}M)\n")
            f.write(f"Trainable Parameters: {result['trainable_params']:,}\n")
            f.write(f"FLOPs: {result['flops']}\n")
            f.write(f"Model Size: {result['param_size_mb']:.2f} MB\n")
            f.write(f"Inference Time: {result['avg_inference_time_ms']:.2f} ± {result['std_inference_time_ms']:.2f} ms\n")
            f.write(f"Throughput: {result['throughput_fps']:.2f} FPS\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Model Complexity')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='List of models to analyze (default: all)'
    )
    
    args = parser.parse_args()
    main(args)