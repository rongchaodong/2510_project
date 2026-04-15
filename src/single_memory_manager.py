import torch
import torch.nn.functional as F
import time

class PagedAttentionEngineWithPreemption:
    def __init__(self, num_gpu_blocks, num_cpu_blocks, block_size, num_heads, head_dim, device="cuda"):
        self.device = device
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        print(f"[{device.upper()}] 分配 GPU KV Cache 池 ({num_gpu_blocks} 块)...")
        self.gpu_k_pool = torch.zeros((num_gpu_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
        self.gpu_v_pool = torch.zeros((num_gpu_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
        
        print(f"[CPU] 分配 CPU 备用内存池 ({num_cpu_blocks} 块) 用于 Swap...")
        # 建议在真实系统中使用 pinned memory (pin_memory=True) 以加速 PCIe 传输
        self.cpu_k_pool = torch.zeros((num_cpu_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device="cpu", pin_memory=True)
        self.cpu_v_pool = torch.zeros((num_cpu_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device="cpu", pin_memory=True)
        
        self.gpu_free_blocks = list(range(num_gpu_blocks))
        self.cpu_free_blocks = list(range(num_cpu_blocks))
        
        # 记录 Request 状态
        self.block_tables = {}      # req_id -> [block_idx_1, block_idx_2, ...]
        self.request_status = {}    # req_id -> "GPU" 或 "SWAPPED_TO_CPU"

    def allocate_gpu_blocks(self, req_id, num_blocks):
        """正常在 GPU 上分配块"""
        if len(self.gpu_free_blocks) < num_blocks:
            return False # 显存不足，需要触发抢占
        
        allocated = [self.gpu_free_blocks.pop(0) for _ in range(num_blocks)]
        if req_id not in self.block_tables:
            self.block_tables[req_id] = []
        self.block_tables[req_id].extend(allocated)
        self.request_status[req_id] = "GPU"
        return True

    # ==========================================
    # 抢占机制 A: Drop and Recompute (丢弃并重算)
    # ==========================================
    def preempt_drop(self, req_id):
        """释放该请求的所有 GPU 块，彻底清空页表"""
        blocks = self.block_tables.pop(req_id, [])
        self.gpu_free_blocks.extend(blocks)
        self.request_status.pop(req_id, None)
        return len(blocks)

    def simulate_recompute(self, seq_len):
        """模拟重算全量 Prompt Attention 的时间消耗"""
        # 构造一个能体现 O(N^2) 复杂度的矩阵乘法来模拟重算开销
        dummy_q = torch.randn((1, self.num_heads, seq_len, self.head_dim), dtype=torch.float16, device=self.device)
        dummy_k = torch.randn((1, self.num_heads, seq_len, self.head_dim), dtype=torch.float16, device=self.device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 模拟 Attention 计算
        scores = torch.matmul(dummy_q, dummy_k.transpose(-2, -1))
        _ = F.softmax(scores, dim=-1)
        
        torch.cuda.synchronize()
        return time.perf_counter() - start

    # ==========================================
    # 抢占机制 B: Swap Out & Swap In (换页机制)
    # ==========================================
    def preempt_swap_out(self, req_id):
        """将 GPU 上的块通过 PCIe 拷贝到 CPU 上"""
        gpu_blocks = self.block_tables[req_id]
        if len(self.cpu_free_blocks) < len(gpu_blocks):
            raise RuntimeError("CPU 内存也满了！系统彻底崩溃。")

        cpu_blocks = [self.cpu_free_blocks.pop(0) for _ in range(len(gpu_blocks))]
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 将数据从 Device (GPU) 转移到 Host (CPU)
        for i, (g_idx, c_idx) in enumerate(zip(gpu_blocks, cpu_blocks)):
            self.cpu_k_pool[c_idx].copy_(self.gpu_k_pool[g_idx])
            self.cpu_v_pool[c_idx].copy_(self.gpu_v_pool[g_idx])
            
        torch.cuda.synchronize()
        swap_time = time.perf_counter() - start
        
        # 更新状态与页表
        self.gpu_free_blocks.extend(gpu_blocks)
        self.block_tables[req_id] = cpu_blocks
        self.request_status[req_id] = "SWAPPED_TO_CPU"
        
        return swap_time

    def swap_in(self, req_id):
        """将 CPU 上的块拷贝回 GPU，恢复执行"""
        cpu_blocks = self.block_tables[req_id]
        if len(self.gpu_free_blocks) < len(cpu_blocks):
            return False, 0.0 # 无法 Swap In，GPU 依然满载
            
        gpu_blocks = [self.gpu_free_blocks.pop(0) for _ in range(len(cpu_blocks))]
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 将数据从 Host (CPU) 转移回 Device (GPU)
        for i, (c_idx, g_idx) in enumerate(zip(cpu_blocks, gpu_blocks)):
            self.gpu_k_pool[g_idx].copy_(self.cpu_k_pool[c_idx])
            self.gpu_v_pool[g_idx].copy_(self.cpu_v_pool[c_idx])
            
        torch.cuda.synchronize()
        swap_time = time.perf_counter() - start
        
        # 更新状态与页表
        self.cpu_free_blocks.extend(cpu_blocks)
        self.block_tables[req_id] = gpu_blocks
        self.request_status[req_id] = "GPU"
        
        return True, swap_time

# ==========================================
# 核心评测：在 3080 上跑出性能差异
# ==========================================
def run_preemption_benchmark():
    print("\n" + "="*50)
    print("开始执行抢占机制 Benchmark (Recompute vs Swap)")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告: 未检测到 GPU，无法测试真实 PCIe 带宽与算力差异。")
        return

    # 初始化配置
    BLOCK_SIZE = 16
    NUM_HEADS = 32
    HEAD_DIM = 128
    
    # 构建引擎
    engine = PagedAttentionEngineWithPreemption(
        num_gpu_blocks=500,  # 模拟少量 GPU 显存
        num_cpu_blocks=2000, # CPU 内存很充裕
        block_size=BLOCK_SIZE, num_heads=NUM_HEADS, head_dim=HEAD_DIM, device=device
    )

    # 设定一个长文本请求 (例如 2048 个 tokens, 需要 128 个块)
    req_id = "request_long_001"
    seq_len = 2048
    num_blocks_needed = seq_len // BLOCK_SIZE
    
    # 1. 正常分配并填充假数据 (模拟已经生成到一半)
    engine.allocate_gpu_blocks(req_id, num_blocks_needed)
    
    print("\n[突发情况] 系统突然接收到高优先级任务，显存耗尽，必须踢出当前任务！")
    
    # --------------------------------------------------
    # 路线 A 测试: Drop and Recompute (丢弃并重算)
    # --------------------------------------------------
    print("\n--- 策略 A: 丢弃并重算 ---")
    engine.preempt_drop(req_id)
    print("已丢弃缓存。等待 GPU 空闲...")
    # 假设此时 GPU 空闲了，需要恢复该任务：
    engine.allocate_gpu_blocks(req_id, num_blocks_needed)
    recompute_time = engine.simulate_recompute(seq_len)
    print(f"恢复完成！重新计算 {seq_len} 个 Token 的 Attention 耗时: {recompute_time*1000:.2f} ms")
    
    # 清理环境，为路线 B 准备
    engine.preempt_drop(req_id) 
    
    # --------------------------------------------------
    # 路线 B 测试: Swap Out & In (PCIe 换页)
    # --------------------------------------------------
    print("\n--- 策略 B: CPU 换页 (Swap) ---")
    engine.allocate_gpu_blocks(req_id, num_blocks_needed) # 重新分配初始状态
    
    swap_out_time = engine.preempt_swap_out(req_id)
    print(f"已 Swap Out 到 CPU。释放 GPU 显存耗时: {swap_out_time*1000:.2f} ms")
    
    # 假设此时 GPU 空闲了，需要恢复该任务：
    success, swap_in_time = engine.swap_in(req_id)
    print(f"恢复完成！从 CPU Swap In 回显存耗时: {swap_in_time*1000:.2f} ms")
    total_swap_time = swap_out_time + swap_in_time
    
    # --------------------------------------------------
    # 输出最终结论
    # --------------------------------------------------
    print("\n" + "="*50)
    print(" Benchmark 结论")
    print("="*50)
    print(f"请求长度: {seq_len} tokens")
    print(f"重算策略 (Recompute) 总延迟: {recompute_time*1000:.2f} ms")
    print(f"换页策略 (Swap Out+In) 总延迟: {total_swap_time*1000:.2f} ms")
    speedup = recompute_time / total_swap_time if total_swap_time > 0 else 0
    print(f"\n加速比: Swap 策略使得系统恢复速度提升了 {speedup:.2f} 倍！")
    print("="*50)

if __name__ == "__main__":
    run_preemption_benchmark()