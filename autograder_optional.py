import sys
import argparse
import time
import math

import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def repeat_kv(x, num_groups):
    if num_groups == 1:
        return x
    B, H_kv, N, D = x.shape
    x = x.unsqueeze(2).expand(B, H_kv, num_groups, N, D)
    return x.reshape(B, H_kv * num_groups, N, D)

def create_mask_bool(
    seq_len: int,
    window_size: int,
    sink_size: int,
    device=None
    ) -> torch.Tensor:
    
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)
    col = idx.unsqueeze(0)

    sliding = (col <= row) & (col >= row - (window_size - 1))
    sink = (col < sink_size) & (col <= row)

    return sliding | sink

def naive_attention(q, k, v, seq_len, window_size, sink_size):
    return F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=create_mask_bool(seq_len, window_size, sink_size, device=q.device),
        enable_gqa=True,
    )
    
def naive_attention_matmul(Q, K, V, *, is_causal=False, window_size=None, sink_size=None):
    """Matmul-based naive attention (PyTorch), supports GQA, SWA and sinks. Returns O only."""
    batch_size, num_heads_q, seq_len, head_dim = Q.shape
    _, num_heads_kv, _, _ = K.shape

    if num_heads_q != num_heads_kv:
        num_groups = num_heads_q // num_heads_kv
        K = repeat_kv(K, num_groups)
        V = repeat_kv(V, num_groups)

    scale = 1.0 / math.sqrt(head_dim)
    S = (Q @ K.transpose(-1, -2)) * scale

    if is_causal:
        if window_size is None:
            mask = create_mask_bool(seq_len=seq_len, window_size=seq_len, sink_size=0, device=Q.device)
        else:
            if sink_size is None:
                mask = create_mask_bool(seq_len, window_size=window_size, sink_size=0, device=Q.device)
            else:
                mask = create_mask_bool(seq_len, window_size=window_size, sink_size=sink_size, device=Q.device)
        S = S.masked_fill(~mask, -float('inf'))

    P = torch.softmax(S.to(torch.float32), dim=-1).to(Q.dtype)
    O = P @ V
    return O
    
    
def check_backward_correctness(triton_func, problem_num):
    test_cases = [
        (1, 16, 16, 4096, 16, 256, 4),
        (1, 16, 8, 4096, 16, 256, 4),
        (1, 16, 1, 4096, 16, 256, 4),
    ]
    for case in test_cases:
        batch, heads_q, heads_kv, seq_len, dim, window_size, sink_size = case
        
        if problem_num == 8:
            print(f"Running test case: batch={batch}, heads_q={heads_q}, heads_kv={heads_kv}, seq_len={seq_len}, dim={dim}")
        elif problem_num == 9:
            print(f"Running test case: batch={batch}, heads_q={heads_q}, heads_kv={heads_kv}, seq_len={seq_len}, dim={dim}, window_size={window_size}, sink_size={sink_size}")
        else:
            raise ValueError(f"Problem {problem_num} not supported")
        
        q = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        k = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        v = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        
        q_ref, k_ref, v_ref = q.clone().detach().requires_grad_(), k.clone().detach().requires_grad_(), v.clone().detach().requires_grad_()
        
        if problem_num == 8:
            o_ref = naive_attention(q_ref, k_ref, v_ref, seq_len=seq_len, window_size=seq_len, sink_size=0)
            o_triton = triton_func(q, k, v, is_causal=True)
        elif problem_num == 9:
            o_ref = naive_attention(q_ref, k_ref, v_ref, seq_len, window_size, sink_size)
            o_triton = triton_func(q, k, v, window_size=window_size, sink_size=sink_size, is_causal=True)
        else:
            raise ValueError(f"Problem {problem_num} not supported")
            
        
        is_forward_correct = torch.allclose(o_ref, o_triton, atol=1e-2, rtol=1e-2)
        if is_forward_correct:
            print("✅ Forward Pass Results match")
        else:
            print("❌ Forward Pass Results do not match")
        
        dout = torch.rand_like(o_ref)
        o_ref.backward(dout)
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad
        
        o_triton.backward(dout)
        dq_flash, dk_flash, dv_flash = q.grad, k.grad, v.grad
        
        is_dq_correct = torch.allclose(dq_ref, dq_flash, atol=5e-2, rtol=5e-2)
        is_dk_correct = torch.allclose(dk_ref, dk_flash, atol=5e-2, rtol=5e-2)
        is_dv_correct = torch.allclose(dv_ref, dv_flash, atol=5e-2, rtol=5e-2)
        if is_dq_correct:
            print("✅ Backward Pass Results match on dQ")
        else:
            print("❌ Backward Pass Results do not match on dQ")
        if is_dk_correct:
            print("✅ Backward Pass Results match on dK")
        else:
            print("❌ Backward Pass Results do not match on dK")
        if is_dv_correct:
            print("✅ Backward Pass Results match on dV")
        else:
            print("❌ Backward Pass Results do not match on dV")

def _benchmark_forward_backward(triton_forward, torch_forward, *, warmup=5, iters=20):
    def _measure_forward(func):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            with torch.no_grad():
                _ = func()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            with torch.no_grad():
                _ = func()
        torch.cuda.synchronize()
        avg_ms = (time.time() - start) * 1000 / iters
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        return avg_ms, peak_mem_gb

    def _measure_train(func):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Warmup
        for _ in range(warmup):
            o, q, k, v = func(requires_grad=True)
            loss = o.sum()
            loss.backward()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            o, q, k, v = func(requires_grad=True)
            loss = o.sum()
            loss.backward()
        torch.cuda.synchronize()
        avg_ms = (time.time() - start) * 1000 / iters
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        return avg_ms, peak_mem_gb

    # Triton
    fwd_triton_ms, fwd_triton_mem = _measure_forward(lambda: triton_forward(requires_grad=False)[0])
    train_triton_ms, train_triton_mem = _measure_train(triton_forward)
    bwd_triton_ms = max(0.0, train_triton_ms - fwd_triton_ms)

    # Torch (naive)
    fwd_torch_ms, fwd_torch_mem = _measure_forward(lambda: torch_forward(requires_grad=False)[0])
    train_torch_ms, train_torch_mem = _measure_train(torch_forward)
    bwd_torch_ms = max(0.0, train_torch_ms - fwd_torch_ms)

    return {
        'triton': (fwd_triton_ms, bwd_triton_ms, train_triton_ms, fwd_triton_mem, train_triton_mem),
        'torch': (fwd_torch_ms, bwd_torch_ms, train_torch_ms, fwd_torch_mem, train_torch_mem),
    }

def benchmark_problem_8(triton_func):
    print("\n--- Running Performance Benchmark (Problem 8) ---")
    batch, heads_q, heads_kv, seq_len, dim = (1, 16, 1, 4096, 16)
    print(f"Benchmark Config: B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}, Causal=True")
    q_base = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE)
    k_base = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)
    v_base = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)

    def triton_forward(requires_grad=False):
        q = q_base.clone().detach().requires_grad_(requires_grad)
        k = k_base.clone().detach().requires_grad_(requires_grad)
        v = v_base.clone().detach().requires_grad_(requires_grad)
        o = triton_func(q, k, v, is_causal=True)
        return o, q, k, v

    def torch_forward(requires_grad=False):
        q = q_base.clone().detach().requires_grad_(requires_grad)
        k = k_base.clone().detach().requires_grad_(requires_grad)
        v = v_base.clone().detach().requires_grad_(requires_grad)
        o = naive_attention_matmul(q, k, v, is_causal=True, window_size=seq_len, sink_size=0)
        return o, q, k, v

    results = _benchmark_forward_backward(triton_forward, torch_forward)

    (tf_ms, tb_ms, tt_ms, tf_mem, tt_mem) = results['triton']
    (pf_ms, pb_ms, pt_ms, pf_mem, pt_mem) = results['torch']

    print("\n--- Benchmark Results (ms) ---")
    print(f"{'Implementation':<25} | {'Forward':<12} | {'Backward':<12} | {'Fwd+Bwd':<12}")
    print("-" * 70)
    print(f"{'PyTorch (Naive Attention )':<25} | {pf_ms:<12.4f} | {pb_ms:<12.4f} | {pt_ms:<12.4f}")
    print(f"{'Triton (Flash Attention + GQA + SWA + Attention Sinks)':<25} | {tf_ms:<12.4f} | {tb_ms:<12.4f} | {tt_ms:<12.4f}")
    print("-" * 70)
    print(f"Speedup (Forward):  {pf_ms / tf_ms:.2f}x")
    print(f"Speedup (Backward): {pb_ms / tb_ms:.2f}x")
    print(f"Speedup (End-to-End): {pt_ms / tt_ms:.2f}x")

    # Memory comparison
    print("\n--- Memory (Peak, GB) ---")
    print(f"{'Implementation':<25} | {'Forward':<12} | {'Fwd+Bwd':<12}")
    print("-" * 70)
    print(f"{'PyTorch (Naive)':<25} | {pf_mem:<12.4f} | {pt_mem:<12.4f}")
    print(f"{'Triton (Flash Attention + GQA + SWA + Attention Sinks)':<25} | {tf_mem:<12.4f} | {tt_mem:<12.4f}")
    print("-" * 70)
    fwd_mem_saving = (pf_mem / tf_mem) if tf_mem > 0 else float('inf')
    e2e_mem_saving = (pt_mem / tt_mem) if tt_mem > 0 else float('inf')
    print(f"Memory saving (Forward): {fwd_mem_saving:.2f}x")
    print(f"Memory saving (End-to-End): {e2e_mem_saving:.2f}x")

def benchmark_problem_9(triton_func):
    print("\n--- Running Performance Benchmark ---")
    batch, heads_q, heads_kv, seq_len, dim, window_size, sink_size = (1, 16, 1, 4096, 16, 256, 4)
    print(f"Benchmark Config: B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}, W={window_size}, S={sink_size}, Causal=True")
    q_base = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE)
    k_base = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)
    v_base = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE)

    def triton_forward(requires_grad=False):
        q = q_base.clone().detach().requires_grad_(requires_grad)
        k = k_base.clone().detach().requires_grad_(requires_grad)
        v = v_base.clone().detach().requires_grad_(requires_grad)
        o = triton_func(q, k, v, window_size=window_size, sink_size=sink_size, is_causal=True)
        return o, q, k, v

    def torch_forward(requires_grad=False):
        q = q_base.clone().detach().requires_grad_(requires_grad)
        k = k_base.clone().detach().requires_grad_(requires_grad)
        v = v_base.clone().detach().requires_grad_(requires_grad)
        o = naive_attention_matmul(q, k, v, is_causal=True, window_size=window_size, sink_size=sink_size)
        return o, q, k, v

    results = _benchmark_forward_backward(triton_forward, torch_forward)

    (tf_ms, tb_ms, tt_ms, tf_mem, tt_mem) = results['triton']
    (pf_ms, pb_ms, pt_ms, pf_mem, pt_mem) = results['torch']

    print("\n--- Benchmark Results (ms) ---")
    print(f"{'Implementation                                        ':<25} | {'Forward':<12} | {'Backward':<12} | {'Fwd+Bwd':<12}")
    print("-" * 70)
    print(f"{'PyTorch (Naive Attention)                             ':<25} | {pf_ms:<12.4f}ms | {pb_ms:<12.4f}ms | {pt_ms:<12.4f}ms")
    print(f"{'Triton (Flash Attention + GQA + SWA + Attention Sinks)':<25} | {tf_ms:<12.4f}ms | {tb_ms:<12.4f}ms | {tt_ms:<12.4f}ms")
    print("-" * 70)
    print(f"Speedup (Forward):  {pf_ms / tf_ms:.2f}x")
    print(f"Speedup (Backward): {pb_ms / tb_ms:.2f}x")
    print(f"Speedup (End-to-End): {pt_ms / tt_ms:.2f}x")

    # Memory comparison
    print("\n--- Memory (Peak, GB) ---")
    print(f"{'Implementation                                        ':<25} | {'Forward':<12} | {'Fwd+Bwd':<12}")
    print("-" * 70)
    print(f"{'PyTorch (Naive Attentio)                              ':<25} | {pf_mem:<12.4f}GB | {pt_mem:<12.4f}GB")
    print(f"{'Triton (Flash Attention + GQA + SWA + Attention Sinks)':<25} | {tf_mem:<12.4f}GB | {tt_mem:<12.4f}GB")
    print("-" * 70)
    fwd_mem_saving = (pf_mem / tf_mem) if tf_mem > 0 else float('inf')
    e2e_mem_saving = (pt_mem / tt_mem) if tt_mem > 0 else float('inf')
    print(f"Memory saving (End-to-End): {e2e_mem_saving:.2f}x")

def check_problem_8():
    """Checks Problem 8: GQA."""
    problem_num = 8
    print(f"\n--- Running Autograder for Problem {problem_num}: GQA Backward Pass ---")
    try:
        from problem_8 import flash_attention_gqa
    except ImportError:
        print(f"Could not import FlashAttention2Function from solution_{problem_num}.py.")
        return
    
    torch.manual_seed(48)
    check_backward_correctness(flash_attention_gqa, problem_num)
    benchmark_problem_8(flash_attention_gqa)
    

def check_problem_9():
    """Checks Problem 9: GQA + SWDA + Attention Sinks Backward Pass."""
    problem_num = 9
    print(f"\n--- Running Autograder for Problem {problem_num}: GQA + SWDA + Attention Sinks Backward Pass ---")
    try:
        from problem_9 import flash_swda_with_sink
    except ImportError:
        print(f"Could not import FlashAttention2Function from solution_{problem_num}.py.")
        return
    
    torch.manual_seed(48)
    check_backward_correctness(flash_swda_with_sink, problem_num)
    benchmark_problem_9(flash_swda_with_sink)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autograder for Triton Flash Attention assignments.")
    parser.add_argument('--p8', action='store_true', help='Run autograder for Problem 8 (GQA Backward Pass).')
    parser.add_argument('--p9', action='store_true', help='Run autograder for Problem 9 (GQA + SWDA + Attention Sinks Backward Pass).')
    
    if not torch.cuda.is_available():
        print("💥 CUDA not available. Skipping all GPU tests.")
        sys.exit(1)

    args = parser.parse_args()
    if not any(vars(args).values()):
        args.p8 = args.p9 = True
    
    if args.p8:
        check_problem_8()
    if args.p9:
        check_problem_9()