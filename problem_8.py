# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional
@triton.jit
def _flash_attention_forward_gqa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr, L_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    l_stride_b, l_stride_h, l_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for the forward pass of causal FlashAttention with GQA.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 1) ---
    # Your goal is to map the current query head (q_head_idx) to its corresponding shared key/value head (kv_head_idx).
    # 1. Calculate how many query heads are in each group.
    n_q_head_per_group = N_Q_HEADS // N_KV_HEADS
    # 2. Use integer division to find the correct kv_head_idx and clamp to valid range.
    kv_head_idx = q_head_idx // n_q_head_per_group
    kv_head_idx = tl.minimum(kv_head_idx, N_KV_HEADS - 1)
    # --- END OF STUDENT IMPLEMENTATION ---
    #(dhruv) this solution passes all the testa cses but ideally it ahould have edge case handling for when nq_heads is not completely divisible by nkv_heads

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
            (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    q_block = tl.cast(q_block, tl.float32)
    
    qk_scale = softmax_scale * 1.44269504
    
    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 2) ---
        # 1. Modify the pointer arithmetic for K and V to use your `kv_head_idx`.
        # 2. Reuse your working implementation for the online softmax update
        #    from your solution to Problem 4.
        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # Implement the logic for the off-diagonal blocks.
        # This is very similar to the non-causal version from Problem 3.
        # 1. Load the K and V blocks for the current iteration.
        # 2. Compute the attention scores (S_ij).
        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block, allow_tf32=False)
        s_ij *= qk_scale

        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v_block = tl.cast(v_block, tl.float32)

        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # Implement the online softmax update logic (streaming, numerically stable).
        # Ensure consistent fp32 dtype for reductions and dot products.
        # q_block = tl.cast(q_block, tl.float32)
        # k_block = tl.cast(k_block, tl.float32)
        # v_block = tl.cast(v_block, tl.float32)
        # s_ij = tl.cast(s_ij, tl.float32)

        # 1. Find the new running maximum (`m_new`).
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        # 2. Rescale the existing accumulator (`acc`) and denominator (`l_i`).
        l_i_rescaled = (l_i*(tl.exp2(m_i-m_new)))
        mult = tl.exp2(m_i - m_new) # shape: (q_len,)
        acc_rescaled = acc * mult[:, None] 
        # 3. Compute the attention probabilities for the current tile (`p_ij`).
        P_tilde_ij = tl.exp2(s_ij - m_new[:, None])
        # 4. Update the accumulator `acc` using `p_ij` and `v_block`.
        l_new = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij ,tl.cast(v_block, tl.float32), allow_tf32=False)
        # 5. Update the denominator `l_i`.
        l_i = l_new
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 3) ---
        # 1. Modify the pointer arithmetic for K and V to use your `kv_head_idx`.
        # 2. Reuse your working implementation for the masked online softmax
        #    update from your solution to Problem 4.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block, allow_tf32=False)
        s_ij *= qk_scale
        mask = k_offsets[None, :] <= q_offsets[:, None]
        s_ij = tl.where(mask, s_ij, -float('inf'))
        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v_block = tl.cast(v_block, tl.float32)

        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
        # Implement the online softmax update logic (streaming, numerically stable).
        # Ensure consistent fp32 dtype for reductions and dot products.
        # q_block = tl.cast(q_block, tl.float32)
        # k_block = tl.cast(k_block, tl.float32)
        # v_block = tl.cast(v_block, tl.float32)
        # s_ij = tl.cast(s_ij, tl.float32)

        # 1. Find the new running maximum (`m_new`).
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        # 2. Rescale the existing accumulator (`acc`) and denominator (`l_i`).
        l_i_rescaled = (l_i*(tl.exp2(m_i-m_new)))
        mult = tl.exp2(m_i - m_new) # shape: (q_len,)
        acc_rescaled = acc * mult[:, None] 
        # 3. Compute the attention probabilities for the current tile (`p_ij`).
        P_tilde_ij = tl.exp2(s_ij - m_new[:, None])
        # 4. Update the accumulator `acc` using `p_ij` and `v_block`.
        l_new = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij ,tl.cast(v_block, tl.float32), allow_tf32=False)
        # 5. Update the denominator `l_i`.
        l_i = l_new
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---
        # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.maximum(l_i[:, None], 1e-12)
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
            (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
            
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)
    temp = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
        # Use (B,H,S) strides when storing per-row stats
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + temp * m_stride_s
    tl.store(m_ptrs, m_i.to(M_ptr.dtype.element_ty), mask=(temp < SEQ_LEN))
    l_ptrs = L_ptr + batch_idx * l_stride_b + q_head_idx * l_stride_h + temp * l_stride_s
    tl.store(l_ptrs, l_i.to(L_ptr.dtype.element_ty), mask=(temp < SEQ_LEN))



@triton.jit
def _flash_attention_backward_gqa_kernel(
    # Inputs
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr, L_ptr, dO_ptr,
    # Outputs (grads)
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides for Q/K/V/O (assumed same layout for Q, O, dO, dQ)
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Strides for M/L (B,H,S)
    m_stride_b, m_stride_h, m_stride_s,
    l_stride_b, l_stride_h, l_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    LOG2E = 1.4426950408889634
    qk_scale2 = softmax_scale * LOG2E

    # Program ids
    q_block_idx = tl.program_id(axis=0)
    bh_idx = tl.program_id(axis=1)
    batch_idx = bh_idx // N_Q_HEADS
    q_head_idx = bh_idx % N_Q_HEADS

    # GQA mapping: map q_head to kv_head
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # Offsets and pointers for this query block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    hd = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])
    do_ptrs = dO_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])

    q_mask = q_offsets[:, None] < SEQ_LEN
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
    o_block = tl.load(o_ptrs, mask=q_mask, other=0.0)
    do_block = tl.load(do_ptrs, mask=q_mask, other=0.0)
    q_block = tl.cast(q_block, tl.float32)
    o_block = tl.cast(o_block, tl.float32)
    do_block = tl.cast(do_block, tl.float32)
    # Load stored per-row softmax stats
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    l_ptrs = L_ptr + batch_idx * l_stride_b + q_head_idx * l_stride_h + q_offsets * l_stride_s
    m_i = tl.load(m_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)
    l_i = tl.load(l_ptrs, mask=q_offsets < SEQ_LEN, other=1.0)
    l_i = tl.maximum(l_i, 1e-12)

    # delta = sum(dO * O) per row
    delta = tl.sum(do_block * o_block, axis=1)  # (BLOCK_M)

    # Accumulator for dQ
    dQ_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN

        # Load K and V tiles in both orientations as needed
        k_cols_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_rows_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[:, None] * k_stride_s + hd[None, :])
        v_rows_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[:, None] * v_stride_s + hd[None, :])
        v_cols_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[None, :] * v_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_cols = tl.load(k_cols_ptrs, mask=k_mask[None, :], other=0.0)      # (D, N)
        k_rows = tl.load(k_rows_ptrs, mask=k_mask[:, None], other=0.0)      # (N, D)
        v_rows = tl.load(v_rows_ptrs, mask=k_mask[:, None], other=0.0)      # (N, D)
        v_cols = tl.load(v_cols_ptrs, mask=k_mask[None, :], other=0.0)      # (D, N)
        # Cast tiles to fp32 for matmuls
        k_cols = tl.cast(k_cols, tl.float32)
        k_rows = tl.cast(k_rows, tl.float32)
        v_rows = tl.cast(v_rows, tl.float32)
        v_cols = tl.cast(v_cols, tl.float32)

        # Scores in base-2 domain and probabilities
        q_f32 = tl.cast(q_block, tl.float32)
        k_cols = tl.cast(k_cols, tl.float32)
        k_rows = tl.cast(k_rows, tl.float32)
        v_cols = tl.cast(v_cols, tl.float32)
        v_rows = tl.cast(v_rows, tl.float32)
        s2 = tl.dot(q_f32, k_cols, allow_tf32=False) * qk_scale2  # (M, N)
        s2 = tl.where((k_offsets[None, :] < SEQ_LEN), s2, -float('inf'))
        p_tilde = tl.exp2(s2 - m_i[:, None])
        P = p_tilde / l_i[:, None]

        # dV partial via matmul: P^T @ dO  => (N,D)
        do_f32 = tl.cast(do_block, tl.float32)
        dv_partial = tl.dot(tl.trans(P), do_f32, allow_tf32=False)

        # t_block = dO @ V^T using V in (D, N)
        t_block = tl.dot(do_f32, v_cols, allow_tf32=False)

        # dS = P * (t_block - delta[:, None])
        dS = P * (t_block - delta[:, None])

        # dQ += dS @ K_rows, scaled by softmax_scale
        dQ_acc += tl.dot(dS, k_rows, allow_tf32=False) * softmax_scale

        # dK partial via matmul: dS^T @ Q, scaled by softmax_scale
        q_f32 = tl.cast(q_block, tl.float32)
        dk_partial = tl.dot(tl.trans(dS), q_f32, allow_tf32=False) * softmax_scale

        # Atomic add into global dV and dK
        dv_ptrs = dV_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[:, None] * v_stride_s + hd[None, :])
        dk_ptrs = dK_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[:, None] * k_stride_s + hd[None, :])
        tl.atomic_add(dv_ptrs, dv_partial, mask=k_mask[:, None])
        tl.atomic_add(dk_ptrs, dk_partial, mask=k_mask[:, None])

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN

        k_cols_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_rows_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[:, None] * k_stride_s + hd[None, :])
        v_rows_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[:, None] * v_stride_s + hd[None, :])
        v_cols_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[None, :] * v_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_cols = tl.load(k_cols_ptrs, mask=k_mask[None, :], other=0.0)
        k_rows = tl.load(k_rows_ptrs, mask=k_mask[:, None], other=0.0)
        v_rows = tl.load(v_rows_ptrs, mask=k_mask[:, None], other=0.0)
        v_cols = tl.load(v_cols_ptrs, mask=k_mask[None, :], other=0.0)
        k_cols = tl.cast(k_cols, tl.float32)
        k_rows = tl.cast(k_rows, tl.float32)
        v_rows = tl.cast(v_rows, tl.float32)
        v_cols = tl.cast(v_cols, tl.float32)

        # Scores with causal mask inside diagonal tile
        q_f32 = tl.cast(q_block, tl.float32)
        k_cols = tl.cast(k_cols, tl.float32)
        k_rows = tl.cast(k_rows, tl.float32)
        v_cols = tl.cast(v_cols, tl.float32)
        v_rows = tl.cast(v_rows, tl.float32)

        s2 = tl.dot(q_f32, k_cols, allow_tf32=False) * qk_scale2
        causal = k_offsets[None, :] <= q_offsets[:, None] 
        valid = causal & (k_offsets[None, :] < SEQ_LEN)
        s2 = tl.where(valid, s2, -float('inf'))
        p_tilde = tl.exp2(s2 - m_i[:, None])
        P = p_tilde / l_i[:, None]

        # dV partial P^T @ dO
        do_f32 = tl.cast(do_block, tl.float32)
        dv_partial = tl.dot(tl.trans(P), do_f32, allow_tf32=False)

        # t_block = dO @ V^T
        t_block = tl.dot(do_f32, v_cols, allow_tf32=False)
        t_block = tl.where(valid, t_block, 0.0)

        dS = P * (t_block - delta[:, None])
        dS = tl.where(valid, dS, 0.0)

        dQ_acc += tl.dot(dS, k_rows, allow_tf32=False) * softmax_scale

        q_f32 = tl.cast(q_block, tl.float32)
        dk_partial = tl.dot(tl.trans(dS), q_f32, allow_tf32=False) * softmax_scale

        dv_ptrs = dV_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[:, None] * v_stride_s + hd[None, :])
        dk_ptrs = dK_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[:, None] * k_stride_s + hd[None, :])
        tl.atomic_add(dv_ptrs, dv_partial, mask=k_mask[:, None])
        tl.atomic_add(dk_ptrs, dk_partial, mask=k_mask[:, None])

    # Write dQ for this block
    dQ_ptrs = dQ_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])
    tl.store(dQ_ptrs, dQ_acc, mask=q_mask)


class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    


    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)
        L = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        n_q_heads = n_heads
        _flash_attention_forward_gqa_kernel[grid](
        q, k, v, o, M, L,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M, L)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, L = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads
        # Allocate fp32 gradient buffers for safe atomic_add in Triton
        dq_f = torch.zeros_like(q, dtype=torch.float32)
        dk_rep_f = torch.zeros_like(k, dtype=torch.float32)
        dv_rep_f = torch.zeros_like(v, dtype=torch.float32)

        # Implement backward using a numerically stable streaming algorithm with block-wise keys.
        # This avoids allocating full (seq_len x seq_len) attention tensors.
        softmax_scale = ctx.softmax_scale
        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        n_q_heads = n_heads
        _flash_attention_backward_gqa_kernel[grid](
        q, k, v, o, M, L, do,
        dq_f, dk_rep_f, dv_rep_f,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        )
        # Cast back to original dtypes
        return dq_f.to(q.dtype), dk_rep_f.to(k.dtype), dv_rep_f.to(v.dtype), None, None

    @staticmethod
    def backward2(ctx, do):
        q, k, v, o, M, L = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        # Implement backward using a numerically stable streaming algorithm with block-wise keys.
        # This avoids allocating full (seq_len x seq_len) attention tensors.
        scale = ctx.softmax_scale

        # Repeat K/V across groups to align with query heads (GQA expansion)
        num_groups = n_heads // n_kv_heads
        if num_groups == 1:
            k_rep = k
            v_rep = v
        else:
            k_rep = k.unsqueeze(2).expand(batch, n_kv_heads, num_groups, seq_len, head_dim).reshape(batch, n_heads, seq_len, head_dim)
            v_rep = v.unsqueeze(2).expand(batch, n_kv_heads, num_groups, seq_len, head_dim).reshape(batch, n_heads, seq_len, head_dim)

        # Use fp32 for stability
        q_f = q.to(torch.float32)
        k_rep_f = k_rep.to(torch.float32)
        v_rep_f = v_rep.to(torch.float32)
        do_f = do.to(torch.float32)
        o_f = o.to(torch.float32)

        # delta = (dO · O) per row (B, Hq, N, 1)
        delta = (do_f * o_f).sum(dim=-1, keepdim=True)

        # Use stored per-row softmax stats from forward
        m_i = M.to(torch.float32)
        l_i = L.to(torch.float32).clamp_min(1e-12)

        row_idx = torch.arange(seq_len, device=q.device)
        BLOCK_N = 128
        LOG2E = 1.4426950408889634
        scale2 = scale * LOG2E

        # Second pass: compute grads
        dq_f = torch.zeros_like(q_f)
        dk_rep_f = torch.zeros_like(k_rep_f)
        dv_rep_f = torch.zeros_like(v_rep_f)

        for start_n in range(0, seq_len, BLOCK_N):
            end_n = min(seq_len, start_n + BLOCK_N)
            k_block = k_rep_f[:, :, start_n:end_n, :]  # (B, Hq, BN, D)
            v_block = v_rep_f[:, :, start_n:end_n, :]  # (B, Hq, BN, D)

            s2_block = torch.matmul(q_f, k_block.transpose(-1, -2)) * scale2
            col_idx = torch.arange(start_n, end_n, device=q.device)
            valid = (col_idx.view(1, 1, 1, -1) <= row_idx.view(1, 1, -1, 1))
            s2_block = s2_block.masked_fill(~valid, -float('inf'))

            # Reconstruct probabilities in the same base-2 domain as forward
            p_tilde = torch.pow(2.0, s2_block - m_i.unsqueeze(-1))
            denom = l_i.unsqueeze(-1)
            p_block = p_tilde / denom

            # dV += P^T @ dO
            dv_rep_f[:, :, start_n:end_n, :] += torch.matmul(p_block.transpose(-1, -2), do_f)

            # t = dO @ V^T
            t_block = torch.matmul(do_f, v_block.transpose(-1, -2))

            # dS = P ⊙ (t - delta)
            dS_block = p_block * (t_block - delta)

            # dQ += dS @ K
            dq_f += torch.matmul(dS_block, k_block) * scale

            # dK += dS^T @ Q
            dk_rep_f[:, :, start_n:end_n, :] += torch.matmul(dS_block.transpose(-1, -2), q_f) * scale

        # Collapse group dimension back to KV heads for dK and dV
        if num_groups == 1:
            dk_f = dk_rep_f
            dv_f = dv_rep_f
        else:
            dk_f = dk_rep_f.view(batch, n_kv_heads, num_groups, seq_len, head_dim).sum(dim=2)
            dv_f = dv_rep_f.view(batch, n_kv_heads, num_groups, seq_len, head_dim).sum(dim=2)

        dq = dq_f.to(q.dtype)
        dk = dk_f.to(k.dtype)
        dv = dv_f.to(v.dtype)

        return dq, dk, dv, None, None
    

def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)