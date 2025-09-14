import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

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

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    # 3. Phase 2: Diagonal Blocks
    window_start = max(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
    # --- Phase 0: Sink blocks that are before the sliding window ---
    sink_end = min(SINK_SIZE, window_start)
    for start_n in range(0, sink_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        # Attend to sink tokens only (indices < SINK_SIZE), while preserving causality j <= i
        valid = (k_offsets[None, :] < SINK_SIZE)  
        s_ij = tl.where(valid, s_ij, -float('inf'))
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
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # 2. Rescale accumulators; guard all-masked tiles to avoid NaNs.
        no_contrib = m_new == -float('inf')
        alpha = tl.where(no_contrib, 1.0, tl.exp2(m_i - m_new))
        acc_rescaled = acc * alpha[:, None]
        l_i_rescaled = l_i * alpha
        # 3. Compute probabilities safely.
        s_shifted = s_ij - m_new[:, None]
        s_shifted = tl.where(no_contrib[:, None], -float('inf'), s_shifted)
        P_tilde_ij = tl.exp2(s_shifted)
        # 4. Update accumulators.
        l_i = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij, v_block)
        # 5. Update running maximum for next iteration.
        m_i = m_new
    # --- STUDENT IMPLEMENTATION REQUIRED (Part 2: SWA Logic) ---
    # Now, implement the "sliding window" by changing the loop bounds.
    # The kernel should only attend to the `WINDOW_SIZE` most recent key/value tokens.
    # 1. Calculate the starting position of the attention window (window_start).
    # 2. Modify the range of the Phase 1 loop to start from your window_start.

    

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # Hint: You might need to apply the per-element sliding window mask to s_ij.
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        valid = (q_offsets[:, None] - k_offsets[None, :]) < (WINDOW_SIZE)
        valid = valid | (k_offsets[None, :] < SINK_SIZE)
        s_ij = tl.where(valid, s_ij, -float('inf'))
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
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # 2. Rescale accumulators; guard all-masked tiles to avoid NaNs.
        no_contrib = m_new == -float('inf')
        alpha = tl.where(no_contrib, 1.0, tl.exp2(m_i - m_new))
        acc_rescaled = acc * alpha[:, None]
        l_i_rescaled = l_i * alpha
        # 3. Compute probabilities safely.
        s_shifted = s_ij - m_new[:, None]
        s_shifted = tl.where(no_contrib[:, None], -float('inf'), s_shifted)
        P_tilde_ij = tl.exp2(s_shifted)
        # 4. Update accumulators.
        l_i = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij, v_block)
        # 5. Update running maximum for next iteration.
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        # Causal + sliding window mask within the diagonal block
        valid = (k_offsets[None, :] <= q_offsets[:, None]) & \
                ((q_offsets[:, None] - k_offsets[None, :]) <(WINDOW_SIZE )) & \
                (k_offsets[None, :] < SEQ_LEN)
        valid = valid | ((k_offsets[None, :] < SINK_SIZE) & (k_offsets[None, :] <= q_offsets[:, None]) &(k_offsets[None, :] < SEQ_LEN))
        s_ij = tl.where(valid, s_ij, -float('inf'))
        # mask = q_offsets[:, None]-k_offsets[None, :] >= WINDOW_SIZE
        # s_ij = tl.where(mask, s_ij, -float('inf'))
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
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # 2. Rescale accumulators; guard all-masked tiles to avoid NaNs.
        no_contrib = m_new == -float('inf')
        alpha = tl.where(no_contrib, 1.0, tl.exp2(m_i - m_new))
        acc_rescaled = acc * alpha[:, None]
        l_i_rescaled = l_i * alpha
        # 3. Compute probabilities safely.
        s_shifted = s_ij - m_new[:, None]
        s_shifted = tl.where(no_contrib[:, None], -float('inf'), s_shifted)
        P_tilde_ij = tl.exp2(s_shifted)
        # 4. Update accumulators.
        l_i = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij, v_block)
        # 5. Update running maximum for next iteration.
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---
        # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o