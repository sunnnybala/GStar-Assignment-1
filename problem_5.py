import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_gqa_kernel(
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
    # 2. Use integer division to find the correct kv_head_idx.
    kv_head_idx = q_head_idx // n_q_head_per_group # Placeholder: Replace with your calculation
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
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

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
        acc = acc_rescaled + tl.dot(P_tilde_ij ,tl.cast(v_block, tl.float32))
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
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        mask = k_offsets[None, :] <= q_offsets[:, None]
        s_ij = tl.where(mask, s_ij, -float('inf'))
        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

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
        acc = acc_rescaled + tl.dot(P_tilde_ij ,tl.cast(v_block, tl.float32))
        # 5. Update the denominator `l_i`.
        l_i = l_new
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---
        # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True):
    """
    Python wrapper for the GQA-enabled causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_gqa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o