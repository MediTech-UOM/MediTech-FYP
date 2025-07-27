import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Add
from tensorflow.keras import activations

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
    """
    Graph Attention Head with learnable attention coefficients, TF 2.x style but compatible with graph mode.
    Uses tf.nn.dropout with a tensor rate (no Python boolean checks).

    Arguments:
      seq        : Tensor of shape (batch_size, nb_nodes, fea_size)
      out_sz     : int, output feature dimension for each node
      bias_mat   : Tensor (batch_size, nb_nodes, nb_nodes), large negative on non-edges
      activation : activation function, e.g. tf.nn.elu
      in_drop    : float or Tensor, dropout rate on features
      coef_drop  : float or Tensor, dropout rate on attention coefficients
      residual   : bool, whether to add a residual connection
      return_coef: bool, if True, also return attention coefficients

    Returns:
      If return_coef=False:
        Tensor of shape (batch_size, nb_nodes, out_sz)
      If return_coef=True:
        (Tensor [batch_size, nb_nodes, out_sz], attention_coeffs [batch_size, nb_nodes, nb_nodes])
    """
    # 1) Dropout on input features (works even when in_drop is a placeholder)
    if isinstance(in_drop, (int, float)) and in_drop == 0.0:
        seq_dropped = seq
    else:
        seq_dropped = tf.nn.dropout(seq, rate=in_drop)

    # 2) Linear projection via Conv1D with kernel_size=1 (no bias)
    seq_fts = Conv1D(filters=out_sz, kernel_size=1, use_bias=False)(seq_dropped)

    # 3) Compute f₁ and f₂: each is shape (batch, nb_nodes, 1)
    f_1 = Conv1D(filters=1, kernel_size=1)(seq_fts)
    f_2 = Conv1D(filters=1, kernel_size=1)(seq_fts)

    # 4) Build pairwise logits: f₁ + f₂ᵀ
    logits = f_1 + tf.transpose(f_2, perm=[0, 2, 1])  # shape: (batch, nb_nodes, nb_nodes)
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat, axis=-1)

    # 5) Dropout on coefficients
    if isinstance(coef_drop, (int, float)) and coef_drop == 0.0:
        coefs_dropped = coefs
    else:
        coefs_dropped = tf.nn.dropout(coefs, rate=coef_drop)

    # 6) Optionally dropout seq_fts again
    if isinstance(in_drop, (int, float)) and in_drop == 0.0:
        seq_fts_dropped = seq_fts
    else:
        seq_fts_dropped = tf.nn.dropout(seq_fts, rate=in_drop)

    # 7) Weighted sum: coefs_dropped @ seq_fts_dropped → shape (batch, nb_nodes, out_sz)
    vals = tf.matmul(coefs_dropped, seq_fts_dropped)

    # 8) No explicit bias_add needed (original code used tf.contrib.layers.bias_add(vals))
    ret = vals

    # 9) Residual connection if requested
    if residual:
        seq_shape = tf.shape(seq)
        ret_shape = tf.shape(ret)
        # We cannot do Python‐side seq.shape[-1] test because seq.shape[-1] may be unknown
        # Instead, compare dynamic shapes and project if needed:
        cond = tf.equal(tf.shape(seq)[-1], tf.shape(ret)[-1])

        def add_identity():
            return Add()([ret, seq])
        def add_projected():
            proj = Conv1D(filters=ret_shape[-1], kernel_size=1)(seq)
            return Add()([ret, proj])

        ret = tf.cond(cond, add_identity, add_projected)

    # 10) Final activation
    out = activation(ret)

    if return_coef:
        return out, coefs_dropped
    else:
        return out


def attn_head_const_1(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    """
    Graph Attention Head with constant attention (uses adjacency/bias directly).
    Same dropout‐handling style as attn_head.
    """
    # Build adjacency mask from bias
    adj_mat = 1.0 - bias_mat / -1e9  # 1 where edge, 0 otherwise

    # 1) Dropout on features
    if isinstance(in_drop, (int, float)) and in_drop == 0.0:
        seq_dropped = seq
    else:
        seq_dropped = tf.nn.dropout(seq, rate=in_drop)

    # 2) Linear projection
    seq_fts = Conv1D(filters=out_sz, kernel_size=1, use_bias=False)(seq_dropped)

    # 3) Coefficients from adjacency
    logits = adj_mat
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat, axis=-1)

    # 4) Dropout on coefficients
    if isinstance(coef_drop, (int, float)) and coef_drop == 0.0:
        coefs_dropped = coefs
    else:
        coefs_dropped = tf.nn.dropout(coefs, rate=coef_drop)

    # 5) Dropout on seq_fts
    if isinstance(in_drop, (int, float)) and in_drop == 0.0:
        seq_fts_dropped = seq_fts
    else:
        seq_fts_dropped = tf.nn.dropout(seq_fts, rate=in_drop)

    # 6) Weighted sum
    vals = tf.matmul(coefs_dropped, seq_fts_dropped)
    ret = vals

    # 7) Residual connection
    if residual:
        cond = tf.equal(tf.shape(seq)[-1], tf.shape(ret)[-1])
        def add_identity():
            return Add()([ret, seq])
        def add_projected():
            proj = Conv1D(filters=tf.shape(ret)[-1], kernel_size=1)(seq)
            return Add()([ret, proj])
        ret = tf.cond(cond, add_identity, add_projected)

    return activation(ret)


def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    """
    Sparse graph attention head. Assumes:
      - seq shape = (1, nb_nodes, fea_size)
      - adj_mat is a tf.sparse.SparseTensor of shape (nb_nodes, nb_nodes)
    This version converts sparse to dense for simplicity.
    """
    # 1) Dropout on features
    if isinstance(in_drop, (int, float)) and in_drop == 0.0:
        seq_dropped = seq
    else:
        seq_dropped = tf.nn.dropout(seq, rate=in_drop)

    # 2) Linear projection
    seq_fts = Conv1D(filters=out_sz, kernel_size=1, use_bias=False)(seq_dropped)

    # 3) f_1, f_2 computation (both [1, nb_nodes, 1])
    f_1 = Conv1D(filters=1, kernel_size=1)(seq_fts)
    f_2 = Conv1D(filters=1, kernel_size=1)(seq_fts)

    # 4) Build dense adjacency and logits
    f_1_dense = tf.squeeze(f_1, axis=0)  # shape = [nb_nodes, 1]
    f_2_dense = tf.squeeze(f_2, axis=0)  # shape = [nb_nodes, 1]
    dense_adj = tf.sparse.to_dense(adj_mat)  # shape = [nb_nodes, nb_nodes]

    logits_dense = dense_adj * (f_1_dense + tf.transpose(f_2_dense))
    lrelu = tf.nn.leaky_relu(logits_dense)
    bias_dense = dense_adj * -1e9 + dense_adj  # reconstruct bias (large negative off‐edge)
    coefs_dense = tf.nn.softmax(lrelu + bias_dense, axis=-1)

    # 5) Dropout on coefficients
    if isinstance(coef_drop, (int, float)) and coef_drop == 0.0:
        coefs_dropped = coefs_dense
    else:
        coefs_dropped = tf.nn.dropout(coefs_dense, rate=coef_drop)

    # 6) Dropout on seq_fts
    if isinstance(in_drop, (int, float)) and in_drop == 0.0:
        seq_fts_dropped = seq_fts
    else:
        seq_fts_dropped = tf.nn.dropout(seq_fts, rate=in_drop)

    # 7) Weighted sum (need batch dim back)
    coefs_batched = tf.expand_dims(coefs_dropped, axis=0)  # shape = [1, nb_nodes, nb_nodes]
    vals = tf.matmul(coefs_batched, seq_fts_dropped)       # shape = [1, nb_nodes, out_sz]
    ret = vals

    # 8) Residual connection
    if residual:
        cond = tf.equal(tf.shape(seq)[-1], tf.shape(ret)[-1])
        def add_identity():
            return Add()([ret, seq])
        def add_projected():
            proj = Conv1D(filters=tf.shape(ret)[-1], kernel_size=1)(seq)
            return Add()([ret, proj])
        ret = tf.cond(cond, add_identity, add_projected)

    return activation(ret)


def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Simple temporal attention layer.

    Arguments:
      inputs: Tensor of shape (batch, time, features) or tuple (fw, bw) to concatenate
      attention_size: int, hidden dimension for attention
      time_major: bool, if inputs shape is (time, batch, features)
      return_alphas: bool, if True, also return attention weights

    Returns:
      If return_alphas=False: Tensor (batch, features)
      If return_alphas=True: (Tensor [batch, features], alphas [batch, time])
    """
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, axis=2)

    if time_major:
        inputs = tf.transpose(inputs, perm=[1, 0, 2])  # (batch, time, features)

    hidden_size = inputs.shape[-1]

    w_omega = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)  # (batch, time, attention_size)

    vu = tf.tensordot(v, u_omega, axes=1)  # (batch, time)
    alphas = tf.nn.softmax(vu, axis=1)     # (batch, time)

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)  # (batch, features)

    if return_alphas:
        return output, alphas
    else:
        return output
