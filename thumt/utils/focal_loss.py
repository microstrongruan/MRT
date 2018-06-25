import tensorflow as tf


def focal_loss(logits, labels, weights=None, alpha=0.25, gamma=2, scope=None):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        # loss: A (scalar) tensor representing the value of the loss function
        loss: A [batch_size, num_anchors] tensor representing the value of the loss function
    """
    with tf.name_scope(scope or "smoothed_softmax_cross_entropy_with_logits",
                       values=[logits, labels]):

        sigmoid_p = tf.nn.sigmoid(logits)
        zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        vocab_size = tf.shape(logits)[1]

        labels = tf.reshape(labels, [-1])
        labels = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size)
        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = tf.where(labels > zeros, labels - sigmoid_p, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(labels > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (tf.pow(pos_p_sub,gamma)) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (tf.pow(neg_p_sub,gamma)) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent,-1)


if __name__ == '__main__':
    import numpy as np
    a = tf.Variable(np.random.random(([100,32000])).astype(np.float32))
    b = tf.Variable(np.random.random(([10,10])).astype(np.int32))
    x = focal_loss(a, b)
    print('x', x.shape)
    print('label', b.shape)