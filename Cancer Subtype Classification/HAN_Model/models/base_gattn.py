import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class BaseGAttN:
    @staticmethod
    def loss(logits, labels, nb_classes, class_weights):
        """
        Weighted cross‐entropy loss (sparse) for multi‐class classification.

        Args:
          logits: Tensor of shape [N, nb_classes].
          labels: 1D int Tensor of shape [N].
          nb_classes: int, number of classes.
          class_weights: 1D float Tensor of shape [nb_classes].
        """
        # One-hot encode labels → shape [N, nb_classes]
        one_hot_labels = tf.one_hot(labels, nb_classes)
        # Compute sample weights for each example
        sample_wts = tf.reduce_sum(tf.multiply(one_hot_labels, class_weights), axis=-1)
        # Compute sparse cross-entropy per example
        xentropy_per_example = tf.multiply(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits),
            sample_wts
        )
        # Return mean over the batch
        return tf.reduce_mean(xentropy_per_example, name='xentropy_mean')

    @staticmethod
    def training(loss, lr, l2_coef):
        """
        Creates a training operation that minimizes (loss + L2 regularization).
        
        Args:
          loss: scalar loss Tensor.
          lr:   float or scalar Tensor = learning rate.
          l2_coef: float or scalar Tensor = L2 weight decay coefficient.

        Returns:
          A tf.Operation that performs one optimization step.
        """
        # 1) Collect all trainable variables under v1 compatibility
        vars = tf.compat.v1.trainable_variables()

        # 2) Compute L2 loss over all variables except bias / batch‐norm terms
        l2_terms = []
        for v in vars:
            # Skip if variable name indicates a bias or batch-norm scale/offset
            name = v.name.lower()
            if any(skip in name for skip in ['bias', 'beta', 'gamma', 'b', 'g']):
                continue
            l2_terms.append(tf.nn.l2_loss(v))
        lossL2 = tf.add_n(l2_terms) * l2_coef

        # 3) Use AdamOptimizer from v1 compatibility
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # 4) Create the train_op that minimizes (loss + L2)
        train_op = optimizer.minimize(loss + lossL2)
        return train_op

    @staticmethod
    def preshape(logits, labels, nb_classes):
        """
        Reshape logits and labels for use in sparse cross‐entropy.

        Args:
          logits: Tensor of arbitrary shape [*, nb_classes]
          labels: Tensor of arbitrary shape [*]
          nb_classes: int

        Returns:
          log_resh: 2D Tensor [N, nb_classes]
          lab_resh: 1D Tensor [N]
        """
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    @staticmethod
    def confmat(logits, labels):
        """
        Computes a confusion matrix on predicted vs. true labels.

        Args:
          logits: 2D float Tensor [N, nb_classes]
          labels: 1D int Tensor  [N]

        Returns:
          A 2D int Tensor [nb_classes, nb_classes]
        """
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        return tf.compat.v1.confusion_matrix(labels, preds)

    ###########################
    # Adapted from tkipf/gcn  #
    ###########################

    @staticmethod
    def masked_softmax_cross_entropy(logits, labels, mask):
        """
        Softmax cross‐entropy loss with masking.

        Args:
          logits: 2D float Tensor [N, nb_classes]
          labels: 2D float Tensor [N, nb_classes] (one-hot)
          mask:   1D int/float Tensor [N] (0 or 1)

        Returns:
          Scalar loss (float)
        """
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)  # reweight so that sum(mask)/N = 1
        loss *= mask
        return tf.reduce_mean(loss)

    @staticmethod
    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """
        Sigmoid cross‐entropy (for multi-label) with masking.

        Args:
          logits: Tensor [N, num_outputs]
          labels: Tensor [N, num_outputs]
          mask:   1D Tensor [N]

        Returns:
          Scalar loss (float)
        """
        labels = tf.cast(labels, dtype=tf.float32)
        loss_per_row = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss_per_row = tf.reduce_mean(loss_per_row, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss_per_row *= mask
        return tf.reduce_mean(loss_per_row)

    @staticmethod
    def masked_accuracy(logits, labels, mask):
        """
        Classification accuracy with masking.

        Args:
          logits: 2D float Tensor [N, nb_classes]
          labels: 2D float Tensor [N, nb_classes] (one-hot)
          mask:   1D int/float Tensor [N]

        Returns:
          Scalar accuracy (float)
        """
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc_all = tf.cast(correct_pred, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        acc_all *= mask
        return tf.reduce_mean(acc_all)

    @staticmethod
    def micro_f1(logits, labels, mask):
        """
        Micro‐F1 for multi-label (sigmoid) predictions with masking.

        Args:
          logits: 2D float Tensor [N, num_outputs]
          labels: 2D int Tensor  [N, num_outputs]
          mask:   1D int Tensor  [N]

        Returns:
          Scalar F1 score (float)
        """
        # Compute predicted labels: round sigmoid(logits)
        probs = tf.nn.sigmoid(logits)
        predicted = tf.cast(tf.round(probs), tf.int32)
        labels = tf.cast(labels, tf.int32)
        mask = tf.cast(mask, tf.int32)
        mask = tf.expand_dims(mask, -1)  # shape → [N, 1] for broadcasting

        # True positives, true negatives, false positives, false negatives
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        precision = tf.cast(tp, tf.float32) / (tf.cast(tp + fp, tf.float32) + 1e-8)
        recall    = tf.cast(tp, tf.float32) / (tf.cast(tp + fn, tf.float32) + 1e-8)
        fmeasure  = 2 * precision * recall / (precision + recall + 1e-8)
        return tf.cast(fmeasure, tf.float32)
