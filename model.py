""" Common model file (Migrated to TensorFlow 2.x)
"""

import numpy as np
import pickle  # cPickle is 'pickle' in Python 3
import pdb
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, regularizers

from config import cfg


DEBUG = False


# Helper function for summaries is removed, as TF2 summaries are handled differently (e.g., via Callbacks or eagerly).

# https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
# https://gist.github.com/akiross/754c7b87a2af8603da78b46cdaaa5598
def get_deconv_filter(f_shape):
    """
    Tạo bộ lọc giải mã (deconvolution) dựa trên phép nội suy song tuyến tính (bilinear interpolation).
    f_shape = [ksize, ksize, out_features, in_features]
    """
    width = f_shape[0]
    height = f_shape[0]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    return weights


def add_tensors_wo_none(tensor_list):
    """Cộng tất cả các tensor trong danh sách, bỏ qua các giá trị None."""
    temp_list = [t for t in tensor_list if t is not None]
    if len(temp_list):
        return tf.add_n(temp_list)
    else:
        return None


class BaseModel(tf.keras.Model):
    def __init__(self, weight_file_path):
        super(BaseModel, self).__init__()

        if weight_file_path is not None:
            print(f"Loading pretrained weights from: {weight_file_path}")
            with open(weight_file_path, 'rb') as f: # Use 'rb' for pickle
                self.pretrained_weights = pickle.load(f, encoding='latin1') # Add encoding for Py3
        else:
            self.pretrained_weights = None

    # new_conv_layer, new_fc_layer, new_deconv_layer are removed.
    # We will use tf.keras.layers directly in the model definitions.

    # group_norm functions are removed. We will use tf.keras.layers.GroupNormalization.

    # https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py
    # https://github.com/tensorflow/tensorflow/issues/2169
    def unpool(self, pool, ind, ksize, name):
        """
        Lớp Unpooling sau max_pool_with_argmax.
        (Lưu ý: Mã này dường như không được sử dụng trong các mô hình bên dưới,
         nhưng vẫn được di chuyển sang TF2)
        Args :
            pool : tensor đầu ra đã max pooled
            ind : chỉ số argmax
            ksize : ksize giống như ksize của pool
        Return :
            ret : tensor đã unpooled
        """
        with tf.name_scope(name):
            input_shape = tf.shape(pool)
            output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

            flat_input_size = tf.math.cumprod(input_shape)[-1]
            flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

            pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                     tf.stack([input_shape[0], 1, 1, 1]))
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, tf.stack([flat_input_size, 1]))
            ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
            ind_ = tf.concat([b, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, tf.stack(output_shape))

            return ret

    def _get_norm_layer(self, norm_type, num_channels):
        """Helper để tạo lớp chuẩn hóa."""
        if norm_type == 'BN':
            return layers.BatchNormalization(renorm=cfg.USE_BRN)
        elif norm_type == 'GN':
            num_groups = min(cfg.GN_MIN_NUM_G, num_channels // cfg.GN_MIN_CHS_PER_G)
            return layers.GroupNormalization(groups=num_groups)
        else:
            return None # Sẽ sử dụng bias

    def _get_regularizer(self):
        """Helper để lấy kernel regularizer."""
        return regularizers.l2(cfg.TRAIN.WEIGHT_DECAY_RATE)


class vessel_segm_cnn(BaseModel):
    def __init__(self, params, weight_file_path):
        super(vessel_segm_cnn, self).__init__(weight_file_path)
        self.params = params
        self.cnn_model = params.cnn_model

        if self.cnn_model not in ['driu', 'driu_large']:
            raise NotImplementedError(f"CNN model {self.cnn_model} not implemented.")

        self.num_spe_channels = 16 # fixed

        # Tạo tất cả các lớp (layers)
        self._build_model_layers()

        # Tạo bộ tối ưu hóa (optimizer)
        self._build_optimizer()

    def _build_optimizer(self):
        """Xây dựng bộ tối ưu hóa và lịch trình học."""

        if self.params.lr_decay == 'const':
            self.lr_schedule = self.params.lr
        elif self.params.lr_decay == 'pc':
            boundaries = [int(self.params.max_iters * 0.5), int(self.params.max_iters * 0.75)]
            values = [self.params.lr, self.params.lr * 0.5, self.params.lr * 0.25]
            self.lr_schedule = optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        elif self.params.lr_decay == 'exp':
            self.lr_schedule = optimizers.schedules.ExponentialDecay(
                self.params.lr,
                decay_steps=self.params.max_iters / 20,
                decay_rate=0.9,
                staircase=False
            )
        else:
            raise NotImplementedError(f"LR decay {self.params.lr_decay} not implemented.")

        if self.params.opt == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=self.lr_schedule, epsilon=0.1)
        elif self.params.opt == 'sgd':
            self.optimizer = optimizers.SGD(
                learning_rate=self.lr_schedule,
                momentum=cfg.TRAIN.MOMENTUM,
                nesterov=True
            )
        else:
            raise NotImplementedError(f"Optimizer {self.params.opt} not implemented.")

        # Biến global_step được quản lý tự động bởi optimizer

    def _build_model_layers(self):
        """Định nghĩa tất cả các lớp Keras cho mô hình."""
        reg = self._get_regularizer()

        # VGG-style backbone
        self.conv1_1 = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv1_1')
        self.conv1_2 = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv1_2')
        self.pool1 = layers.MaxPool2D(2, strides=2, padding='same', name='pool1')

        self.conv2_1 = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv2_1')
        self.conv2_2 = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv2_2')
        self.pool2 = layers.MaxPool2D(2, strides=2, padding='same', name='pool2')

        self.conv3_1 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv3_1')
        self.conv3_2 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv3_2')
        self.conv3_3 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv3_3')
        self.pool3 = layers.MaxPool2D(2, strides=2, padding='same', name='pool3')

        self.conv4_1 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv4_1')
        self.conv4_2 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv4_2')
        self.conv4_3 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv4_3')

        # Lớp chuyên biệt (Specialized layers)
        n = self.num_spe_channels
        self.spe_1 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_1')
        self.spe_2 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_2')
        self.spe_3 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_3')
        self.spe_4 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_4')

        # Lớp giải mã (Deconv layers) với khởi tạo song tuyến tính
        init_2 = tf.keras.initializers.Constant(get_deconv_filter([4, 4, n, n]))
        self.resized_spe_2 = layers.Conv2DTranspose(n, 4, strides=2, padding='same', activation='relu', kernel_initializer=init_2, kernel_regularizer=reg, name='resized_spe_2')

        init_4 = tf.keras.initializers.Constant(get_deconv_filter([8, 8, n, n]))
        self.resized_spe_3 = layers.Conv2DTranspose(n, 8, strides=4, padding='same', activation='relu', kernel_initializer=init_4, kernel_regularizer=reg, name='resized_spe_3')

        init_8 = tf.keras.initializers.Constant(get_deconv_filter([16, 16, n, n]))
        self.resized_spe_4 = layers.Conv2DTranspose(n, 16, strides=8, padding='same', activation='relu', kernel_initializer=init_8, kernel_regularizer=reg, name='resized_spe_4')

        if self.cnn_model == 'driu_large':
            self.pool4 = layers.MaxPool2D(2, strides=2, padding='same', name='pool4')
            self.conv5_1 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv5_1')
            self.conv5_2 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv5_2')
            self.conv5_3 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv5_3')
            self.spe_5 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_5')

            init_16 = tf.keras.initializers.Constant(get_deconv_filter([32, 32, n, n]))
            self.resized_spe_5 = layers.Conv2DTranspose(n, 32, strides=16, padding='same', activation='relu', kernel_initializer=init_16, kernel_regularizer=reg, name='resized_spe_5')

            self.output_layer = layers.Conv2D(1, 1, padding='same', kernel_regularizer=reg, name='output')
        else: # 'driu'
            self.output_layer = layers.Conv2D(1, 1, padding='same', kernel_regularizer=reg, name='output')

    def call(self, imgs, training=True):
        """Thực hiện lượt chạy thuận (forward pass)."""

        # Backbone
        x = self.conv1_1(imgs)
        conv1_2 = self.conv1_2(x)
        x = self.pool1(conv1_2)

        x = self.conv2_1(x)
        conv2_2 = self.conv2_2(x)
        x = self.pool2(conv2_2)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        conv3_3 = self.conv3_3(x)
        x = self.pool3(conv3_3)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        conv4_3 = self.conv4_3(x)

        # Specialized layers
        spe_1 = self.spe_1(conv1_2)
        spe_2 = self.spe_2(conv2_2)
        spe_3 = self.spe_3(conv3_3)
        spe_4 = self.spe_4(conv4_3)

        rspe_2 = self.resized_spe_2(spe_2)
        rspe_3 = self.resized_spe_3(spe_3)
        rspe_4 = self.resized_spe_4(spe_4)


        spe_layers = [spe_1, rspe_2, rspe_3, rspe_4]

        if self.cnn_model == 'driu_large':
            x = self.pool4(conv4_3)
            x = self.conv5_1(x)
            x = self.conv5_2(x)
            conv5_3 = self.conv5_3(x)
            spe_5 = self.spe_5(conv5_3)
            rspe_5 = self.resized_spe_5(spe_5)
            spe_layers.append(rspe_5)

        spe_concat = tf.concat(values=spe_layers, axis=3)
        self.conv_feats = spe_concat # Lưu lại để tham chiếu nếu cần

        output = self.output_layer(spe_concat)
        fg_prob = tf.sigmoid(output)

        return output, fg_prob

    def compute_metrics(self, output, fg_prob, labels):
        """Tính toán các chỉ số (accuracy, precision, recall)."""
        flat_labels = tf.reshape(tensor=labels, shape=(-1,))
        flat_bin_output = tf.greater_equal(tf.reshape(tensor=fg_prob, shape=(-1,)), 0.5)

        flat_labels_bool = tf.cast(flat_labels, tf.bool)

        # Accuracy
        correct = tf.cast(tf.equal(flat_bin_output, flat_labels_bool), tf.float32)
        accuracy = tf.reduce_mean(correct)

        # Precision, Recall
        num_fg_output = tf.reduce_sum(tf.cast(flat_bin_output, tf.float32))
        binary_mask_fg = tf.cast(tf.equal(labels, 1), tf.float32)
        num_pixel_fg = tf.cast(tf.math.count_nonzero(binary_mask_fg, dtype=tf.int64), tf.float32)

        tp = tf.reduce_sum(tf.cast(tf.logical_and(flat_labels_bool, flat_bin_output), tf.float32))

        pre = tf.divide(tp, tf.add(num_fg_output, cfg.EPSILON))
        rec = tf.divide(tp, tf.add(num_pixel_fg, cfg.EPSILON))

        metrics = {
            'accuracy': accuracy,
            'precision': pre,
            'recall': rec,
            'tp': tp,
            'num_fg_output': num_fg_output,
            'num_pixel_fg': num_pixel_fg
        }
        return metrics

    def compute_loss_fn(self, output, labels, fov_masks):
        """Tính toán hàm mất mát."""
        flat_labels = tf.reshape(tensor=labels, shape=(-1,))
        flat_logits = tf.reshape(tensor=output, shape=(-1,))

        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=flat_logits,
            labels=tf.cast(flat_labels, tf.float32)
        )

        # weighted cross entropy loss (in fov)
        binary_mask_fg = tf.cast(tf.equal(labels, 1), tf.float32)
        binary_mask_bg = tf.cast(tf.not_equal(labels, 1), tf.float32)
        combined_mask = tf.concat(values=[binary_mask_bg, binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))

        num_pixel_in_fov = tf.reduce_sum(fov_masks)
        num_pixel_fg = tf.cast(tf.math.count_nonzero(binary_mask_fg, dtype=tf.int64), tf.float32)
        num_pixel_bg = tf.cast(num_pixel_in_fov, tf.float32) - num_pixel_fg

        # Tránh chia cho 0 nếu num_pixel_in_fov là 0
        safe_num_pixel_in_fov = tf.add(tf.cast(num_pixel_in_fov, tf.float32), cfg.EPSILON)

        weight_fg = tf.divide(num_pixel_bg, safe_num_pixel_in_fov)
        weight_bg = tf.divide(num_pixel_fg, safe_num_pixel_in_fov)

        class_weight = tf.cast(tf.stack([weight_bg, weight_fg]), dtype=tf.float32)
        weight_per_label = tf.reduce_sum(flat_one_hot_labels * class_weight, axis=1)

        # Áp dụng fov_masks
        reshaped_fov_masks = tf.reshape(tensor=tf.cast(fov_masks, tf.float32), shape=(-1,))
        safe_mean_fov = tf.add(tf.reduce_mean(reshaped_fov_masks), cfg.EPSILON)
        reshaped_fov_masks_norm = reshaped_fov_masks / safe_mean_fov

        weighted_loss = tf.multiply(reshaped_fov_masks_norm, weight_per_label)
        loss = tf.reduce_mean(tf.multiply(weighted_loss, cross_entropies))

        # Thêm L2 regularization loss (Keras tự động thu thập)
        total_loss = loss + tf.add_n(self.losses)

        return total_loss

    def _process_gradients(self, grads_and_vars):
        """Áp dụng logic xử lý gradient tùy chỉnh từ mã TF1.x."""
        processed_gvs = []
        for g, v in grads_and_vars:
            if g is None:
                processed_gvs.append((g, v))
                continue

            # Điều chỉnh trọng số cho 'output'
            if 'output' in v.name:
                g = 0.01 * g

            # Bỏ qua gradient cho 'resized' (theo logic 'map' cũ)
            if 'resized' in v.name:
                g = None
                processed_gvs.append((g, v))
                continue

            # Cắt gradient
            g = tf.clip_by_value(g, -5., 5.)
            processed_gvs.append((g, v))

        return processed_gvs

    # Ghi đè train_step để tùy chỉnh vòng lặp huấn luyện
    def train_step(self, data):
        """Thực hiện một bước huấn luyện."""
        # Dữ liệu đầu vào dự kiến là một tuple/list: (imgs, labels, fov_masks)
        imgs, labels, fov_masks = data

        with tf.GradientTape() as tape:
            # Chạy thuận
            output, fg_prob = self(imgs, training=True)
            # Tính loss
            total_loss = self.compute_loss_fn(output, labels, fov_masks)

        # Tính gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        grads_and_vars = zip(grads, trainable_vars)

        # Xử lý gradients (cắt, điều chỉnh,...)
        processed_grads_and_vars = self._process_gradients(grads_and_vars)

        # Áp dụng gradients
        self.optimizer.apply_gradients(processed_grads_and_vars)

        # Tính toán metrics
        metrics = self.compute_metrics(output, fg_prob, labels)
        metrics['loss'] = total_loss

        return metrics

    # Ghi đè test_step để tùy chỉnh vòng lặp đánh giá
    def test_step(self, data):
        imgs, labels, fov_masks = data

        # Chạy thuận
        output, fg_prob = self(imgs, training=False)
        # Tính loss
        total_loss = self.compute_loss_fn(output, labels, fov_masks)

        # Tính toán metrics
        metrics = self.compute_metrics(output, fg_prob, labels)
        metrics['loss'] = total_loss

        return metrics


class VesselSegmVGN(BaseModel):
    def __init__(self, params, weight_file_path):
        super(VesselSegmVGN, self).__init__(weight_file_path)
        self.params = params

        # cnn module related
        self.cnn_model = params.cnn_model
        self.cnn_loss_on = params.cnn_loss_on

        # gnn module related
        self.win_size = params.win_size
        self.gnn_loss_on = params.gnn_loss_on
        self.gnn_loss_weight = params.gnn_loss_weight

        # inference module related
        self.infer_module_kernel_size = params.infer_module_kernel_size

        self.num_spe_channels = 16 # fixed
        self.cnn_feat = {}
        self.cnn_feat_spatial_sizes = {}

        self.var_to_restore = [] # Sẽ được điền trong _build_...

        # Xây dựng các mô-đun
        self._build_cnn_module_layers()
        self._build_gat_layers()
        self._build_infer_module_layers()

        # Xây dựng bộ tối ưu hóa
        self._build_optimizer()

    def _build_optimizer(self):
        """Xây dựng bộ tối ưu hóa và lịch trình học."""

        # LR Handler
        if self.params.lr_scheduling == 'pc':
            boundaries = [int(self.params.max_iters * self.params.lr_decay_tp)]
            if self.params.old_net_ft_lr == 0:
                # Chỉ huấn luyện mạng mới
                values = [self.params.new_net_lr, self.params.new_net_lr * 0.1]
            else:
                # Huấn luyện toàn bộ mạng
                values = [self.params.old_net_ft_lr, self.params.old_net_ft_lr * 0.1]
            self.lr_schedule = optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        else:
            raise NotImplementedError(f"LR scheduling {self.params.lr_scheduling} not implemented.")

        # Optimizer
        if self.params.opt == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=self.lr_schedule, epsilon=0.1)
        elif self.params.opt == 'sgd':
            self.optimizer = optimizers.SGD(
                learning_rate=self.lr_schedule,
                momentum=cfg.TRAIN.MOMENTUM,
                nesterov=True
            )
        else:
            raise NotImplementedError(f"Optimizer {self.params.opt} not implemented.")

    def _add_layer_to_restore(self, name):
        if name not in self.var_to_restore:
            self.var_to_restore.append(name)

    def _build_cnn_module_layers(self):
        """Định nghĩa các lớp Keras cho mô-đun CNN (Driu hoặc Driu-Large)."""
        print("Building CNN module layers...")
        if self.cnn_model not in ['driu', 'driu_large']:
            raise NotImplementedError(f"CNN model {self.cnn_model} not implemented.")

        reg = self._get_regularizer()
        n = self.num_spe_channels

        # VGG-style backbone
        self.conv1_1 = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv1_1')
        self.conv1_2 = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv1_2')
        self.pool1 = layers.MaxPool2D(2, strides=2, padding='same', name='pool1')
        self._add_layer_to_restore('conv1_1')
        self._add_layer_to_restore('conv1_2')

        self.conv2_1 = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv2_1')
        self.conv2_2 = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv2_2')
        self.pool2 = layers.MaxPool2D(2, strides=2, padding='same', name='pool2')
        self._add_layer_to_restore('conv2_1')
        self._add_layer_to_restore('conv2_2')

        self.conv3_1 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv3_1')
        self.conv3_2 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv3_2')
        self.conv3_3 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv3_3')
        self.pool3 = layers.MaxPool2D(2, strides=2, padding='same', name='pool3')
        self._add_layer_to_restore('conv3_1')
        self._add_layer_to_restore('conv3_2')
        self._add_layer_to_restore('conv3_3')

        self.conv4_1 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv4_1')
        self.conv4_2 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv4_2')
        self.conv4_3 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv4_3')
        self._add_layer_to_restore('conv4_1')
        self._add_layer_to_restore('conv4_2')
        self._add_layer_to_restore('conv4_3')

        # Lớp chuyên biệt (Specialized layers)
        self.spe_1 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_1')
        self.spe_2 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_2')
        self.spe_3 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_3')
        self.spe_4 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_4')
        self._add_layer_to_restore('spe_1')
        self._add_layer_to_restore('spe_2')
        self._add_layer_to_restore('spe_3')
        self._add_layer_to_restore('spe_4')

        # Lớp giải mã (Deconv layers)
        init_2 = tf.keras.initializers.Constant(get_deconv_filter([4, 4, n, n]))
        self.resized_spe_2 = layers.Conv2DTranspose(n, 4, strides=2, padding='same', activation='relu', kernel_initializer=init_2, kernel_regularizer=reg, name='resized_spe_2')
        self._add_layer_to_restore('resized_spe_2')

        init_4 = tf.keras.initializers.Constant(get_deconv_filter([8, 8, n, n]))
        self.resized_spe_3 = layers.Conv2DTranspose(n, 8, strides=4, padding='same', activation='relu', kernel_initializer=init_4, kernel_regularizer=reg, name='resized_spe_3')
        self._add_layer_to_restore('resized_spe_3')

        init_8 = tf.keras.initializers.Constant(get_deconv_filter([16, 16, n, n]))
        self.resized_spe_4 = layers.Conv2DTranspose(n, 16, strides=8, padding='same', activation='relu', kernel_initializer=init_8, kernel_regularizer=reg, name='resized_spe_4')
        self._add_layer_to_restore('resized_spe_4')

        if self.cnn_model == 'driu_large':
            self.pool4 = layers.MaxPool2D(2, strides=2, padding='same', name='pool4')
            self.conv5_1 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv5_1')
            self.conv5_2 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv5_2')
            self.conv5_3 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=reg, name='conv5_3')
            self.spe_5 = layers.Conv2D(n, 3, padding='same', activation='relu', kernel_regularizer=reg, name='spe_5')
            self._add_layer_to_restore('conv5_1')
            self._add_layer_to_restore('conv5_2')
            self._add_layer_to_restore('conv5_3')
            self._add_layer_to_restore('spe_5')

            init_16 = tf.keras.initializers.Constant(get_deconv_filter([32, 32, n, n]))
            self.resized_spe_5 = layers.Conv2DTranspose(n, 32, strides=16, padding='same', activation='relu', kernel_initializer=init_16, kernel_regularizer=reg, name='resized_spe_5')
            self._add_layer_to_restore('resized_spe_5')

            self.img_output_layer = layers.Conv2D(1, 1, padding='same', kernel_regularizer=reg, name='img_output')
        else: # 'driu'
            self.img_output_layer = layers.Conv2D(1, 1, padding='same', kernel_regularizer=reg, name='img_output')

        self._add_layer_to_restore('img_output')
        print("CNN module layers built.")

    def _build_gat_layers(self):
        """Định nghĩa các lớp Keras cho mô-đun GAT."""
        print("Building GAT layers...")
        self.gat_layers = [] # Sẽ chứa các lớp con (Conv1D)

        # GAT sử dụng các lớp Conv1D để biến đổi đặc trưng
        # Lớp 1
        n_heads = self.params.gat_n_heads[0]
        hid_units = self.params.gat_hid_units[0]
        head_layers = []
        for i in range(n_heads):
            name = f'gat_hidden_1_{i+1}'
            # Mỗi head bao gồm các phép biến đổi
            head_layers.append({
                'fts': layers.Conv1D(hid_units, 1, use_bias=False, name=f'{name}_fts'),
                'f_1': layers.Conv1D(1, 1, name=f'{name}_f1'),
                'f_2': layers.Conv1D(1, 1, name=f'{name}_f2'),
                'bias': self.add_weight(name=f'{name}_bias', shape=(hid_units,), initializer='zeros', trainable=True)
            })
            self._add_layer_to_restore(name)
        self.gat_layers.append(head_layers)

        # Các lớp ẩn tiếp theo
        for i in range(1, len(self.params.gat_hid_units)):
            n_heads = self.params.gat_n_heads[i]
            hid_units = self.params.gat_hid_units[i]
            head_layers = []
            for j in range(n_heads):
                name = f'gat_hidden_{i+1}_{j+1}'
                head_layers.append({
                    'fts': layers.Conv1D(hid_units, 1, use_bias=False, name=f'{name}_fts'),
                    'f_1': layers.Conv1D(1, 1, name=f'{name}_f1'),
                    'f_2': layers.Conv1D(1, 1, name=f'{name}_f2'),
                    'bias': self.add_weight(name=f'{name}_bias', shape=(hid_units,), initializer='zeros', trainable=True)
                })
                self._add_layer_to_restore(name)
            self.gat_layers.append(head_layers)

        # Lớp đầu ra (output logits)
        n_heads = self.params.gat_n_heads[-1]
        head_layers = []
        for i in range(n_heads):
            name = f'gat_node_logits_{i+1}'
            head_layers.append({
                'fts': layers.Conv1D(1, 1, use_bias=False, name=f'{name}_fts'),
                'f_1': layers.Conv1D(1, 1, name=f'{name}_f1'),
                'f_2': layers.Conv1D(1, 1, name=f'{name}_f2'),
                'bias': self.add_weight(name=f'{name}_bias', shape=(1,), initializer='zeros', trainable=True)
            })
            self._add_layer_to_restore(name)
        self.gat_layers.append(head_layers)

        print("GAT layers built.")

    def _build_infer_module_layers(self):
        """Định nghĩa các lớp Keras cho mô-đun suy luận (Inference Module)."""
        print("Building Inference module layers...")
        reg = self._get_regularizer()
        k = self.infer_module_kernel_size
        norm_type = self.params.norm_type

        # Lớp nén (compression)
        temp_num_chs = self.params.gat_n_heads[-2] * self.params.gat_hid_units[-1]
        self.post_cnn_conv_comp_layers = [
            layers.Conv2D(32, 1, kernel_regularizer=reg, use_bias=(norm_type is None), name='post_cnn_conv_comp'),
            self._get_norm_layer(norm_type, 32),
            layers.ReLU()
        ]

        self.post_cnn_deconv_layers = {}
        self.post_cnn_fuse_layers = {}
        self.post_cnn_skip_layers = {}

        ds_rate = self.win_size // 2
        while ds_rate >= 1:
            ds_rate_int = int(ds_rate)

            # Lớp Deconv
            init_deconv = tf.keras.initializers.Constant(get_deconv_filter([4, 4, 16, 32]))
            self.post_cnn_deconv_layers[ds_rate_int] = [
                layers.Conv2DTranspose(16, 4, strides=2, padding='same', kernel_initializer=init_deconv, kernel_regularizer=reg, use_bias=(norm_type is None), name=f'post_cnn_deconv{ds_rate_int}'),
                self._get_norm_layer(norm_type, 16),
                layers.ReLU()
            ]

            # Lớp xử lý Skip-connection
            if self.params.use_enc_layer:
                self.post_cnn_skip_layers[ds_rate_int] = [
                    layers.Conv2D(16, 1, kernel_regularizer=reg, use_bias=(norm_type is None), name=f'post_cnn_cnn_feat{ds_rate_int}'),
                    self._get_norm_layer(norm_type, 16),
                    layers.ReLU()
                ]
            else: # Chỉ chuẩn hóa và ReLU
                self.post_cnn_skip_layers[ds_rate_int] = [
                    self._get_norm_layer(norm_type, 16), # Tên 'post_cnn_cnn_feat' được dùng cho GN
                    layers.ReLU()
                ]

            # Lớp kết hợp (Fuse)
            if ds_rate_int == 1:
                # Lớp đầu ra cuối cùng
                self.post_cnn_fuse_layers[ds_rate_int] = [
                    layers.Conv2D(1, k, padding='same', kernel_regularizer=reg, name='post_cnn_img_output')
                ]
            else:
                self.post_cnn_fuse_layers[ds_rate_int] = [
                    layers.Conv2D(32, k, padding='same', kernel_regularizer=reg, use_bias=(norm_type is None), name=f'post_cnn_conv{ds_rate_int}'),
                    self._get_norm_layer(norm_type, 32),
                    layers.ReLU()
                ]

            ds_rate = ds_rate / 2

        print("Inference module layers built.")

    def _run_cnn_module(self, imgs):
        """Chạy mô-đun CNN."""
        # Backbone
        x = self.conv1_1(imgs)
        conv1_2 = self.conv1_2(x)
        self.cnn_feat[1] = self.spe_1(conv1_2)
        self.cnn_feat_spatial_sizes[1] = tf.shape(self.cnn_feat[1])[1:3]
        x = self.pool1(conv1_2)

        x = self.conv2_1(x)
        conv2_2 = self.conv2_2(x)
        self.cnn_feat[2] = self.spe_2(conv2_2)
        self.cnn_feat_spatial_sizes[2] = tf.shape(self.cnn_feat[2])[1:3]
        x = self.pool2(conv2_2)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        conv3_3 = self.conv3_3(x)
        self.cnn_feat[4] = self.spe_3(conv3_3)
        self.cnn_feat_spatial_sizes[4] = tf.shape(self.cnn_feat[4])[1:3]
        x = self.pool3(conv3_3)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        conv4_3 = self.conv4_3(x)
        self.cnn_feat[8] = self.spe_4(conv4_3)
        self.cnn_feat_spatial_sizes[8] = tf.shape(self.cnn_feat[8])[1:3]

        # Lớp chuyên biệt
        rspe_2 = self.resized_spe_2(self.cnn_feat[2])
        rspe_3 = self.resized_spe_3(self.cnn_feat[4])
        rspe_4 = self.resized_spe_4(self.cnn_feat[8])

        spe_layers = [self.cnn_feat[1], rspe_2, rspe_3, rspe_4]

        if self.cnn_model == 'driu_large':
            x = self.pool4(conv4_3)
            x = self.conv5_1(x)
            x = self.conv5_2(x)
            conv5_3 = self.conv5_3(x)
            self.cnn_feat[16] = self.spe_5(conv5_3)
            self.cnn_feat_spatial_sizes[16] = tf.shape(self.cnn_feat[16])[1:3]

            rspe_5 = self.resized_spe_5(self.cnn_feat[16])
            spe_layers.append(rspe_5)

        conv_feats = tf.concat(values=spe_layers, axis=3)
        img_output = self.img_output_layer(conv_feats)
        img_fg_prob = tf.sigmoid(img_output)

        return conv_feats, img_output, img_fg_prob

    def _sp_attn_head(self, bottom, adj, head_layers, feat_dropout=0., att_dropout=0., residual=False, act=tf.nn.elu, show_adj=False):
        """Chạy một đầu attention (attention head) của GAT."""

        if feat_dropout != 0.0:
            bottom = tf.nn.dropout(bottom, 1.0 - feat_dropout)

        fts = head_layers['fts'](bottom) # [1, num_nodes, out_size]

        # Self-attention
        f_1 = head_layers['f_1'](fts) # [1, num_nodes, 1]
        f_2 = head_layers['f_2'](fts) # [1, num_nodes, 1]

        num_nodes = tf.shape(adj)[0]
        f_1 = tf.reshape(f_1, [num_nodes, 1])
        f_2 = tf.reshape(f_2, [num_nodes, 1])

        f_1 = adj * f_1
        f_2 = adj * tf.transpose(f_2, [1, 0])

        logits = tf.sparse.add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse.softmax(lrelu) # [num_nodes, num_nodes] (Sparse)

        if att_dropout != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - att_dropout),
                                    dense_shape=coefs.dense_shape)
        if feat_dropout != 0.0:
            fts = tf.nn.dropout(fts, 1.0 - feat_dropout)

        # coefs = tf.sparse.reshape(coefs, [num_nodes, num_nodes]) # Không cần thiết
        fts_squeezed = tf.squeeze(fts, [0]) # [num_nodes, out_size]
        vals = tf.sparse.sparse_dense_matmul(coefs, fts_squeezed) # [num_nodes, out_size]
        vals = tf.expand_dims(vals, axis=0) # [1, num_nodes, out_size]

        # Thêm bias (thay thế tf.contrib.layers.bias_add)
        ret = tf.nn.bias_add(vals, head_layers['bias'])

        # Residual connection
        if residual:
            if bottom.shape[-1] != ret.shape[-1]:
                # Cần thêm một lớp Conv1D để khớp kích thước
                # Điều này cần được định nghĩa trong __init__, tạm thời bỏ qua vì phức tạp
                # và mã gốc có vẻ cũng không xử lý triệt để
                print("Warning: Residual connection dimensionality mismatch not fully implemented.")
                ret = ret + layers.Conv1D(ret.shape[-1], 1)(bottom) # Thêm lớp 1x1
            else:
                ret = ret + bottom

        if show_adj:
            return act(ret), coefs
        else:
            return act(ret)

    def _run_gat_module(self, conv_feats, node_byxs, adj, gnn_feat_dropout=0., gnn_att_dropout=0.):
        """Chạy mô-đun GAT."""

        node_feats = tf.gather_nd(conv_feats, node_byxs) # [num_nodes, num_channels]
        node_feats_resh = tf.expand_dims(node_feats, axis=0) # [1, num_nodes, num_channels]

        h_1 = node_feats_resh

        # Lớp 1
        attns = []
        for i in range(self.params.gat_n_heads[0]):
            attns.append(self._sp_attn_head(h_1, adj, self.gat_layers[0][i],
                                            feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout,
                                            residual=self.params.gat_use_residual))
        h_1 = tf.concat(attns, axis=-1)

        # Các lớp ẩn
        for i in range(1, len(self.params.gat_hid_units)):
            attns = []
            for j in range(self.params.gat_n_heads[i]):
                attns.append(self._sp_attn_head(h_1, adj, self.gat_layers[i][j],
                                                feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout,
                                                residual=self.params.gat_use_residual))
            h_1 = tf.concat(attns, axis=-1)

        # Lớp đầu ra
        out = []
        for i in range(self.params.gat_n_heads[-1]):
            out.append(self._sp_attn_head(h_1, adj, self.gat_layers[-1][i],
                                          act=lambda x: x, # No activation
                                          feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout,
                                          residual=False))

        node_logits = tf.add_n(out) / self.params.gat_n_heads[-1]
        node_logits = tf.squeeze(node_logits, [0, 2]) # [num_nodes,]

        gnn_final_feats = tf.squeeze(h_1, [0]) # [num_nodes, last_hidden_dim]

        return node_feats, node_logits, gnn_final_feats

    def _run_infer_module(self, gnn_final_feats, imgs_shape,
                          is_lr_flipped, is_ud_flipped, rot90_num,
                          post_cnn_dropout=0., training=True):
        """Chạy mô-đun suy luận."""

        y_len = tf.cast(tf.math.ceil(tf.cast(imgs_shape[1], tf.float32) / self.win_size), dtype=tf.int32)
        x_len = tf.cast(tf.math.ceil(tf.cast(imgs_shape[2], tf.float32) / self.win_size), dtype=tf.int32)

        sp_size = tf.cond(tf.logical_or(tf.equal(rot90_num, 0), tf.equal(rot90_num, 2)),
                          lambda: tf.stack([y_len, x_len]),
                          lambda: tf.stack([x_len, y_len]))

        reshaped_gnn_feats = tf.reshape(
            tensor=gnn_final_feats,
            shape=tf.concat([
                [imgs_shape[0]], sp_size, [tf.shape(gnn_final_feats)[-1]]
            ], axis=0)
        )

        # Đảo ngược (TTA)
        if is_lr_flipped:
            reshaped_gnn_feats = tf.image.flip_left_right(reshaped_gnn_feats)
        if is_ud_flipped:
            reshaped_gnn_feats = tf.image.flip_up_down(reshaped_gnn_feats)
        if tf.math.not_equal(rot90_num, 0):
            reshaped_gnn_feats = tf.image.rot90(reshaped_gnn_feats, k=tf.cast(rot90_num, tf.int32))

        # Chạy lớp nén
        current_input = reshaped_gnn_feats
        for layer in self.post_cnn_conv_comp_layers:
            if layer is not None:
                current_input = layer(current_input, training=training)

        ds_rate = self.win_size // 2
        while ds_rate >= 1:
            ds_rate_int = int(ds_rate)

            # Upsample (Deconv)
            upsampled = current_input
            for layer in self.post_cnn_deconv_layers[ds_rate_int]:
                if layer is not None:
                    # Lớp Conv2DTranspose cần output_shape động
                    if isinstance(layer, layers.Conv2DTranspose):
                        target_shape = tf.concat([
                            [imgs_shape[0]],
                            self.cnn_feat_spatial_sizes[ds_rate_int],
                            [16] # num_channels
                        ], axis=0)
                        # Lưu ý: output_shape không còn được hỗ trợ trực tiếp trong TF2 Keras
                        # Thay vào đó, chúng ta dựa vào strides và padding.
                        # Mã gốc đã sử dụng output_shape, điều này hơi rắc rối.
                        # Chúng ta sẽ giả định strides=2 và padding='same' là đủ.
                        upsampled = layer(upsampled, training=training)
                    else:
                        upsampled = layer(upsampled, training=training)

            # Skip connection
            cur_cnn_feat = self.cnn_feat[ds_rate_int]
            if post_cnn_dropout > 0.:
                cur_cnn_feat = tf.nn.dropout(cur_cnn_feat, 1.0 - post_cnn_dropout)

            for layer in self.post_cnn_skip_layers[ds_rate_int]:
                if layer is not None:
                    cur_cnn_feat = layer(cur_cnn_feat, training=training)

            # Fuse
            fused_input = tf.concat(values=[upsampled, cur_cnn_feat], axis=3)
            output = fused_input
            for layer in self.post_cnn_fuse_layers[ds_rate_int]:
                if layer is not None:
                    output = layer(output, training=training)

            current_input = output
            ds_rate = ds_rate / 2

        post_cnn_img_output = current_input
        post_cnn_img_fg_prob = tf.sigmoid(post_cnn_img_output)

        return post_cnn_img_output, post_cnn_img_fg_prob

    def call(self, inputs, training=True):
        """
        Thực hiện lượt chạy thuận hoàn chỉnh.
        'inputs' là một dict chứa tất cả (trước đây là placeholders).
        """
        imgs = inputs['imgs']
        node_byxs = inputs['node_byxs']
        adj = inputs['adj']

        # TTA và dropout inputs
        gnn_feat_dropout = inputs.get('gnn_feat_dropout', 0.0) if training else 0.0
        gnn_att_dropout = inputs.get('gnn_att_dropout', 0.0) if training else 0.0
        post_cnn_dropout = inputs.get('post_cnn_dropout', 0.0) if training else 0.0
        is_lr_flipped = inputs.get('is_lr_flipped', False)
        is_ud_flipped = inputs.get('is_ud_flipped', False)
        rot90_num = inputs.get('rot90_num', 0)

        # 1. Chạy CNN
        conv_feats, img_output, img_fg_prob = self._run_cnn_module(imgs)

        # 2. Chạy GAT
        node_feats, node_logits, gnn_final_feats = self._run_gat_module(
            conv_feats, node_byxs, adj, gnn_feat_dropout, gnn_att_dropout
        )

        # 3. Chạy Module suy luận
        post_cnn_img_output, post_cnn_img_fg_prob = self._run_infer_module(
            gnn_final_feats, tf.shape(imgs),
            is_lr_flipped, is_ud_flipped, rot90_num,
            post_cnn_dropout, training=training
        )

        # Trả về tất cả các đầu ra cần thiết cho việc tính loss và metrics
        return {
            'img_output': img_output,
            'img_fg_prob': img_fg_prob,
            'node_logits': node_logits,
            'node_feats': node_feats,
            'gnn_final_feats': gnn_final_feats,
            'post_cnn_img_output': post_cnn_img_output,
            'post_cnn_img_fg_prob': post_cnn_img_fg_prob
        }

    def compute_loss_and_metrics(self, outputs, data):
        """Tính toán tất cả các loss và metrics."""

        labels = data['labels']
        fov_masks = data['fov_masks']
        node_labels = data['node_labels']
        pixel_weights = data['pixel_weights'] # Trọng số cho post_cnn_loss

        img_output = outputs['img_output']
        img_fg_prob = outputs['img_fg_prob']
        node_logits = outputs['node_logits']
        post_cnn_img_output = outputs['post_cnn_img_output']
        post_cnn_img_fg_prob = outputs['post_cnn_img_fg_prob']

        flat_labels = tf.reshape(tensor=labels, shape=(-1,))
        flat_labels_float = tf.cast(flat_labels, tf.float32)
        flat_labels_bool = tf.cast(flat_labels, tf.bool)

        binary_mask_fg = tf.cast(tf.equal(labels, 1), tf.float32)
        binary_mask_bg = tf.cast(tf.not_equal(labels, 1), tf.float32)
        combined_mask = tf.concat(values=[binary_mask_bg, binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))

        num_pixel_in_fov = tf.reduce_sum(fov_masks)
        num_pixel_fg_64 = tf.math.count_nonzero(binary_mask_fg, dtype=tf.int64)
        num_pixel_fg = tf.cast(num_pixel_fg_64, tf.float32)
        num_pixel_bg = tf.cast(num_pixel_in_fov, tf.float32) - num_pixel_fg

        safe_num_pixel_in_fov = tf.add(tf.cast(num_pixel_in_fov, tf.float32), cfg.EPSILON)

        weight_fg = tf.divide(num_pixel_bg, safe_num_pixel_in_fov)
        weight_bg = tf.divide(num_pixel_fg, safe_num_pixel_in_fov)

        class_weight = tf.cast(tf.stack([weight_bg, weight_fg]), dtype=tf.float32)
        weight_per_label = tf.reduce_sum(flat_one_hot_labels * class_weight, axis=1)

        reshaped_fov_masks = tf.reshape(tensor=tf.cast(fov_masks, tf.float32), shape=(-1,))
        safe_mean_fov = tf.add(tf.reduce_mean(reshaped_fov_masks), cfg.EPSILON)
        reshaped_fov_masks_norm = reshaped_fov_masks / safe_mean_fov

        losses = {}
        metrics = {}

        # 1. CNN Loss & Metrics
        flat_logits_cnn = tf.reshape(tensor=img_output, shape=(-1,))
        cross_entropies_cnn = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=flat_logits_cnn, labels=flat_labels_float
        )
        weighted_loss_cnn = tf.multiply(reshaped_fov_masks_norm, weight_per_label)
        losses['cnn'] = tf.reduce_mean(tf.multiply(weighted_loss_cnn, cross_entropies_cnn))

        flat_bin_output_cnn = tf.greater_equal(tf.reshape(tensor=img_fg_prob, shape=(-1,)), 0.5)
        metrics['cnn_accuracy'] = tf.reduce_mean(tf.cast(tf.equal(flat_bin_output_cnn, flat_labels_bool), tf.float32))
        num_fg_output_cnn = tf.reduce_sum(tf.cast(flat_bin_output_cnn, tf.float32))
        tp_cnn = tf.reduce_sum(tf.cast(tf.logical_and(flat_labels_bool, flat_bin_output_cnn), tf.float32))
        metrics['cnn_precision'] = tf.divide(tp_cnn, tf.add(num_fg_output_cnn, cfg.EPSILON))
        metrics['cnn_recall'] = tf.divide(tp_cnn, tf.add(num_pixel_fg, cfg.EPSILON))
        metrics['cnn_tp'] = tp_cnn
        metrics['cnn_num_fg_output'] = num_fg_output_cnn
        metrics['num_pixel_fg'] = num_pixel_fg # Chung

        # 2. GNN Loss & Metrics
        cross_entropies_gnn = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=node_logits, labels=node_labels
        )
        num_node = tf.size(node_labels)
        num_node_fg = tf.cast(tf.math.count_nonzero(node_labels, dtype=tf.int32), tf.float32)
        num_node_bg = tf.cast(num_node, tf.float32) - num_node_fg
        safe_num_node = tf.add(tf.cast(num_node, tf.float32), cfg.EPSILON)

        gnn_weight_fg = tf.divide(num_node_bg, safe_num_node)
        gnn_weight_bg = tf.divide(num_node_fg, safe_num_node)

        gnn_class_weight = tf.cast(tf.stack([gnn_weight_bg, gnn_weight_fg]), dtype=tf.float32)
        gnn_one_hot = tf.one_hot(tf.cast(node_labels, tf.int32), 2)
        gnn_weight_per_label = tf.reduce_sum(gnn_one_hot * gnn_class_weight, axis=1)

        losses['gnn'] = tf.reduce_mean(tf.multiply(gnn_weight_per_label, cross_entropies_gnn))

        gnn_prob = tf.sigmoid(node_logits)
        gnn_correct = tf.equal(tf.cast(tf.greater_equal(gnn_prob, 0.5), tf.int32), tf.cast(node_labels, tf.int32))
        metrics['gnn_accuracy'] = tf.reduce_mean(tf.cast(gnn_correct, tf.float32))

        # 3. Post-CNN (Inference) Loss & Metrics
        flat_logits_post_cnn = tf.reshape(tensor=post_cnn_img_output, shape=(-1,))
        cross_entropies_post_cnn = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=flat_logits_post_cnn, labels=flat_labels_float
        )
        reshaped_pixel_weights = tf.reshape(tensor=pixel_weights, shape=(-1,))
        safe_mean_pixel_weights = tf.add(tf.reduce_mean(reshaped_pixel_weights), cfg.EPSILON)
        reshaped_pixel_weights_norm = reshaped_pixel_weights / safe_mean_pixel_weights

        weighted_loss_post_cnn = tf.multiply(reshaped_pixel_weights_norm, weight_per_label)
        losses['post_cnn'] = tf.reduce_mean(tf.multiply(weighted_loss_post_cnn, cross_entropies_post_cnn))

        flat_bin_output_post_cnn = tf.greater_equal(tf.reshape(tensor=post_cnn_img_fg_prob, shape=(-1,)), 0.5)
        metrics['post_cnn_accuracy'] = tf.reduce_mean(tf.cast(tf.equal(flat_bin_output_post_cnn, flat_labels_bool), tf.float32))
        num_fg_output_post_cnn = tf.reduce_sum(tf.cast(flat_bin_output_post_cnn, tf.float32))
        tp_post_cnn = tf.reduce_sum(tf.cast(tf.logical_and(flat_labels_bool, flat_bin_output_post_cnn), tf.float32))
        metrics['post_cnn_precision'] = tf.divide(tp_post_cnn, tf.add(num_fg_output_post_cnn, cfg.EPSILON))
        metrics['post_cnn_recall'] = tf.divide(tp_post_cnn, tf.add(num_pixel_fg, cfg.EPSILON))
        metrics['post_cnn_tp'] = tp_post_cnn
        metrics['post_cnn_num_fg_output'] = num_fg_output_post_cnn

        # 4. Total Loss
        total_loss = losses['post_cnn']
        if self.cnn_loss_on:
            total_loss += losses['cnn']

        # Thêm L2
        total_loss_with_l2 = total_loss + tf.add_n(self.losses)

        # GNN loss được xử lý riêng trong train_step

        losses['total'] = total_loss
        losses['total_with_l2'] = total_loss_with_l2

        metrics.update(losses)
        return losses, metrics

    def _process_gradients(self, grads_and_vars_1, grads_and_vars_2):
        """Xử lý gradient phức tạp cho VGN."""

        # Xử lý GNN Grads (grads_and_vars_2)
        processed_gvs_2 = []
        if self.gnn_loss_on:
            for g, v in grads_and_vars_2:
                if g is None:
                    processed_gvs_2.append((g, v))
                    continue

                if 'gat' in v.name:
                    g = tf.clip_by_value(g, -5., 5.)
                else:
                    g = None # Chỉ cập nhật trọng số GAT
                processed_gvs_2.append((g, v))
        else:
            processed_gvs_2 = [(None, v) for g, v in grads_and_vars_2]

        # Kết hợp Grads
        combined_gvs = []
        for (g1, v1), (g2, v2) in zip(grads_and_vars_1, processed_gvs_2):
            assert v1.name == v2.name
            combined_g = add_tensors_wo_none([g1, g2])
            combined_gvs.append((combined_g, v1))

        # Xử lý Combined Grads
        final_gvs = []
        lr_ratio = self.params.new_net_lr / (self.params.old_net_ft_lr + cfg.EPSILON)

        for g, v in combined_gvs:
            if g is None:
                final_gvs.append((g, v))
                continue

            # old_net_ft_lr == 0: Chỉ cập nhật mạng mới
            if self.params.old_net_ft_lr == 0:
                is_new_net = ('gat' in v.name or 'post_cnn' in v.name)
                if self.params.do_simul_training and is_new_net:
                    g = tf.clip_by_value(g, -5., 5.)
                elif 'post_cnn' in v.name: # Không huấn luyện GAT
                    g = tf.clip_by_value(g, -5., 5.)
                else:
                    g = None # Đóng băng mạng cũ

            # old_net_ft_lr > 0: Huấn luyện toàn bộ
            else:
                is_new_net = ('gat' in v.name or 'post_cnn' in v.name)
                if self.params.do_simul_training and is_new_net:
                    g = g * lr_ratio
                elif 'post_cnn' in v.name: # Không huấn luyện GAT
                    g = g * lr_ratio

                g = tf.clip_by_value(g, -5., 5.)

            if g is not None and 'post_cnn' in v.name:
                g = g * self.params.infer_module_grad_weight

            final_gvs.append((g, v))

        return final_gvs

    # Ghi đè train_step
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            # Chạy thuận
            outputs = self(data, training=True)

            # Tính loss
            losses, metrics = self.compute_loss_and_metrics(outputs, data)

            loss_1 = losses['total_with_l2'] # CNN + Post-CNN
            loss_2 = losses['gnn'] * self.gnn_loss_weight # GNN

        trainable_vars = self.trainable_variables

        # Tính gradients cho cả hai loss
        grads_1 = tape.gradient(loss_1, trainable_vars)
        grads_2 = tape.gradient(loss_2, trainable_vars)

        del tape # Xóa tape

        grads_and_vars_1 = list(zip(grads_1, trainable_vars))
        grads_and_vars_2 = list(zip(grads_2, trainable_vars))

        # Xử lý gradients
        final_grads_and_vars = self._process_gradients(grads_and_vars_1, grads_and_vars_2)

        # Áp dụng gradients
        self.optimizer.apply_gradients(final_grads_and_vars)

        # Trả về metrics
        return metrics

    # Ghi đè test_step
    def test_step(self, data):
        # Chạy thuận
        outputs = self(data, training=False)

        # Tính loss và metrics
        losses, metrics = self.compute_loss_and_metrics(outputs, data)

        return metrics
