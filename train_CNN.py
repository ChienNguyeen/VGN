"""
Script huấn luyện cho vessel_segm_cnn, đã được nâng cấp lên TensorFlow 2.x
"""
import numpy as np
import os
import pdb
import skimage.io
import argparse
import tensorflow as tf
from tqdm import tqdm # Thêm tqdm để theo dõi tiến độ

from config import cfg
# Sử dụng mô hình Keras đã được nâng cấp từ common_model_tf2.py
from model import vessel_segm_cnn
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a vessel_segm_cnn network (TF2 version)')
    parser.add_argument('--dataset', default='DRIVE', help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1 or HRF', type=str)
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)
    parser.add_argument('--use_fov_mask', default=False, help='Whether to use fov masks', type=bool)
    parser.add_argument('--opt', default='adam', help='Optimizer to use: Can be sgd or adam', type=str)
    parser.add_argument('--lr', default=1e-02, help='Learning rate to use: Can be any floating point number', type=float)
    parser.add_argument('--lr_decay', default='pc', help='Learning rate decay to use: Can be const or pc or exp', type=str)
    parser.add_argument('--max_iters', default=50000, help='Maximum number of iterations', type=int)
    parser.add_argument('--pretrained_model', default='../pretrained_model/VGG_imagenet.npy', help='path for a pretrained model(.npy)', type=str)
    parser.add_argument('--save_root', default='DRIU_DRIVE', help='root path to save trained models and test results', type=str)

    args = parser.parse_args()
    return args

def setup_gpu_memory_growth():
    """Cấu hình GPU cho TensorFlow 2.x để bật 'memory growth'."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Yêu cầu TensorFlow chỉ cấp phát bộ nhớ GPU khi cần thiết
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Lỗi xảy ra nếu memory growth đã được thiết lập trước đó
            print(e)

def load_pretrained_weights(model, data_path, ignore_missing=False):
    """
    Tải trọng số đã huấn luyện trước từ tệp .npy (VGG) vào mô hình Keras.
    """
    print(f"Loading pretrained weights from {data_path}...")
    # allow_pickle for Python3 compatibility when loading .npy dicts
    data_dict = np.load(data_path, allow_pickle=True, encoding='latin1').item()

    for key in data_dict: # key là tên lớp, ví dụ: 'conv1_1'
        try:
            # Lấy lớp Keras từ mô hình bằng tên (ví dụ: model.conv1_1)
            layer = getattr(model, key)
            sub_dict = data_dict[key] # Đây là dict {'weights': ..., 'biases': ...}

            # Tạo danh sách trọng số để set
            weights_to_set = []
            if 'weights' in sub_dict:
                weights_to_set.append(sub_dict['weights'])
            if 'biases' in sub_dict:
                weights_to_set.append(sub_dict['biases'])

            if weights_to_set:
                layer.set_weights(weights_to_set)
                # print(f"Assigned pretrained weights to {key}") # Bỏ comment nếu muốn xem chi tiết

        except (AttributeError, ValueError, KeyError) as e:
            # AttributeError: nếu mô hình không có lớp tên `key`
            # ValueError: nếu hình dạng trọng số không khớp
            # KeyError: nếu 'weights'/'biases' không có trong sub_dict
            print(f"Warning: Could not load weights for {key}. Error: {e}")
            if not ignore_missing:
                raise

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    # --- Cấu hình GPU (thay thế cho tf.ConfigProto) ---
    setup_gpu_memory_growth()

    # --- Chuẩn bị dữ liệu (giữ nguyên) ---
    if args.dataset == 'DRIVE':
        train_set_txt_path = cfg.TRAIN.DRIVE_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.DRIVE_SET_TXT_PATH
    elif args.dataset == 'STARE':
        train_set_txt_path = "data/STARE/list/train.txt"
        test_set_txt_path = "data/STARE/list/test.txt"
    elif args.dataset == 'CHASE_DB1':
        train_set_txt_path = cfg.TRAIN.CHASE_DB1_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.CHASE_DB1_SET_TXT_PATH
    elif args.dataset == 'HRF':
        train_set_txt_path = cfg.TRAIN.HRF_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.HRF_SET_TXT_PATH
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    with open(train_set_txt_path) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]

    if args.dataset == 'HRF':
        test_img_names = [test_img_names[i] for i in range(7, len(test_img_names), 20)]

    len_train = len(train_img_names)
    len_test = len(test_img_names)

    data_layer_train = util.DataLayer(train_img_names, is_training=True)
    data_layer_test = util.DataLayer(test_img_names, is_training=False)

    # --- Chuẩn bị đường dẫn (giữ nguyên) ---
    model_save_path = args.save_root + '/' + cfg.TRAIN.MODEL_SAVE_PATH if len(args.save_root) > 0 else cfg.TRAIN.MODEL_SAVE_PATH
    res_save_path = args.save_root + '/' + cfg.TEST.RES_SAVE_PATH if len(args.save_root) > 0 else cfg.TEST.RES_SAVE_PATH
    if len(args.save_root) > 0 and not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.isdir(res_save_path):
        os.mkdir(res_save_path)

    # --- Khởi tạo Mô hình và Trình tối ưu hóa (TF2) ---
    network = vessel_segm_cnn(args, None)

    # --- Checkpoint (thay thế tf.train.Saver) ---
    ckpt = tf.train.Checkpoint(model=network, optimizer=network.optimizer, step=network.optimizer.iterations)
    manager = tf.train.CheckpointManager(ckpt, model_save_path, max_to_keep=100)

    # --- Summary Writer (thay thế tf.summary.FileWriter) ---
    summary_writer = tf.summary.create_file_writer(model_save_path)

    # --- KHU VỰC SỬA 1: KÍCH THƯỚC MỤC TIÊU ---
    # Kích thước mục tiêu mới (chia hết cho 32)
    TARGET_H = 608
    TARGET_W = 704

    # --- Khởi tạo/Tải trọng số (thay thế sess.run và load) ---
    print("Building model by running a dummy forward pass...")
    _, blobs_dummy = data_layer_train.forward()
    dummy_img_tensor_raw = tf.convert_to_tensor(blobs_dummy['img'], dtype=tf.float32)

    # Resize/pad dummy tensor về kích thước cố định, dùng 'bilinear'
    dummy_img_tensor = tf.image.resize_with_pad(
        dummy_img_tensor_raw, TARGET_H, TARGET_W, method='bilinear'
    )
    _ = network(dummy_img_tensor, training=False)
    print("Model built.")

    if args.pretrained_model is not None:
        load_pretrained_weights(network, args.pretrained_model, ignore_missing=True)

    f_log = open(os.path.join(model_save_path, 'log.txt'), 'w')
    last_snapshot_iter = -1
    timer = util.Timer()

    train_loss_list = []
    test_loss_list = []
    print("Training the model...")
    for iter in range(args.max_iters):

        timer.tic()

        # get one batch
        _, blobs_train = data_layer_train.forward()

        if args.use_fov_mask:
            fov_masks = blobs_train['fov']
        else:
            fov_masks = np.ones(blobs_train['label'].shape, dtype=blobs_train['label'].dtype)

        # --- KHU VỰC SỬA 2: TRAINING LOOP ---
        # Chuyển đổi numpy arrays sang Tensors
        img_tensor_raw = tf.convert_to_tensor(blobs_train['img'], dtype=tf.float32)
        label_tensor_raw = tf.convert_to_tensor(blobs_train['label'], dtype=tf.int64)
        fov_tensor_raw = tf.convert_to_tensor(fov_masks, dtype=tf.int64)

        # 1. Resize/pad ảnh (dùng 'bilinear')
        img_tensor = tf.image.resize_with_pad(
            img_tensor_raw, TARGET_H, TARGET_W, method='bilinear'
        )

        # 2. Resize/pad label và fov (dùng 'nearest')
        # Giả định label_tensor_raw và fov_tensor_raw đã là 4D: [B, H, W, 1]
        label_tensor_padded = tf.image.resize_with_pad(
            label_tensor_raw, TARGET_H, TARGET_W, method='nearest'
        )
        fov_tensor_padded = tf.image.resize_with_pad(
            fov_tensor_raw, TARGET_H, TARGET_W, method='nearest'
        )

        # Gọi hàm train_step tùy chỉnh
        # Truyền trực tiếp các tensor 4D đã pad
        train_data = (img_tensor, label_tensor_padded, fov_tensor_padded)
        metrics = network.train_step(train_data)
        # --- KẾT THÚC SỬA 2 ---

        # Lấy kết quả (là EagerTensors, chuyển sang numpy)
        loss_val = metrics['loss'].numpy()
        accuracy_val = metrics['accuracy'].numpy()
        pre_val = metrics['precision'].numpy()
        rec_val = metrics['recall'].numpy()

        timer.toc()
        train_loss_list.append(loss_val)

        if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
            print(f'iter: {iter + 1} / {args.max_iters}, loss: {loss_val:.4f}, '
                  f'accuracy: {accuracy_val:.4f}, precision: {pre_val:.4f}, recall: {rec_val:.4f}')
            print('speed: {:.3f}s / iter'.format(timer.average_time))

        # --- Lưu Checkpoint (thay thế saver.save) ---
        if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
            last_snapshot_iter = iter
            save_path = manager.save(checkpoint_number=iter + 1)
            print('Wrote snapshot to: {:s}'.format(save_path))

        # --- Vòng lặp Đánh giá (thay thế sess.run) ---
        if (iter + 1) % cfg.TRAIN.TEST_ITERS == 0:

            all_labels = np.zeros((0,))
            all_preds = np.zeros((0,))

            print("Running testing...")
            num_test_batches = int(np.ceil(float(len_test) / cfg.TRAIN.BATCH_SIZE))

            for _ in tqdm(range(num_test_batches), total=num_test_batches):

                # get one batch
                img_list, blobs_test = data_layer_test.forward()

                imgs_np = blobs_test['img']
                labels_np = blobs_test['label'] # Đây là numpy gốc (B, H, W, 1)
                if args.use_fov_mask:
                    fov_masks_np = blobs_test['fov'] # Đây là numpy gốc (B, H, W, 1)
                else:
                    fov_masks_np = np.ones(labels_np.shape, dtype=labels_np.dtype)

                # --- KHU VỰC SỬA 3: TESTING LOOP (FORWARD PASS) ---
                # Chuyển sang Tensors
                imgs_raw = tf.convert_to_tensor(imgs_np, dtype=tf.float32)
                labels_raw = tf.convert_to_tensor(labels_np, dtype=tf.int64)
                fov_masks_raw = tf.convert_to_tensor(fov_masks_np, dtype=tf.int64)

                # 1. Resize/pad ảnh (dùng 'bilinear')
                imgs = tf.image.resize_with_pad(
                    imgs_raw, TARGET_H, TARGET_W, method='bilinear'
                )

                # 2. Resize/pad label và fov (dùng 'nearest')
                labels_padded = tf.image.resize_with_pad(
                    labels_raw, TARGET_H, TARGET_W, method='nearest'
                )
                fov_masks_padded = tf.image.resize_with_pad(
                    fov_masks_raw, TARGET_H, TARGET_W, method='nearest'
                )

                # Chạy forward pass (với ảnh đã pad)
                output, fg_prob_map_tensor = network(imgs, training=False)

                # Tính loss (với output, label, fov đã pad)
                loss_val_tensor = network.compute_loss_fn(output, labels_padded, fov_masks_padded)
                # --- KẾT THÚC SỬA 3 ---

                # Chuyển kết quả về numpy để xử lý
                loss_val = loss_val_tensor.numpy()

                # --- KHU VỰC SỬA 4: CROP OUTPUT VỀ KÍCH THƯỚC GỐC ---
                # Lấy kích thước gốc từ numpy array ban đầu
                original_h = labels_np.shape[1]
                original_w = labels_np.shape[2]

                # Cắt (crop) phần padding khỏi output
                fg_prob_map_tensor_cropped = tf.image.crop_to_bounding_box(
                    fg_prob_map_tensor,
                    offset_height=0,
                    offset_width=0,
                    target_height=original_h,
                    target_width=original_w
                )
                # Chuyển tensor ĐÃ CROP về numpy
                fg_prob_map = fg_prob_map_tensor_cropped.numpy()
                # --- KẾT THÚC SỬA 4 ---

                test_loss_list.append(loss_val)

                # So sánh với labels_np (numpy GỐC)
                all_labels = np.concatenate((all_labels, np.reshape(labels_np, (-1))))

                # Nhân fg_prob_map (đã crop) với fov_masks_np (GỐC)
                fg_prob_map = fg_prob_map * fov_masks_np.astype(float)
                all_preds = np.concatenate((all_preds, np.reshape(fg_prob_map, (-1))))

                # save qualitative results (phần này là numpy nên giữ nguyên)
                cur_batch_size = len(img_list)

                # fg_prob_map (đã crop) bây giờ có shape (B, original_h, original_w, 1)
                # Cần reshape về (B, H, W) để lưu ảnh
                reshaped_fg_prob_map = fg_prob_map.reshape((cur_batch_size, original_h, original_w))
                reshaped_output = reshaped_fg_prob_map >= 0.5

                for img_idx in range(cur_batch_size):
                    cur_test_img_path = img_list[img_idx]
                    slash_indices = util.find(cur_test_img_path, '/')
                    if len(slash_indices) > 0:
                        temp_name = cur_test_img_path[slash_indices[-1] + 1:]
                    else:
                        temp_name = cur_test_img_path

                    cur_reshaped_fg_prob_map = (reshaped_fg_prob_map[img_idx, :, :] * 255).astype(np.uint8)
                    cur_reshaped_output = (reshaped_output[img_idx, :, :].astype(np.uint8) * 255)

                    cur_fg_prob_save_path = os.path.join(res_save_path, temp_name + '_prob.png')
                    cur_output_save_path = os.path.join(res_save_path, temp_name + '_output.png')

                    skimage.io.imsave(cur_fg_prob_save_path, cur_reshaped_fg_prob_map)
                    skimage.io.imsave(cur_output_save_path, cur_reshaped_output)

            # Tính toán metrics (giữ nguyên)
            auc_test, ap_test = util.get_auc_ap_score(all_labels, all_preds)
            all_labels_bin = np.copy(all_labels).astype(bool)
            all_preds_bin = all_preds >= 0.5
            all_correct = all_labels_bin == all_preds_bin
            acc_test = np.mean(all_correct.astype(np.float32))

            # --- Ghi Summary (thay thế tf.Summary) ---
            train_loss_mean = float(np.mean(train_loss_list)) if train_loss_list else 0.0
            test_loss_mean = float(np.mean(test_loss_list)) if test_loss_list else 0.0

            with summary_writer.as_default(step=iter + 1):
                tf.summary.scalar("train_loss", train_loss_mean)
                tf.summary.scalar("test_loss", test_loss_mean)
                tf.summary.scalar("test_acc", float(acc_test))
                tf.summary.scalar("test_auc", float(auc_test))
                tf.summary.scalar("test_ap", float(ap_test))
            summary_writer.flush() # Đẩy summary ra file

            print(f'iter: {iter + 1} / {args.max_iters}, train_loss: {train_loss_mean:.4f}')
            print(f'iter: {iter + 1} / {args.max_iters}, test_loss: {test_loss_mean:.4f}, '
                  f'test_acc: {acc_test:.4f}, test_auc: {auc_test:.4f}, test_ap: {ap_test:.4f}')

            f_log.write(f'iter: {iter + 1} / {args.max_iters}\n')
            f_log.write(f'train_loss {train_loss_mean}\n')
            f_log.write(f'iter: {iter + 1} / {args.max_iters}\n')
            f_log.write(f'test_loss {test_loss_mean}\n')
            f_log.write(f'test_acc {acc_test}\n')
            f_log.write(f'test_auc {auc_test}\n')
            f_log.write(f'test_ap {ap_test}\n')
            f_log.flush()

            train_loss_list = []
            test_loss_list = []

    if last_snapshot_iter != iter:
        save_path = manager.save(checkpoint_number=iter + 1)
        print('Wrote snapshot to: {:s}'.format(save_path))

    f_log.close()
    # (Không cần sess.close())
    print("Training complete.")
