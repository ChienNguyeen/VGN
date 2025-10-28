# updated by syshin (180825)
# Migrated to Python 3 / TensorFlow 2 by Gemini
# do the following steps before running this script
# (1) run a script to generate training/test graphs (e.g., 'GenGraph/make_graph_db.py')
# (2) place the generated graphs ('.graph_res')
# and cnn results ('_prob.png') in
# a new directory 'args.save_root/graph'

import numpy as np
import os
import argparse
import skimage.io
import networkx as nx
import pickle as pkl
import multiprocessing
import sys
import tensorflow as tf
from tqdm import tqdm

# TF2/Py3 Change: Thêm các thư viện cần thiết
import _init_paths
from config import cfg
from model import VesselSegmVGN # TF2/Py3 Change: Sửa tên model từ file model.py
import util

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a VesselSegmVGN network (TF2/Py3 version)')
    # Các tham số đã được điều chỉnh để khớp với model.py
    parser.add_argument('--dataset', default='STARE', help='Dataset to use: Can be DRIVE or STARE', type=str)
    parser.add_argument('--save_root', default='VGN_STARE', help='Root path to save models and results', type=str)
    
    # --- Đường dẫn tới Pretrained CNN Model ---
    # Cần một checkpoint của mô hình CNN đã huấn luyện (từ train_CNN.py)
    parser.add_argument('--pretrained_cnn_ckpt', default='DRIU_DRIVE/train/ckpt-50000', help='Path for a pretrained CNN checkpoint', type=str)

    # --- Tham số cho việc tạo đồ thị (nếu cần cập nhật) ---
    parser.add_argument('--win_size', default=8, help='Window size for graph nodes', type=int)
    parser.add_argument('--edge_geo_dist_thresh', default=20, help='Threshold for geodesic distance for graph edges', type=float)

    # --- Tham số mô hình VGN ---
    parser.add_argument('--cnn_model', default='driu', help='CNN backbone', type=str)
    parser.add_argument('--cnn_loss_on', default=True, help='Whether to include CNN loss', type=bool)
    parser.add_argument('--gnn_loss_on', default=True, help='Whether to include GNN loss', type=bool)
    parser.add_argument('--gnn_loss_weight', default=1.0, help='Weight for GNN loss', type=float)
    parser.add_argument('--gat_n_heads', default=[4, 4], help='Numbers of heads in each GAT layer', type=list)
    parser.add_argument('--gat_hid_units', default=[16], help='Numbers of hidden units per head', type=list)
    parser.add_argument('--infer_module_kernel_size', default=3, help='Kernel size for inference module', type=int)
    parser.add_argument('--norm_type', default='BN', help='Normalization type (BN or GN)', type=str)
    parser.add_argument('--use_enc_layer', default=True, help='Use encoder layer in inference module', type=bool)
    parser.add_argument('--do_simul_training', default=True, help='Simultaneous training of GNN and Inference Module', type=bool)
    parser.add_argument('--infer_module_grad_weight', default=1.0, help='Gradient weight for inference module', type=float)

    # --- Tham số Optimizer ---
    parser.add_argument('--max_iters', default=50000, help='Maximum number of iterations', type=int)
    parser.add_argument('--opt', default='adam', help='Optimizer', type=str)
    parser.add_argument('--lr_scheduling', default='pc', help='Learning rate scheduling', type=str)
    parser.add_argument('--new_net_lr', default=1e-4, help='Learning rate for new (GNN/Infer) layers', type=float)
    parser.add_argument('--old_net_ft_lr', default=1e-5, help='Learning rate for fine-tuning old (CNN) layers', type=float)
    parser.add_argument('--lr_decay_tp', default=0.5, help='Learning rate decay time point', type=float)
    
    # --- Tham số Graph Update ---
    # TF2/Py3 Change: Tạm thời vô hiệu hóa graph update vì nó rất phức tạp để di chuyển
    # Bạn có thể bật lại nếu đã có script create_graphs.py tương thích TF2
    parser.add_argument('--use_graph_update', default=False, help='Whether to update graphs during training (currently disabled)', type=bool)
    parser.add_argument('--graph_update_period', default=10000, help='Graph update period', type=int)

    args = parser.parse_args()
    return args

def setup_gpu_memory_growth():
    """Cấu hình GPU cho TensorFlow 2.x."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def get_pixel_weights(probmap, fov, labels, thresh=0.9):
    """Tạo trọng số pixel cho inference loss, dựa trên kết quả CNN."""
    weights = np.zeros(labels.shape, dtype=np.float32)
    correct_pixels = (probmap >= thresh) == (labels >= 0.5)
    uncertain_pixels = (probmap > (1 - thresh)) & (probmap < thresh)
    weights[~correct_pixels] = 1.0
    weights[uncertain_pixels] = 1.0
    weights = weights * fov
    return weights

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    setup_gpu_memory_growth()

    # --- Chuẩn bị đường dẫn ---
    # Script này sẽ tạo một thư mục con 'graph' bên trong save_root để chứa đồ thị
    temp_graph_save_path = os.path.join(args.save_root, cfg.TRAIN.TEMP_GRAPH_SAVE_PATH)
    model_save_path = os.path.join(args.save_root, cfg.TRAIN.MODEL_SAVE_PATH)
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    if not os.path.exists(temp_graph_save_path):
        os.makedirs(temp_graph_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # --- Chuẩn bị danh sách file ---
    # File config.py đã có sẵn đường dẫn ../STARE/train.txt, phù hợp với yêu cầu của bạn
    if args.dataset == 'STARE':
        train_set_txt_path = cfg.TRAIN.STARE_SET_TXT_PATH
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not configured in this script.")

    with open(train_set_txt_path) as f:
        train_img_names_raw = [x.strip() for x in f.readlines()]

    # Sửa đổi đường dẫn để trỏ đến các file đồ thị trong thư mục tạm thời
    # Ví dụ: data/STARE/images/im0001 -> VGN_STARE/graph/im0001
    train_img_names = []
    for path in train_img_names_raw:
        base_name = os.path.basename(path)
        train_img_names.append(os.path.join(temp_graph_save_path, base_name))
    
    print(f"Expecting to find graph files in: {temp_graph_save_path}")
    print("Please run the graph generation script first!")

    # --- Khởi tạo DataLayer và Model ---
    data_layer_train = util.GraphDataLayer(train_img_names, is_training=True, win_size=args.win_size, edge_geo_dist_thresh=args.edge_geo_dist_thresh)
    network = VesselSegmVGN(args, weight_file_path=None)

    # --- Build mô hình và tải trọng số CNN đã huấn luyện ---
    # Lấy một batch giả để xác định shape và build model
    _, blobs_dummy = data_layer_train.forward()
    dummy_img = tf.convert_to_tensor(blobs_dummy['img'], dtype=tf.float32)
    
    # Tạo đồ thị giả
    dummy_graph = blobs_dummy['graph']
    dummy_adj = nx.adjacency_matrix(dummy_graph)
    dummy_adj_tuple = util.sparse_to_tuple(dummy_adj)
    dummy_adj_sparse = tf.SparseTensor(*dummy_adj_tuple)
    dummy_node_byxs = util.get_node_byx_from_graph(dummy_graph, blobs_dummy['num_of_nodes_list'])
    
    dummy_inputs = {
        'imgs': dummy_img, 'node_byxs': dummy_node_byxs, 'adj': dummy_adj_sparse,
    }
    _ = network(dummy_inputs, training=False)
    print("Model built successfully.")

    if args.pretrained_cnn_ckpt:
        print(f"Loading pretrained CNN weights from: {args.pretrained_cnn_ckpt}")
        # Tạo một checkpoint chỉ cho các biến của CNN
        cnn_vars = [v for v in network.trainable_variables if 'gat' not in v.name and 'post_cnn' not in v.name]
        cnn_ckpt = tf.train.Checkpoint(**{v.name.split(':')[0]: v for v in cnn_vars})
        try:
            cnn_ckpt.restore(args.pretrained_cnn_ckpt).expect_partial()
            print("CNN weights restored successfully.")
        except Exception as e:
            print(f"Error restoring CNN weights: {e}")

    # --- Checkpoint và Summary (TF2 style) ---
    ckpt = tf.train.Checkpoint(model=network, optimizer=network.optimizer)
    manager = tf.train.CheckpointManager(ckpt, model_save_path, max_to_keep=10)
    summary_writer = tf.summary.create_file_writer(model_save_path)

    # --- Vòng lặp Huấn luyện ---
    print("Starting VGN model training...")
    for iter_num in tqdm(range(args.max_iters)):
        # Lấy một batch dữ liệu
        _, blobs_train = data_layer_train.forward()

        # Chuyển đổi dữ liệu sang Tensors
        imgs = tf.convert_to_tensor(blobs_train['img'], dtype=tf.float32)
        labels = tf.convert_to_tensor(blobs_train['label'], dtype=tf.float32)
        fov_masks = tf.convert_to_tensor(blobs_train['fov'], dtype=tf.float32)
        probmap = tf.convert_to_tensor(blobs_train['probmap'], dtype=tf.float32)

        # Xử lý đồ thị
        graph = blobs_train['graph']
        adj = nx.adjacency_matrix(graph, weight=None).astype(float)
        adj_tuple = util.sparse_to_tuple(sp.coo_matrix(adj))
        adj_sparse = tf.SparseTensor(*adj_tuple)
        
        node_labels = np.array([graph.nodes[i]['label'] for i in range(graph.number_of_nodes())], dtype=np.float32)
        node_byxs = util.get_node_byx_from_graph(graph, blobs_train['num_of_nodes_list'])

        pixel_weights = get_pixel_weights(blobs_train['probmap']/255.0, blobs_train['fov'], blobs_train['label'])

        # Đóng gói tất cả inputs
        train_data = {
            'imgs': imgs,
            'node_byxs': tf.convert_to_tensor(node_byxs, dtype=tf.int32),
            'adj': adj_sparse,
            'labels': labels,
            'fov_masks': fov_masks,
            'node_labels': tf.convert_to_tensor(node_labels),
            'pixel_weights': tf.convert_to_tensor(pixel_weights),
            'is_lr_flipped': blobs_train['vec_aug_on'][0],
            'is_ud_flipped': blobs_train['vec_aug_on'][1],
            'rot90_num': blobs_train['rot_angle'] / 90.0 if blobs_train['vec_aug_on'][2] else 0,
            'gnn_feat_dropout': 0.5, # Giá trị mặc định từ code gốc
            'gnn_att_dropout': 0.5,  # Giá trị mặc định từ code gốc
            'post_cnn_dropout': 0.1, # Giá trị mặc định từ code gốc
        }
        
        # Chạy một bước huấn luyện
        metrics = network.train_step(train_data)

        if (iter_num + 1) % 100 == 0:
            with summary_writer.as_default(step=iter_num + 1):
                for name, value in metrics.items():
                    tf.summary.scalar(name, value)
            
            tqdm.write(f"Iter {iter_num+1}, Total Loss: {metrics['total_with_l2']:.4f}, "
                       f"GNN Acc: {metrics['gnn_accuracy']:.4f}, Post-CNN Acc: {metrics['post_cnn_accuracy']:.4f}")

        if (iter_num + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
            manager.save(checkpoint_number=iter_num + 1)

    manager.save(checkpoint_number=args.max_iters)
    print("Training complete.")