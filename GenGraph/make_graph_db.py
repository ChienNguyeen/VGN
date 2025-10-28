# -*- coding: utf-8 -*-
"""
Script to create a graph database from CNN probability maps.
Updated for Python 3 and dynamic paths.
"""
from __future__ import print_function # For compatibility
import numpy as np
import skimage.io
import os
import networkx as nx
import pickle as pkl
import multiprocessing
import argparse
import skfmm # Thư viện cho Fast Marching Method

# Bạn có thể cần cài đặt thư viện bwmorph
# pip install bwmorph
from bwmorph import bwmorph
from config import cfg
import util
from tqdm import tqdm

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Make a graph db')
    parser.add_argument('--dataset', default='STARE', help='Dataset to use: Can be DRIVE, STARE, etc.', type=str)
    
    # --- THAM SỐ ĐƯỜNG DẪN MỚI ---
    parser.add_argument('--data_root', default='data', help='Root directory of the dataset', type=str)
    parser.add_argument('--cnn_result_path', required=True, help='Path to the directory containing _prob.png files from CNN', type=str)
    parser.add_argument('--graph_save_path', required=True, help='Path to the directory to save the output .graph_res files', type=str)
    
    # --- Các tham số gốc ---
    parser.add_argument('--use_multiprocessing', default=True, help='Whether to use python multiprocessing', type=bool)
    parser.add_argument('--win_size', default=8, help='Window size for graph node sampling', type=int)
    parser.add_argument('--edge_method', default='geo_dist', help='Edge construction method: geo_dist or eu_dist', type=str)
    parser.add_argument('--edge_dist_thresh', default=20, help='Distance threshold for edge construction', type=float)

    args = parser.parse_args()
    return args

# SỬA PYTHON 3: Thay đổi chữ ký hàm để nhận một tuple
def generate_graph_using_srns(packed_args):
    """
    Hàm chính để tạo đồ thị cho một ảnh.
    """
    # SỬA PYTHON 3: Giải nén tuple
    img_name, im_root_path, cnn_result_root_path, graph_save_path, params = packed_args

    win_size_str = '%.2d_%.2d' % (params.win_size, params.edge_dist_thresh)

    # Xác định phần mở rộng file và kích thước ảnh dựa trên dataset
    if 'DRIVE' in img_name:
        im_ext, label_ext, len_y, len_x = '_image.tif', '_label.gif', 592, 592
    elif 'STARE' in img_name:
        im_ext, label_ext, len_y, len_x = '.ppm', '.ah.ppm', 704, 704
    elif 'CHASE_DB1' in img_name:
        im_ext, label_ext, len_y, len_x = '.jpg', '_1stHO.png', 1024, 1024
    elif 'HRF' in img_name:
        im_ext, label_ext, len_y, len_x = '.bmp', '.tif', 768, 768
    else:
        raise ValueError("Unknown dataset in image name: " + img_name)

    # Xây dựng đường dẫn
    cur_filename = os.path.basename(img_name)
    print('Processing ' + cur_filename)

    cur_im_path = os.path.join(im_root_path, cur_filename + im_ext)
    cur_res_prob_path = os.path.join(cnn_result_root_path, cur_filename + '_prob.png')
    cur_res_graph_savepath = os.path.join(graph_save_path, cur_filename + '_' + win_size_str + '.graph_res')

    # Đọc ảnh xác suất
    vesselness = skimage.io.imread(cur_res_prob_path)
    vesselness = vesselness.astype(float) / 255.0

    # Đảm bảo ảnh có kích thước cố định bằng padding
    temp = np.copy(vesselness)
    vesselness = np.zeros((len_y, len_x), dtype=temp.dtype)
    vesselness[:temp.shape[0], :temp.shape[1]] = temp

    # 1. TÌM CÁC NÚT (NODE SAMPLING)
    # Tìm các điểm cực đại cục bộ trong mỗi cửa sổ (window)
    im_y, im_x = vesselness.shape
    y_quan = list(range(0, im_y, params.win_size))
    x_quan = list(range(0, im_x, params.win_size))
    
    max_pos = []
    for y_start in y_quan:
        for x_start in x_quan:
            y_end = min(y_start + params.win_size, im_y)
            x_end = min(x_start + params.win_size, im_x)
            cur_patch = vesselness[y_start:y_end, x_start:x_end]
            
            if np.sum(cur_patch) == 0:
                # Nếu vùng trống, lấy điểm trung tâm
                max_pos.append((y_start + cur_patch.shape[0] // 2, x_start + cur_patch.shape[1] // 2))
            else:
                # Tìm vị trí có xác suất cao nhất
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_start + temp[0], x_start + temp[1]))

    graph = nx.Graph()

    # Thêm các nút đã tìm thấy vào đồ thị
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        # Lấy nhãn (label) từ ground truth để lưu trữ
        # Lưu ý: Cần có file ground truth để gán nhãn cho nút
        # Tạm thời bỏ qua nếu không có, hoặc mặc định là 0
        node_label = 0 # Mặc định
        graph.add_node(node_idx, y=node_y, x=node_x, label=node_label)

    # 2. TẠO CÁC CẠNH (EDGE CONSTRUCTION)
    speed = vesselness
    edge_dist_thresh_sq = params.edge_dist_thresh ** 2
    node_list = list(graph.nodes)

    for i, n in enumerate(tqdm(node_list, desc=f"Building edges for {cur_filename}", leave=False)):
        
        node_data = graph.nodes[n]
        if speed[node_data['y'], node_data['x']] < 0.1: # Bỏ qua các nút ở vùng xác suất quá thấp
            continue

        if params.edge_method == 'geo_dist':
            # Tính khoảng cách trắc địa (geodesic distance)
            phi = np.ones_like(speed)
            phi[node_data['y'], node_data['x']] = -1
            
            # skfmm.travel_time tính thời gian "di chuyển" từ nút n đến các điểm khác
            # Tốc độ di chuyển chính là giá trị xác suất (vesselness)
            tt = skfmm.travel_time(phi, speed, narrow=params.edge_dist_thresh)

            for n_comp in node_list[i + 1:]:
                comp_node_data = graph.nodes[n_comp]
                geo_dist = tt[comp_node_data['y'], comp_node_data['x']]
                if geo_dist < params.edge_dist_thresh:
                    graph.add_edge(n, n_comp, weight=1.0) # Có thể dùng trọng số phức tạp hơn

        elif params.edge_method == 'eu_dist':
            # Tính khoảng cách Euclidean
            for n_comp in node_list[i + 1:]:
                comp_node_data = graph.nodes[n_comp]
                eu_dist_sq = (comp_node_data['y'] - node_data['y']) ** 2 + (comp_node_data['x'] - node_data['x']) ** 2
                if eu_dist_sq < edge_dist_thresh_sq:
                    graph.add_edge(n, n_comp, weight=1.0)
        else:
            raise NotImplementedError

    # 3. LƯU ĐỒ THỊ
    nx.write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)
    print(f"Saved graph for {cur_filename} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    # --- SỬA PYTHON 3: Xây dựng đường dẫn động ---
    im_root_path = os.path.join(args.data_root, args.dataset, 'images')
    train_set_txt_path = os.path.join(args.data_root, args.dataset, 'train.txt')
    test_set_txt_path = os.path.join(args.data_root, args.dataset, 'test.txt')

    # Đảm bảo thư mục lưu đồ thị tồn tại
    if not os.path.exists(args.graph_save_path):
        os.makedirs(args.graph_save_path)

    with open(train_set_txt_path) as f:
        train_img_names = [os.path.basename(x.strip()) for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_img_names = [os.path.basename(x.strip()) for x in f.readlines()]

    all_img_names = train_img_names + test_img_names
    
    # SỬA PYTHON 3: Chuẩn bị đối số cho hàm map
    func_args = []
    for img_name in all_img_names:
        # Đóng gói các đối số vào một tuple
        packed_arg = (img_name, im_root_path, args.cnn_result_path, args.graph_save_path, args)
        func_args.append(packed_arg)

    # Chạy xử lý
    if args.use_multiprocessing:
        pool = multiprocessing.Pool(processes=min(os.cpu_count(), 16))
        pool.map(generate_graph_using_srns, func_args)
        pool.close()
        pool.join()
    else:
        for arg in func_args:
            generate_graph_using_srns(arg)
            
    print("\nGraph generation complete!")