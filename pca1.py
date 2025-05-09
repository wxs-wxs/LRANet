import os
import json
import numpy as np
from scipy.interpolate import splprep, splev
import torch
from scipy.sparse import csc_matrix
from scipy.special import comb as n_over_k
from numpy.linalg import norm
from sklearn.decomposition import IncrementalPCA
import cv2
from skimage import measure
import math
from tqdm import tqdm

def resample_line2(line, n):
    """Resample n points on a line.

    Args:
        line (ndarray): The points composing a line.
        n (int): The resampled points number.

    Returns:
        resampled_line (ndarray): The points composing the resampled line.
    """

    assert line.ndim == 2
    assert line.shape[0] >= 2
    assert line.shape[1] == 2
    assert isinstance(n, int)
    assert n > 0

    length_list = [
        norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
    ]
    total_length = sum(length_list)
    length_cumsum = np.cumsum([0.0] + length_list)
    delta_length = total_length / (float(n) + 1e-8)

    current_edge_ind = 0
    resampled_line = [line[0]]

    for i in range(1, n):
        current_line_len = i * delta_length

        while current_line_len >= length_cumsum[current_edge_ind + 1]:
            current_edge_ind += 1
        current_edge_end_shift = current_line_len - length_cumsum[
            current_edge_ind]
        end_shift_ratio = current_edge_end_shift / length_list[
            current_edge_ind]
        current_point = line[current_edge_ind] + (
            line[current_edge_ind + 1] -
            line[current_edge_ind]) * end_shift_ratio
        resampled_line.append(current_point)

    resampled_line.append(line[-1])
    resampled_line = np.array(resampled_line)

    return resampled_line


def resample_polygon(top_line,bot_line, n=None):

    resample_line = []
    for polygon in [top_line, bot_line]:
        if polygon.shape[0] >= 3:
            x,y = polygon[:,0], polygon[:,1]
            tck, u = splprep([x, y], k=3 if polygon.shape[0] >=5 else 2, s=0)
            u = np.linspace(0, 1, num=n, endpoint=True)
            out = splev(u, tck)
            new_polygon = np.stack(out, axis=1).astype('float32')
        else:
            new_polygon = resample_line2(polygon, n-1)

        resample_line.append(np.array(new_polygon))

    resampled_polygon = np.concatenate([resample_line[0], resample_line[1]]).flatten()
    resampled_polygon = np.expand_dims(resampled_polygon,axis=0)
    return resampled_polygon




def reorder_poly_edge2(points):
    """Get the respective points composing head edge, tail edge, top
    sideline and bottom sideline.

    Args:
        points (ndarray): The points composing a text polygon.

    Returns:
        head_edge (ndarray): The two points composing the head edge of text
            polygon.
        tail_edge (ndarray): The two points composing the tail edge of text
            polygon.
        top_sideline (ndarray): The points composing top curved sideline of
            text polygon.
        bot_sideline (ndarray): The points composing bottom curved sideline
            of text polygon.
    """

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2
    orientation_thr=2.0
    head_inds, tail_inds = find_head_tail(points,
                                                orientation_thr)
    head_edge, tail_edge = points[head_inds], points[tail_inds]

    pad_points = np.vstack([points, points])
    if tail_inds[1] < 1:
        tail_inds[1] = len(points)
    sideline1 = pad_points[head_inds[1]:tail_inds[1]]
    sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
    sideline_mean_shift = np.mean(
        sideline1, axis=0) - np.mean(
            sideline2, axis=0)

    if sideline_mean_shift[1] > 0:
        top_sideline, bot_sideline = sideline2, sideline1
    else:
        top_sideline, bot_sideline = sideline1, sideline2

    return head_edge, tail_edge, top_sideline, bot_sideline
    

def vector_angle(vec1, vec2):
    if vec1.ndim > 1:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
    if vec2.ndim > 1:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
    else:
        unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
    return np.arccos(
        np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

def vector_slope(vec):
    assert len(vec) == 2
    return abs(vec[1] / (vec[0] + 1e-8))

def vector_sin(vec):
    assert len(vec) == 2
    return vec[1] / (norm(vec) + 1e-8)

def vector_cos(vec):
    assert len(vec) == 2
    return vec[0] / (norm(vec) + 1e-8)

def find_head_tail(points, orientation_thr):

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2
    assert isinstance(orientation_thr, float)

    if len(points) > 4:
        pad_points = np.vstack([points, points[0]])
        edge_vec = pad_points[1:] - pad_points[:-1]

        theta_sum = []
        adjacent_vec_theta = []
        for i, edge_vec1 in enumerate(edge_vec):
            adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
            adjacent_edge_vec = edge_vec[adjacent_ind]
            temp_theta_sum = np.sum(
                vector_angle(edge_vec1, adjacent_edge_vec))
            temp_adjacent_theta = vector_angle(
                adjacent_edge_vec[0], adjacent_edge_vec[1])
            theta_sum.append(temp_theta_sum)
            adjacent_vec_theta.append(temp_adjacent_theta)
        theta_sum_score = np.array(theta_sum) / np.pi
        adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
        poly_center = np.mean(points, axis=0)
        edge_dist = np.maximum(
            norm(pad_points[1:] - poly_center, axis=-1),
            norm(pad_points[:-1] - poly_center, axis=-1))
        dist_score = edge_dist / np.max(edge_dist)
        position_score = np.zeros(len(edge_vec))
        score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
        score += 0.35 * dist_score
        if len(points) % 2 == 0:
            position_score[(len(score) // 2 - 1)] += 1
            position_score[-1] += 1
        score += 0.1 * position_score
        pad_score = np.concatenate([score, score])
        score_matrix = np.zeros((len(score), len(score) - 3))
        x = np.arange(len(score) - 3) / float(len(score) - 4)
        gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
            (x - 0.5) / 0.5, 2.) / 2)
        gaussian = gaussian / np.max(gaussian)
        for i in range(len(score)):
            score_matrix[i, :] = score[i] + pad_score[
                (i + 2):(i + len(score) - 1)] * gaussian * 0.3

        head_start, tail_increment = np.unravel_index(
            score_matrix.argmax(), score_matrix.shape)
        tail_start = (head_start + tail_increment + 2) % len(points)
        head_end = (head_start + 1) % len(points)
        tail_end = (tail_start + 1) % len(points)

        if head_end > tail_end:
            head_start, tail_start = tail_start, head_start
            head_end, tail_end = tail_end, head_end
        head_inds = [head_start, head_end]
        tail_inds = [tail_start, tail_end]
    else:
        if vector_slope(points[1] - points[0]) + vector_slope(
                points[3] - points[2]) < vector_slope(
                    points[2] - points[1]) + vector_slope(points[0] -
                                                                points[3]):
            horizontal_edge_inds = [[0, 1], [2, 3]]
            vertical_edge_inds = [[3, 0], [1, 2]]
        else:
            horizontal_edge_inds = [[3, 0], [1, 2]]
            vertical_edge_inds = [[0, 1], [2, 3]]

        vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                points[vertical_edge_inds[0][1]]) + norm(
                                    points[vertical_edge_inds[1][0]] -
                                    points[vertical_edge_inds[1][1]])
        horizontal_len_sum = norm(
            points[horizontal_edge_inds[0][0]] -
            points[horizontal_edge_inds[0][1]]) + norm(
                points[horizontal_edge_inds[1][0]] -
                points[horizontal_edge_inds[1][1]])

        if vertical_len_sum > horizontal_len_sum * orientation_thr:
            head_inds = horizontal_edge_inds[0]
            tail_inds = horizontal_edge_inds[1]
        else:
            head_inds = vertical_edge_inds[0]
            tail_inds = vertical_edge_inds[1]

    return head_inds, tail_inds



def clockwise(head_edge, tail_edge, top_sideline, bot_sideline):
    hc = head_edge.mean(axis=0)
    tc = tail_edge.mean(axis=0)
    d = (((hc - tc) ** 2).sum()) ** 0.5 + 0.1
    dx = np.abs(hc[0] - tc[0])
    if not dx / d <= 1:
        print(dx / d)
    angle = np.arccos(dx / d)
    PI = 3.1415926
    direction = 0 if angle <= PI / 4 else 1  # 0 horizontal, 1 vertical
    if top_sideline[0, direction] > top_sideline[-1, direction]:
        top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
    else:
        top_indx = np.arange(0, top_sideline.shape[0])
    top_sideline = top_sideline[top_indx]
    if direction == 1 and top_sideline[0, direction] < top_sideline[-1, direction]:
        top_indx = np.arange(top_sideline.shape[0] - 1, -1, -1)
        top_sideline = top_sideline[top_indx]

    if bot_sideline[0, direction] > bot_sideline[-1, direction]:
        bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
    else:
        bot_indx = np.arange(0, bot_sideline.shape[0])
    bot_sideline = bot_sideline[bot_indx]
    if direction == 1 and bot_sideline[0, direction] < bot_sideline[-1, direction]:
        bot_indx = np.arange(bot_sideline.shape[0] - 1, -1, -1)
        bot_sideline = bot_sideline[bot_indx]
    if top_sideline[:, 1 - direction].mean() > bot_sideline[:, 1 - direction].mean():
        top_sideline, bot_sideline = bot_sideline, top_sideline

    return top_sideline, bot_sideline, direction
    

def reorder_poly_edge(points):

    assert points.ndim == 2
    assert points.shape[0] >= 4
    assert points.shape[1] == 2

    head_edge, tail_edge, top_sideline, bot_sideline = reorder_poly_edge2(points)
    top_sideline, bot_sideline,_ = clockwise(head_edge, tail_edge, top_sideline, bot_sideline)
   
    return top_sideline, bot_sideline[::-1]

def get_gradient2(im):  # 返回的是一个map，这个map是bool型的，map将mask的边缘标记为true
    dist_map = cv2.distanceTransform(im, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # 计算每个点到最近的背景点的距离
    dd = (dist_map > 0).astype(np.uint8) + (dist_map < 2).astype(np.uint8)
    edges = dd == 2  # TD : use this edges, which provides more info than another one
    return edges

def get_gradient(im):
    h, w = im.shape[0], im.shape[1]
    # im = add_edge(im)
    instance_id = np.unique(im)[1]
    # delete line
    mask = np.zeros((im.shape[0], im.shape[1]))
    mask.fill(instance_id)
    boolmask = (im == mask)
    im = im * boolmask  # only has object

    y = np.gradient(im)[0]
    x = np.gradient(im)[1]
    gradient = abs(x) + abs(y)
    bool_gradient = gradient.astype(bool)
    mask.fill(1)
    gradient_map = mask * bool_gradient * boolmask
    # gradient_map = gradient_map[5:h+5,5:w+5]
    # 2d gradient map
    return gradient_map

def inner_dot(instance_mask, point):
    xp, yp = point
    h, w = instance_mask.shape
    bool_inst_mask = instance_mask.astype(bool)
    neg_bool_inst_mask = 1 - bool_inst_mask
    dot_mask = np.zeros(instance_mask.shape)
    insth, instw = instance_mask.shape
    dot_mask[yp][xp] = 1
    if yp + 1 >= h or yp - 1 < 0 or xp + 1 >= w or xp - 1 < 0:
        return False
    fill_mask = np.zeros((3, 3))
    fill_mask.fill(1)
    dot_mask[yp - 1: yp + 2, xp - 1: xp + 2] = fill_mask
    not_inner = (neg_bool_inst_mask * dot_mask).any()
    # print(np.sum(neg_bool_inst_mask),np.sum(dot_mask))
    # print('neg_bool',np.unique(dot_mask))
    return not not_inner

def centerdot(instance_mask):
    # boundingorder x, y
    bool_inst_mask = instance_mask.astype(bool)
    x, y, w, h = cv2.boundingRect(instance_mask)
    avg_center_float = (x + w / 2, y + h / 2)  # w,h
    avg_center = (int(avg_center_float[0]), int(avg_center_float[1]))
    temp = np.zeros(instance_mask.shape)
    temp[int(avg_center[1])][int(avg_center[0])] = 1
    if (bool_inst_mask == temp).any() and inner_dot(instance_mask, avg_center):
        return avg_center_float
    else:

        inst_mask_h, inst_mask_w = np.where(instance_mask)

        # get gradient_map
        gradient_map = get_gradient(instance_mask)
        grad_h, grad_w = np.where(gradient_map == 1)

        # inst_points
        inst_points = np.array(
            [[inst_mask_w[i], inst_mask_h[i]] for i in range(len(inst_mask_h))]
        )
        # edge_points
        bounding_order = np.array([[grad_w[i], grad_h[i]] for i in range(len(grad_h))])

        distance_result = distance.cdist(inst_points, bounding_order, "euclidean")
        sum_distance = np.sum(distance_result, 1)
        center_index = np.argmin(sum_distance)

        center_distance = (inst_points[center_index][0], inst_points[center_index][1])
        times_num = 0
        while not inner_dot(instance_mask, center_distance):
            times_num += 1
            sum_distance = np.delete(sum_distance, center_index)
            if len(sum_distance) == 0:
                print("no center")
                # raise TOsmallError

            center_index = np.argmin(sum_distance)
            center_distance = (
                inst_points[center_index][0],
                inst_points[center_index][1],
            )
        return center_distance

def get_centroid(instance_mask):
    props = measure.regionprops(measure.label(instance_mask))
    center_y, center_x = props[0].centroid
    # outside the seg mask, use inscribed center
    if instance_mask[int(center_y), int(center_x)] == 0:
        center_x, center_y = get_inscribed_center(instance_mask)
    return center_x, center_y

def get_inscribed_center(instance_mask):
    dist_map = cv2.distanceTransform(instance_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, _, _, center = cv2.minMaxLoc(dist_map)
    center_x, center_y = center
    return center_x, center_y

def get_center_hybrid(instance_mask):
    props = measure.regionprops(measure.label(instance_mask))
    sdt = props[0].solidity
    if sdt > 0.85:
        center_x, center_y = get_centroid(instance_mask)
    else:
        center_x, center_y = get_inscribed_center(instance_mask)
    return center_x, center_y

def to_np(data):
    return data.cpu().numpy()

def trans_polarone_to_another(ori_deg, assisPolar, center_coord, im_shape):  # 计算中心点到mask边界上的点的距离r，这个r是试出来的
    """
    make sure that the r,theta you want to assis not outof index
    assisPolar = (r,deg)
    center_coord = (center_x,center_y)
    """
    assis_r = np.array(assisPolar[0], np.float32)
    ori_deg = np.array(ori_deg, np.float32)
    x = -1
    y = -1
    while not (x >= 0 and x < im_shape[1] and y >= 0 and y < im_shape[0]):
        x, y = cv2.polarToCart(assis_r, ori_deg, angleInDegrees=True)
        x += center_coord[0]
        y += center_coord[1]
        x = int(x)
        y = int(y)
        ori_r = assis_r
        assis_r -= 0.1
    return ori_r

def getOrientedPoints(instance):  # 返回中心点坐标和中心点到mask边缘上点的距离r
    # first get center point
    instance = instance.astype(np.uint8)

    center_x, center_y = get_center_hybrid(instance)
    # your implementation of get gradient, it is a bool map
    edges = get_gradient2(instance)
    index_h, index_w = np.where(edges == 1)
    centerpoints_array = np.array([center_x, center_y])
    # distance_all = distance.cdist(edgepoints_array,centerpoints_array,'euclidean')

    edgeDict = (
        {}
    )  # we create a dict for which key = 0, 1,2,3,...359 value list of distance
    # generate empty list for all the angle
    for i in range(360):
        edgeDict[str(i)] = []
    for i in range(len(index_h)):
        # # calculate the degree based on center point
        # clockwise
        # i want to get a deg section of each points
        deg_1 = -np.arctan2(1, 0) + np.arctan2(
            index_w[i] - center_x, index_h[i] - center_y
        )
        deg_1 = deg_1 * 180 / np.pi
        if deg_1 < 0:
            deg_1 += 360
        deg_2 = -np.arctan2(1, 0) + np.arctan2(
            index_w[i] + 1 - center_x, index_h[i] + 1 - center_y
        )
        deg_2 = deg_2 * 180 / np.pi
        if deg_2 < 0:
            deg_2 += 360
        deg_3 = -np.arctan2(1, 0) + np.arctan2(
            index_w[i] - center_x, index_h[i] + 1 - center_y
        )
        deg_3 = deg_3 * 180 / np.pi
        if deg_3 < 0:
            deg_3 += 360
        deg_4 = -np.arctan2(1, 0) + np.arctan2(
            index_w[i] + 1 - center_x, index_h[i] - center_y
        )
        deg_4 = deg_4 * 180 / np.pi
        if deg_4 < 0:
            deg_4 += 360
        deg1 = min(deg_1, deg_2, deg_3, deg_4)
        deg2 = max(deg_1, deg_2, deg_3, deg_4)
        # calculate distance
        dot_array = np.array([index_w[i], index_h[i]])
        distance_r = np.linalg.norm(dot_array - centerpoints_array)
        # consider when deg = 0
        if int(deg2 - deg1) > 100:
            for deg in range(0, math.ceil(deg1)):
                edgeDict[str(int(deg))].append(distance_r)
            for deg in range(math.ceil(deg2), 360):
                edgeDict[str(int(deg))].append(distance_r)
        else:
            for deg in range(math.ceil(deg1), math.ceil(deg2)):
                edgeDict[str(int(deg))].append(distance_r)

    """
    change start_points
    """
    # find the largest r for each deg
    try:
        edgeDict = {k: np.max(np.array(edgeDict[k])) for k in edgeDict.keys()}
    except ValueError:
        for index_deg in range(360):
            if len(edgeDict[str(index_deg)]) == 0:
                search_deg = index_deg
                while len(edgeDict[str(search_deg % 360)]) == 0:
                    search_deg += 1
                search_info = edgeDict[str(search_deg % 360)]

                for r_info in search_info:
                    # TD: should be the order of (r, deg) !!
                    assisPolar = (r_info, search_deg % 360)
                    center_coord = (center_x, center_y)
                    trans_r = trans_polarone_to_another(
                        index_deg, assisPolar, center_coord, instance.shape
                    )
                    edgeDict[str(index_deg)].append(trans_r)
        edgeDict = {k: np.max(np.array(edgeDict[k])) for k in edgeDict.keys()}
    points = [edgeDict[str(deg_num)] for deg_num in range(360)]  # start 0 deg

    return points, center_x, center_y

def comp_iou(mask1, mask2):
    mask_overlap = mask1.astype(np.uint8) + mask2.astype(np.uint8)
    non_lap = (mask_overlap == 1).astype(np.uint8)
    over_lap = (mask_overlap == 2).astype(np.uint8)
    iou = over_lap.sum() / (over_lap.sum() + non_lap.sum())
    return iou

def f( x, y, x1, x2, y1, y2):  # 判断点（x,y）在（x1,y1）和（x2,y2）这两个点组成的直线的哪个位置：返回的是正数则在直线的左侧，负数则是右侧，为0表示就在直线上
    # line equation give (x1, y1), (x2, y2)
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def reorg_cc(part1, part2):
    output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    stats = output[2]

    loc = []
    for i in range(num_labels):
        loc.append(output[1] == i)

    areas = stats[:, -1]
    ind = np.argmax(areas[1:])
    ind += 1

    keep = np.zeros(num_labels)
    keep[0] = 1
    keep[ind] = 1

    for i in range(num_labels):
        if keep[i] == 0:
            part2[loc[i]] = 1
            part1[loc[i]] = 0

    return part1, part2

def hierarchy_encoding(img_size, text_mask, node_num, result, max_depth=1):
    mask = text_mask
    h, w = img_size

    instance_second = np.copy(mask)
    if max_depth == 0:
        points, center_x, center_y = getOrientedPoints(instance_second)  # 这里的points其实是r

        centers = np.array([center_x, center_y]).astype(np.float32)
        center_xs = centers[0]  # 1
        center_ys = centers[1]  # 1

        # center_xs = center_x  # 1
        # center_ys = center_y  # 1

        # pdb.set_trace()
        idx = np.linspace(0, 360, node_num, endpoint=False).astype(np.int32)
        # rs = np.float32(data['r'])[idx]  # N, 360
        rs = np.float32(points)[idx]  # N, 360
        num = len(rs)
        rs = rs.astype(np.float32)  # N, 360
        # theta_list = np.arange(num - 1, -1, -1).reshape(num).astype(np.float32) # 360
        theta_list = np.flip(idx, axis=0).astype(np.float32)
        x, y = cv2.polarToCart(rs, theta_list, angleInDegrees=True)  # 360    360
        x = x + center_xs.astype(np.float32)  # 360
        y = y + center_ys.astype(np.float32)  # 360

        # x = np.clip(x, bboxs_x1, bboxs_x2).reshape(num, 1)  # 360,1
        # y = np.clip(y, bboxs_y1, bboxs_y2).reshape(num, 1)  # 360,1

        x = x.reshape(num, 1)  # 360,1
        y = y.reshape(num, 1)  # 360,1
        polygons = np.concatenate((x, y), axis=-1)  # 360,2
        result.append(polygons)
        return result

    y, x = np.where(mask == 1)
    data = np.array([(x[k], y[k]) for k in range(len(y))])

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data.astype(np.float32), mean)
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    p2 = cntr - 0.1 * eigenvectors[1, :] * eigenvalues[1, 0]

    x1, y1 = cntr
    x2, y2 = p2

    part1 = np.zeros_like(mask)
    part2 = np.zeros_like(mask)
    for d in data:
        if f(d[0], d[1], x1, x2, y1, y2) >= 0:
            part1[d[1], d[0]] = 1
        elif f(d[0], d[1], x1, x2, y1, y2) < 0:
            part2[d[1], d[0]] = 1

    output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S)  # 这里返回的连通域的数目是包括了背景在里面的
    num_labels = output[0]
    if num_labels > 2:  # 所以这里的意思是如果去除背景之后，有一个以上的连通域就进行这个操作
        part1, part2 = reorg_cc(part1, part2)  # 这个操作的作用就是确保一刀下去只分成两份，如果分成了超过两份就把其余面积比较小的部分加到面积最大的两个部分的其中之一

    output = cv2.connectedComponentsWithStats(part2.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    if num_labels > 2:
        part1, part2 = reorg_cc(part2, part1)

    output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    assert num_labels == 2

    output = cv2.connectedComponentsWithStats(part2.astype(np.uint8), 4, cv2.CV_32S)
    num_labels = output[0]
    assert num_labels == 2

    hierarchy_encoding((h, w), part1, node_num, result, max_depth - 1)
    hierarchy_encoding((h, w), part2, node_num, result, max_depth - 1)


totaltext_train = json.load(open("data/totaltext/totaltext_train.json"))
anno = totaltext_train['annotations']
img_info = totaltext_train['images']
resample_lines = []

for i in tqdm(range(len(anno))):
    result = []
    anno_i = anno[i]['segmentation'][0]
    img_id = anno[i]['image_id']
    h = img_info[img_id]['height']
    w = img_info[img_id]['width']
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon = np.array(anno_i).reshape(-1,2).astype(np.float32)
    polygon = polygon.reshape((1, -1, 2))
    
    #if polygon.shape[0] % 2 !=0:
    #    continue

    cv2.fillPoly(mask, np.round(polygon).astype(np.int32), 1)
    # pdb.set_trace()
    # print('generate_lra_maps')
    hierarchy_encoding((h, w), mask, 360, result, max_depth=1)
    # print(result.shape)
    resample_line1 = result[0].flatten()
    resample_line2 = result[1].flatten()
    resample_line1 = np.expand_dims(resample_line1,axis=0)
    resample_line2 = np.expand_dims(resample_line2,axis=0)
    #resample_line = np.concatenate((result[0], result[1]), axis=0).flatten()
    # top_sideline, bot_sideline = reorder_poly_edge(polygon)
    #resample_line = resample_polygon(top_sideline, bot_sideline, 7)
    # resample_line = resample_polygon(top_sideline, bot_sideline, 180)
    resample_lines.append(resample_line1)
    resample_lines.append(resample_line2)


resample_lines = np.concatenate(resample_lines,axis=0)

#n_components = 16
n_components = 28
pca = IncrementalPCA(n_components=n_components, copy=False)
pca.fit(resample_lines)
components_c = pca.components_.astype(np.float32)
output_path = os.path.join('pca_' +str(n_components)
                            + '.npz')
print("Save the pca matrix: " + output_path)
np.savez(output_path,
            components_c=components_c,)
