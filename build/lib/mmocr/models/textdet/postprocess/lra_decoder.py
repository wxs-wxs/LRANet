import torch
import numpy as np
import torch.nn.functional as F
from mmocr.core.evaluation.utils import boundary_iou
import cv2
import mmcv

flag = 1

def poly_nms(polygons, threshold, with_index=False):
    assert isinstance(polygons, list)
    keep_poly = []
    keep_index = []
    if len(polygons) != 0:
        polygons = np.array(polygons)
        scores = polygons[:, -1]
        sorted_index = np.argsort(scores)
        polygons = polygons[sorted_index]

        index = [i for i in range(polygons.shape[0])]
        vaild_index = np.ones(len(index))
        invalid_index = np.where(vaild_index==0)
        index = np.delete(index, invalid_index)

        while len(index) > 0:
            keep_poly.append(polygons[index[-1]].tolist())
            keep_index.append(sorted_index[index[-1]])
            A = polygons[index[-1]][:-1]
            index = np.delete(index, -1)

            iou_list = np.zeros((len(index), ))
            for i in range(len(index)):
                B = polygons[index[i]][:-1]
                iou_list[i] = boundary_iou(A, B)
            remove_index = np.where(iou_list > threshold)
            index = np.delete(index, remove_index)

    if with_index:
        return keep_poly, keep_index
    else:
        return keep_poly


def grow_mask(mask_ap_list):
    if len(mask_ap_list) == 0:
        return None
    elif len(mask_ap_list) == 1:
        return mask_ap_list[0]

    # only grow if more than one mask
    mask_ap = 0
    for mm in mask_ap_list:
        if len(mm) == 0:  # skip leaf node with no center found
            continue
        mask_ap = (mask_ap + mm).clip(0, 1).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_ap = cv2.morphologyEx(mask_ap, cv2.MORPH_CLOSE, kernel)
    return mask_ap

def generate_mask(poly_list, img_size):
    h, w = img_size
    poly_part_list = []
    poly_part_list1 = poly_list[:360, :]
    poly_part_list.append(poly_part_list1)
    poly_part_list2 = []
    if len(poly_list) > 360:
        poly_part_list2 = poly_list[360:, :]
        poly_part_list.append(poly_part_list2)
    mask_list = []
    for poly_part in poly_part_list:
        text_part_mask = np.zeros((h, w), dtype=np.uint8)
        #poly_part = poly_part.reshape((-1, 1, 2)).cpu().numpy()
        poly_part = poly_part.reshape((-1, 1, 2)).cpu().numpy()
        hull = cv2.convexHull(poly_part)
        cv2.fillPoly(text_part_mask, np.round(hull).astype(np.int32), 1)
        #cv2.fillPoly(text_part_mask, np.round(poly_part).astype(np.int32), 1)
        mask_list.append(text_part_mask)
    return mask_list

def get_contour_points(mask):
    mask[mask==1] = 255
    # 查找mask中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None  # 如果没有找到轮廓，返回空
    
    # 选取最大的轮廓（如果有多个轮廓）
    contour = max(contours, key=cv2.contourArea)

    show_mask = np.zeros(mask.shape)
    cv2.fillPoly(show_mask, contours, 255)
    cv2.imwrite("/vhome/wangxuesheng/LRANet-main/mask/img1.jpg", show_mask)
    show_mask1 = mask
    show_mask1[show_mask1==1] = 255
    cv2.imwrite("/vhome/wangxuesheng/LRANet-main/mask/mask1.jpg", show_mask1)
    # 得到轮廓点，并进行顺时针排列
    epsilon = 0.01 * cv2.arcLength(contour, True)  # 对轮廓进行近似
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 返回顺时针排列的轮廓点
    return approx[:, 0, :]  # 提取x, y坐标

def unify_poly(polygons, number):
    for i, polygon in enumerate(polygons):
        # 计算需要插入的点的数量
        #print(polygon)
        insert_point_number = number - polygon.shape[0]
        
        if insert_point_number > 0:
            # 在每次插入之前，从头开始遍历以保持正确的索引
            for _ in range(insert_point_number):
                # 使用第一个点作为要插入的点
                insert_point_x = polygon[0, 0]
                insert_point_y = polygon[0, 1]
                intermediate_point = np.array([insert_point_x, insert_point_y])
                # 插入点，并更新 polygons 列表中的 polygon
                polygons[i] = np.insert(polygons[i], 1, intermediate_point, axis=0)
    
    return np.array(polygons)

def generate_hull(polygon):
    hull = cv2.convexHull(polygon)
    return hull

def lra_decode( preds, U_t,
                        scale,
                        score_thr=0.1, 
                        shift=0.1,
                        num_coefficients=28
                        ):
    assert isinstance(preds, list)

    tr_pred = preds[0][0][0].sigmoid()
    ssr_pred = preds[2][0][0].sigmoid()
    score_pred = tr_pred * ssr_pred   
    tr_pred_mask = score_pred > score_thr 
    k = 10
    top_k_values, top_k_indices = torch.topk(score_pred.view(-1), k)
    #tr_pred_mask = torch.ones(score_pred, dtype=torch.bool)
    boundaries = []
    reg_pred = preds[3][0].permute(1, 2, 0)
    lra_pred = reg_pred.flatten(0,1)
    lra_c = lra_pred[tr_pred_mask.reshape(-1)]
    rows, cols = tr_pred_mask.nonzero(as_tuple=True)
    xy_text = torch.stack((rows, cols), dim=1)
    lra_c1 = lra_c[:,:num_coefficients]
    lra_c2 = lra_c[:,num_coefficients:]
    polygons1 = torch.matmul(lra_c1, U_t)
    polygons2 = torch.matmul(lra_c2, U_t)
    polygons = torch.cat((polygons1, polygons2), dim=1)
    #polygons = torch.matmul(lra_c, U_t)
    #polygons = lra_c
    polygons = polygons.reshape(-1,polygons.shape[-1]//2,2)
    #print(polygons[0])
    #把预测出来的点先按照各个部分生成mask,然后再把mask融合。比如一个文本被分成了两部分，那么把这两部分多边形分别生成mask，再把他们的mask融合
    img_size =tuple(int(dim * scale) for dim in tr_pred.shape)
    #print(polygons[0])

    score = score_pred[tr_pred_mask].reshape(-1, 1)
    boundaries = []
    #if len(polygons) > 0:
    if polygons.shape[0] > 0:
        polygons[:, :, 0] += (xy_text[:, 1, None])  # 因为矩阵U和系数lra_c并不包含位置信息，事实上U和Lra_c乘除来的结果只是文本的轮廓而已，要找出文本轮廓在图像中的具体位置就要加上相对坐标
        polygons[:, :, 1] += (xy_text[:, 0, None])
        polygons += shift
        ###########################################q
        # 
        # text_polys_revise = []
        # for polygon in polygons:
        #     mask_list = generate_mask(polygon, img_size)
        #     text_mask = grow_mask(mask_list)
        #     poly = get_contour_points(text_mask)
        #     text_polys_revise.append(poly)
        # text_polys_revise = unify_poly(text_polys_revise, 360)
        # polygons = torch.from_numpy(text_polys_revise).to(xy_text.device).to(dtype=torch.float32)
        #polygons = polygons.reshape(polygons.shape[0], -1)
        ############################################
        polygons = polygons * scale
        hulls = []
        for polygon in polygons:
            polygon = polygon.reshape((-1, 1, 2)).cpu().numpy()
            hull = generate_hull(polygon)
            hull = np.array(hull).reshape(-1, 2)
            hulls.append(hull)
        polygons = unify_poly(hulls, 360)
        polygons = torch.from_numpy(polygons).to(xy_text.device)
        polygons = polygons.reshape(polygons.shape[0], -1)
        ############################################
        #polygons = polygons.reshape(polygons.shape[0], -1) * scale
        polygons2 = torch.cat((polygons, score), dim=1)
        polygons2 = polygons2.data.cpu().numpy().tolist() 
        boundaries = boundaries + polygons2
    return boundaries