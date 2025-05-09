import Polygon
import cv2
import numpy as np
from numpy.linalg import norm
import mmocr.utils.check_argument as check_argument
from .textsnake_targets import TextSnakeTargets
from scipy.interpolate import splprep, splev
PI = 3.1415926
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
import math
from skimage import measure
from PIL import Image, ImageDraw
import torch
import copy

@PIPELINES.register_module()
class LRATargetsNew(TextSnakeTargets):

    def __init__(self,
                 path_lra,
                 num_coefficients=14,
                 num_points=360,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0)),
                 num_samples = 3,
                 with_area=True,
                 ):
        super().__init__()
        assert isinstance(level_size_divisors, tuple)
        assert isinstance(level_proportion_range, tuple)
        assert len(level_size_divisors) == len(level_proportion_range)
        self.with_area = with_area
        self.num_samples = num_samples
        self.num_coefficients = num_coefficients
        self.num_points = num_points * 2
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = level_size_divisors
        self.level_proportion_range = level_proportion_range
        U_t = np.load(path_lra)['components_c']
        print('U_t.shape:{}'.format(U_t.shape))
        self.U_t = U_t

    def get_gradient2(self, im): # 返回的是一个map，这个map是bool型的，map将mask的边缘标记为true
        dist_map = cv2.distanceTransform(im, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) # 计算每个点到最近的背景点的距离
        dd = (dist_map > 0).astype(np.uint8) + (dist_map < 2).astype(np.uint8)
        edges = dd == 2  # TD : use this edges, which provides more info than another one
        return edges

    def get_gradient(self, im):
        h, w = im.shape[0], im.shape[1]
        #im = add_edge(im)
        instance_id = np.unique(im)[1]
        # delete line
        mask = np.zeros((im.shape[0], im.shape[1]))
        mask.fill(instance_id)
        boolmask = (im == mask)
        im = im * boolmask  # only has object

        y = np.gradient(im)[0]
        x = np.gradient(im)[1]
        gradient = abs(x)+abs(y)
        bool_gradient = gradient.astype(bool)
        mask.fill(1)
        gradient_map = mask*bool_gradient*boolmask
        #gradient_map = gradient_map[5:h+5,5:w+5]
        # 2d gradient map
        return gradient_map

    def inner_dot(self, instance_mask, point):
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
        dot_mask[yp - 1 : yp + 2, xp - 1 : xp + 2] = fill_mask
        not_inner = (neg_bool_inst_mask * dot_mask).any()
        # print(np.sum(neg_bool_inst_mask),np.sum(dot_mask))
        # print('neg_bool',np.unique(dot_mask))
        return not not_inner

    def centerdot(self, instance_mask):
        # boundingorder x, y
        bool_inst_mask = instance_mask.astype(bool)
        x, y, w, h = cv2.boundingRect(instance_mask)
        avg_center_float = (x + w / 2, y + h / 2)  # w,h
        avg_center = (int(avg_center_float[0]), int(avg_center_float[1]))
        temp = np.zeros(instance_mask.shape)
        temp[int(avg_center[1])][int(avg_center[0])] = 1
        if (bool_inst_mask == temp).any() and self.inner_dot(instance_mask, avg_center):
            return avg_center_float
        else:

            inst_mask_h, inst_mask_w = np.where(instance_mask)

            # get gradient_map
            gradient_map = self.get_gradient(instance_mask)
            grad_h, grad_w = np.where(gradient_map == 1)

            # inst_points
            inst_points = np.array(
                [[inst_mask_w[i], inst_mask_h[i]] for i in range(len(inst_mask_h))]
            )
            # edge_points
            bounding_order = np.array([[grad_w[i], grad_h[i]] for i in range(len(grad_h))])

            distance_result = self.distance.cdist(inst_points, bounding_order, "euclidean")
            sum_distance = np.sum(distance_result, 1)
            center_index = np.argmin(sum_distance)

            center_distance = (inst_points[center_index][0], inst_points[center_index][1])
            times_num = 0
            while not self.inner_dot(instance_mask, center_distance):
                times_num += 1
                sum_distance = np.delete(sum_distance, center_index)
                if len(sum_distance) == 0:
                    print("no center")
                    #raise TOsmallError

                center_index = np.argmin(sum_distance)
                center_distance = (
                    inst_points[center_index][0],
                    inst_points[center_index][1],
                )
            return center_distance


    def get_centroid(self, instance_mask):
        props = measure.regionprops(measure.label(instance_mask))
        center_y, center_x = props[0].centroid
        # outside the seg mask, use inscribed center
        if instance_mask[int(center_y), int(center_x)] == 0:
            center_x, center_y = self.get_inscribed_center(instance_mask)
        return center_x, center_y


    def get_inscribed_center(self, instance_mask):
        dist_map = cv2.distanceTransform(instance_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, _, _, center = cv2.minMaxLoc(dist_map)
        center_x, center_y = center
        return center_x, center_y


    def get_center_hybrid(self, instance_mask):
        props = measure.regionprops(measure.label(instance_mask))
        sdt = props[0].solidity
        if sdt > 0.85:
            center_x, center_y = self.get_centroid(instance_mask)
        else:
            center_x, center_y = self.get_inscribed_center(instance_mask)
        return center_x, center_y


    def to_np(data):
        return data.cpu().numpy()
    
    def trans_polarone_to_another(self, ori_deg, assisPolar, center_coord, im_shape): # 计算中心点到mask边界上的点的距离r，这个r是试出来的
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

    def getOrientedPoints(self, instance): # 返回中心点坐标和中心点到mask边缘上点的距离r
    # first get center point
        instance = instance.astype(np.uint8)

        center_x, center_y = self.get_center_hybrid(instance)
        # your implementation of get gradient, it is a bool map
        edges = self.get_gradient2(instance)
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
                        trans_r = self.trans_polarone_to_another( 
                            index_deg, assisPolar, center_coord, instance.shape
                        )
                        edgeDict[str(index_deg)].append(trans_r)
            edgeDict = {k: np.max(np.array(edgeDict[k])) for k in edgeDict.keys()}
        points = [edgeDict[str(deg_num)] for deg_num in range(360)]  # start 0 deg

        return points, center_x, center_y

    def comp_iou(self, mask1, mask2):
        mask_overlap = mask1.astype(np.uint8) + mask2.astype(np.uint8)
        non_lap = (mask_overlap == 1).astype(np.uint8)
        over_lap = (mask_overlap == 2).astype(np.uint8)
        iou = over_lap.sum() / (over_lap.sum() + non_lap.sum())
        return iou
       
    def f(self, x, y, x1, x2, y1, y2): # 判断点（x,y）在（x1,y1）和（x2,y2）这两个点组成的直线的哪个位置：返回的是正数则在直线的左侧，负数则是右侧，为0表示就在直线上
        # line equation give (x1, y1), (x2, y2)
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def reorg_cc(self, part1, part2):
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

    def hierarchy_encoding(self, img_size, text_mask, node_num, result, max_depth=1):
        mask = text_mask
        h, w = img_size

        instance_second = np.copy(mask)
        if max_depth == 0:
            points, center_x, center_y = self.getOrientedPoints(instance_second) # 这里的points其实是r
            
            centers = np.array([center_x, center_y]).astype(np.float32)
            center_xs = centers[0]  # 1
            center_ys = centers[1]  # 1

            #center_xs = center_x  # 1
            #center_ys = center_y  # 1

            #pdb.set_trace()
            idx = np.linspace(0, 360, node_num, endpoint=False).astype(np.int32)
            #rs = np.float32(data['r'])[idx]  # N, 360
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
            if self.f(d[0], d[1], x1, x2, y1, y2) >= 0:
                part1[d[1], d[0]] = 1
            elif self.f(d[0], d[1], x1, x2, y1, y2) < 0:
                part2[d[1], d[0]] = 1

        output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S) # 这里返回的连通域的数目是包括了背景在里面的
        num_labels = output[0]
        if num_labels > 2: # 所以这里的意思是如果去除背景之后，有一个以上的连通域就进行这个操作
            part1, part2 = self.reorg_cc(part1, part2) # 这个操作的作用就是确保一刀下去只分成两份，如果分成了超过两份就把其余面积比较小的部分加到面积最大的两个部分的其中之一

        output = cv2.connectedComponentsWithStats(part2.astype(np.uint8), 4, cv2.CV_32S)
        num_labels = output[0]
        if num_labels > 2:
            part1, part2 = self.reorg_cc(part2, part1)

        output = cv2.connectedComponentsWithStats(part1.astype(np.uint8), 4, cv2.CV_32S)
        num_labels = output[0]
        assert num_labels == 2

        output = cv2.connectedComponentsWithStats(part2.astype(np.uint8), 4, cv2.CV_32S)
        num_labels = output[0]
        assert num_labels == 2

        self.hierarchy_encoding(img_size, part1, node_num, result, max_depth - 1)
        self.hierarchy_encoding(img_size, part2, node_num, result, max_depth - 1)

    def generate_center_region_mask(self, img_size, text_polys):# 就是生成了一个被从高度上压缩了的区域的mask
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """
        
        #pdb.set_trace()
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size # 120, 120
        #print('h={}, w={}:'.format(h, w))
        center_region_mask = np.zeros((h, w), np.uint8)
        center_region_boxes = []
        for poly in text_polys:
            assert len(poly) == 1
            polygon_points = poly[0].reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            # resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                    resampled_top_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                    resampled_bot_line[i + 1] -
                    center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def generate_lra_maps(self, img_size, text_polys,text_polys_idx=None, img=None, level_size=None):
        #pdb.set_trace()
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size # 60,60
        coeff_maps = np.zeros((self.num_coefficients * 2, h, w), dtype=np.float32) ##############
        for poly,poly_idx in zip(text_polys, text_polys_idx):# len(text_polys)表示这张图片中有多少个文本
            assert len(poly) == 1
            result = []
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(text_instance).reshape((1, -1, 2))
            cv2.fillPoly(mask, np.round(polygon).astype(np.int32), 1)
            self.hierarchy_encoding(img_size, mask, self.num_points // 2, result, max_depth=1)
            outline1 = result[0].flatten()
            outline2 = result[1].flatten()
            outline1 = np.expand_dims(outline1, axis=1)
            outline2 = np.expand_dims(outline2, axis=1)
            lra_coeff1 = np.matmul(self.U_t, outline1)
            lra_coeff2 = np.matmul(self.U_t, outline2)
            lra_coeff = np.concatenate((lra_coeff1, lra_coeff2), axis=0)
            # resample_line = np.concatenate((result[0], result[1]), axis=0).flatten()
            # resample_line = np.expand_dims(resample_line, axis=1)
            # lra_coeff = resample_line
            # U_t_merge = np.concatenate((self.U_t, self.U_t), axis=1)
            # lra_coeff = np.matmul(U_t_merge, resample_line)
            #print(f'lra_coeff:{lra_coeff}')
            yx = np.argwhere(mask > 0.5)
            y, x = yx[:, 0], yx[:, 1]
            batch_T = np.zeros((h, w, self.num_coefficients, 2))
            batch_T[y,x,:,:] = lra_coeff.reshape(-1,2) # 文本区域上的每个点都对应一个lra_coeff
            #print(f'batch_T:{batch_T[y,x,:,:]}')
            batch_T = batch_T.reshape(h, w, -1).transpose(2, 0, 1)
            coeff_maps[:, y,x] = batch_T[:,y,x]
            #print(f'coeff_maps:{coeff_maps[:,y,x]}')

        return coeff_maps # 最后返回的coeff_maps中包含了各个文本区域所对应的lra_coeff，不同的文本区域对应不同的lra_coeff

    def generate_text_region_mask(self, img_size, text_polys, text_polys_idx): #为文本区域生成掩码，用像素值1，2，3等区分开来
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """
        #pdb.set_trace()
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly, poly_idx in zip(text_polys, text_polys_idx):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(
                np.round(text_instance), dtype=np.int).reshape((1, -1, 2))
            if self.with_area:
                cv2.fillPoly(text_region_mask, polygon, poly_idx)
            else:
                cv2.fillPoly(text_region_mask, polygon, 1)
        return text_region_mask

    def generate_level_targets(self, img_size, text_polys, ignore_polys,img=None):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
            :param img:
        """
        #pdb.set_trace()
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_text_polys_idx = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        polygons_area = []
        level_maps = []
        for poly_idx, poly in enumerate(text_polys):
            assert len(poly) == 1
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            tl_x, tl_y, box_w, box_h = cv2.boundingRect(polygon)

            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append([poly[0] / lv_size_divs[ind]])
                    lv_text_polys_idx[ind].append(poly_idx+1)

            if self.with_area:
                polygon_area = Polygon.Polygon(poly[0].reshape(-1,2)).area()
                polygons_area.append(polygon_area)


        for ignore_poly in ignore_polys:
            assert len(ignore_poly) == 1
            text_instance = [[ignore_poly[0][i], ignore_poly[0][i + 1]]
                             for i in range(0, len(ignore_poly[0]), 2)]
            polygon = np.array(text_instance, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_ignore_polys[ind].append(
                        [ignore_poly[0] / lv_size_divs[ind]])


        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)
            text_region = self.generate_text_region_mask(
                level_img_size, lv_text_polys[ind], lv_text_polys_idx[ind])[None]
            current_level_maps.append(text_region)

            center_region = self.generate_center_region_mask(
                    level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)

            effective_mask = self.generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])[None] # 把mask中的polygons_ignore部分设置为0，其余部分为1
            current_level_maps.append(effective_mask)

            lra_coeff_maps = self.generate_lra_maps(
                level_img_size, lv_text_polys[ind],lv_text_polys_idx[ind])

            current_level_maps.append(lra_coeff_maps)
            level_maps.append(np.concatenate(current_level_maps)) # 在axis=0上把这些maps拼接到一起

        transformed_polys = []
        for j in range(len(text_polys)):
            result = []
            polygon = text_polys[j][0].reshape(-1,2)
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = polygon.reshape((1, -1, 2))
            cv2.fillPoly(mask, np.round(polygon).astype(np.int32), 1)
            mask1 = copy.deepcopy(mask)
            self.hierarchy_encoding(img_size, mask, self.num_points // 2, result, max_depth=1)
            outline1 = result[0].flatten()
            outline2 = result[1].flatten()
            outline1 = np.expand_dims(outline1, axis=1)
            outline2 = np.expand_dims(outline2, axis=1)
            lra_coeff1 = np.matmul(self.U_t, outline1)
            lra_coeff2 = np.matmul(self.U_t, outline2)
            transformed_poly1 = np.matmul(self.U_t.transpose(), lra_coeff1).flatten()
            transformed_poly2 = np.matmul(self.U_t.transpose(), lra_coeff2).flatten()
            #lra_coeff = np.concatenate((lra_coeff1, lra_coeff2), axis=0)
            #resample_line = np.concatenate((result[0], result[1]), axis=0).flatten()
            #resample_line = np.expand_dims(resample_line, axis=1)
            #lra_coeff = np.matmul(self.U_t, resample_line)
            #transformed_poly = np.matmul(self.U_t.transpose(), lra_coeff).flatten() # （28，1）
            #transformed_poly = resample_line
            transformed_poly = np.concatenate((transformed_poly1, transformed_poly2), axis=0)
            #mask2 = np.zeros((h, w), dtype=np.uint8)
            #polys = transformed_poly.reshape((1,-1,2))
            #cv2.fillPoly(mask2, np.round(polys).astype(np.int32), 1)
            #iou = self.comp_iou(mask1, mask2)
            #print(iou)
            transformed_polys.append(transformed_poly)    
        transformed_polys = np.array(transformed_polys)

        if transformed_polys.shape[0] > 0:
            transformed_polys = np.concatenate([transformed_polys] * self.num_samples, axis=0)
            
        if self.with_area and len(polygons_area) > 0:
            polygons_area = np.array(polygons_area)
        else:
            polygons_area = np.array([])

        return level_maps, polygons_area, transformed_polys

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """
        
        assert isinstance(results, dict)
        #
        #pdb.set_trace()
        #print('generate_targets')
        polygon_masks = results['gt_masks'].masks
        polygon_masks_ignore = results['gt_masks_ignore'].masks
        gt_texts = results['texts']
        h, w, _ = results['img_shape']

        level_maps, polygons_area, transformed_polys = self.generate_level_targets((h, w), polygon_masks,
                                                 polygon_masks_ignore, results['img'])
        
        mask = np.zeros((h, w))
        for polygon in transformed_polys:
            polygon = polygon.reshape((-1, 1, 2))
            polygon = polygon.astype(np.int32)
            cv2.polylines(mask, [polygon], isClosed=True, color=255, thickness=2)     
        cv2.imwrite('mask/'+results['img_info']['filename'], mask)

        mask1 = np.zeros((h, w))
        for polygon in polygon_masks:
            polygon = np.array(polygon)
            polygon = polygon.reshape((-1, 1, 2))
            polygon = polygon.astype(np.int32)
            cv2.polylines(mask1, [polygon], isClosed=True, color=255, thickness=2)  
        cv2.imwrite('mask1/'+results['img_info']['filename'], mask1)
        #print(results.keys())
        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        mapping = {
            'p3_maps': level_maps[0],
            'p4_maps': level_maps[1],
            'p5_maps': level_maps[2],
            'polygons_area': polygons_area,
            'gt_texts': DC(gt_texts, cpu_only=True),
            'lra_polys': transformed_polys
        }
        
        for key, value in mapping.items():
            results[key] = value

        return results