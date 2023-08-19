import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import shutil

from util_class import AxisDetect
from ocr_api_infermodel import run_first_det_api, post_process_first_det_api, ocr_det_api, ocr_rec_api, post_process

def rotate_img_func(img, angle):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    #获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    #计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    #调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img

def draw_first_task(image, circleaxis, select_sector):
    image_draft = image.copy()
    tgt_height, tgt_width, _ = image_draft.shape
    
    circle_mask = np.zeros([tgt_height, tgt_width])
    cv2.circle(circle_mask, (int(circleaxis.centx), int(circleaxis.centy)), int(circleaxis.radius), 1, -1)
    circle_mask = circle_mask.astype('uint8')
    circle_mask_inv = np.where(circle_mask, 0, 1).astype('uint8')
    
    image_background = image_draft * circle_mask_inv[:, :, None]
    image_foreground = image_draft * circle_mask[:, :, None]
    
    radius = circleaxis.radius + 5
    rotate_image_foreground = image_foreground[circleaxis.centy - radius : circleaxis.centy + radius, circleaxis.centx - radius : circleaxis.centx + radius]
    rotate_image_foreground = rotate_img_func(rotate_image_foreground, select_sector.rotate_angle)
    h, w, _ = rotate_image_foreground.shape
    rotate_image_foreground = rotate_image_foreground[int(h/2) - radius: int(h/2) + radius, int(w/2) - radius: int(w/2) + radius]
    image_foreground[circleaxis.centy - radius : circleaxis.centy + radius, circleaxis.centx - radius : circleaxis.centx + radius] = rotate_image_foreground
    
    image_cat = image_foreground + image_background
    return np.hstack([image, image_cat])

def draw_word(image, _transcriptions_str, saveroot):
    height, width, _ = image.shape
    cy, cx = int(height/2), int(width/2)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.imshow(image)
    ax.text(x=cx, y=cy, s=_transcriptions_str, ha='center', va='baseline',  color='red')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig(osp.join(saveroot, 'final_word.png'))
    return cv2.imread(osp.join(saveroot, 'final_word.png'))
    

def main():
    imroot = '../../../2023_06_organize_data/2023_06_12'
    #imroot = '2023_06_organize_data/2023_06_13'
    saveroot = 'output/test_debug/'
    if not osp.exists(saveroot):
        os.makedirs(saveroot)

    imlist = os.listdir(imroot)
    imlist = list(filter(lambda x: x[-4:] == '.png', imlist))
    
    axisDetect = AxisDetect()
    for idx, imname in enumerate(imlist):
        imgname = osp.join(imroot, imname)
        image = cv2.imread(imgname)
        
        # first step detect axis
        circleaxis, log_info = axisDetect.infer(imgname)
        print(log_info)
        if circleaxis is None or len(circleaxis.axis_angle) == 0:
            print(f"{idx}-step first step axis failed: {imgname}")
            continue
        sectors = circleaxis.cal_sector()
        # import ipdb;ipdb.set_trace()
        # for sector in sectors:
        #     cont = sector.points
        #     cv2.drawContours(image, [np.array(cont).astype('int')], 0, (0, 255, 0), 5)
        
        # infer first det
        det_box_json = run_first_det_api(imgname)
        circleaxis.modify_det_box(det_box_json)
        for det_box in det_box_json['boxes']:
            for sector in sectors:
                sector.cal_contail_ocr(det_box)
        if len(det_box_json['boxes']) >= 3:
           print("first detect error more than 3 circle {imname}, no rotate")
           image_cat = np.hstack([image, image])
        else:
            select_sectors = list(filter(lambda x: x.with_det==False, sectors))
            select_sector = select_sectors[0]
            image_cat = draw_first_task(image, circleaxis, select_sector)
        # cv2.imwrite(osp.join(saveroot, imname), image_cat)
        
        # infer second det
        det_sectors =  list(filter(lambda x: x.with_det==True, sectors))
        for sector_i, sector in enumerate(det_sectors):
            post_process_first_det_api(sector_i, image, sector.det_box, saveroot)
            second_dt_boxes_json, first_det_filename = ocr_det_api(sector_i, saveroot)
            sector.ocr_boxes = second_dt_boxes_json
            sector.first_det_file = first_det_filename
        
        core_sector = sorted(det_sectors, key=lambda x: -len(x.ocr_boxes))[0]
        core_sector.is_core = True
        shutil.copyfile(core_sector.first_det_file, osp.join(saveroot, 'first_det_result.png'))
        # infer ocr rec
        _transcriptions_str = ocr_rec_api(core_sector.ocr_boxes, saveroot)
        _transcriptions_str = post_process(core_sector.ocr_boxes)[1]
        core_sector._transcriptions_str = _transcriptions_str
        
        final_image = draw_word(image, _transcriptions_str, saveroot)
        _height = image_cat.shape[0]
        _width = int(_height / final_image.shape[0] * final_image.shape[1])
        final_image = cv2.resize(final_image, (_width, _height))
        image_cat = np.hstack([image_cat, final_image])
        cv2.imwrite(osp.join(saveroot, imname), image_cat)
        
if __name__ == '__main__':
    main()
