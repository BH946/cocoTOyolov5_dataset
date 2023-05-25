import os
import shutil
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFile

###########################################

# 설정
INPUT_FOLDER_NAME = ["trash4_3(원본_3차분류_roboflow6개)\\train",
                     "trash4_4(원본_3차분류_네이버 전부)\\test"] # 이미지 증강할 폴더명 => Yolov5 형태 기준 코드 작성

###########################################

for cur in range(len(INPUT_FOLDER_NAME)):
    # input
    images_path = INPUT_FOLDER_NAME[cur]+'\\images'
    labels_path = INPUT_FOLDER_NAME[cur]+'\\labels'
    datas_img = []
    datas_lab = []
    for root, dirs, files in os.walk(images_path):
        for file in files:
            datas_img = files
    for root, dirs, files in os.walk(labels_path):
        for file in files:
            datas_lab = files

    # run
    for i in range(0, len(datas_img)):
        fileNameImg = datas_img[i]
        inputDirImg = f'.\\{INPUT_FOLDER_NAME[cur]}\\images\\{fileNameImg}'
        fileNameImg = fileNameImg.replace(".jpg", "")
        inputDirLab = f'.\\{INPUT_FOLDER_NAME[cur]}\\labels\\{fileNameImg}.txt'
        with Image.open(inputDirImg) as img:    
            width, height = img.size
            bbox = [0, 0, 0, 0]
            items = []
            darrs = []
            with open(inputDirLab, 'r') as lab:
                while(lab.readable()):
                    item = lab.readline()
                    if(item == ''): break
                    item = item.replace('\n','')
                    item = item.split(' ') # 배열 -> 0:class, 1~4:bbox
                    items.append(item)
            for k in range(len(items)):
                item = items[k]
                bbox = [float(item[1]), float(item[2]), float(item[3]), float(item[4])] # input 라벨링
                bbox = [bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height] # 상대좌표 -> 절대좌표(중심점,너비,높이) 좌표로 변환
                # 1. cutout -> 4방향
                darr = [[float(bbox[0]+bbox[2]/2),float(bbox[1]+bbox[3]/2)],
                        [float(bbox[0]-bbox[2]/2),float(bbox[1]+bbox[3]/2)],
                        [float(bbox[0]+bbox[2]/2),float(bbox[1]-bbox[3]/2)],
                        [float(bbox[0]-bbox[2]/2),float(bbox[1]-bbox[3]/2)]]
                darrs.append(darr)
            for j in range(0,4):
                img_cutout = img.copy()
                draw = ImageDraw.Draw(img_cutout)
                for k in range(0, len(items)):
                    item = items[k]
                    bbox = [float(item[1]), float(item[2]), float(item[3]), float(item[4])] # input 라벨링
                    bbox = [bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height] # 상대좌표 -> 절대좌표(중심점,너비,높이) 좌표로 변환
                    P1 = (float(bbox[0]),float(bbox[1])) # 고정
                    P2 = (darrs[k][j][0],darrs[k][j][1])
                    draw.rectangle((P1,P2), fill=(0, 0, 0, 0)) # fill=(0, 0, 0, 0) 은 검정 투명
                # img 증강
                img_cutout.save(f'.\\{INPUT_FOLDER_NAME[cur]}\\images\\{fileNameImg}_{j}.jpg')
                # lab 복제
                original_file_path = f'.\\{INPUT_FOLDER_NAME[cur]}\\labels\\{fileNameImg}.txt'
                clone_file_path = f'.\\{INPUT_FOLDER_NAME[cur]}\\labels\\{fileNameImg}_{j}.txt'
                shutil.copyfile(original_file_path, clone_file_path)
            # 2. 좌우대칭
            img_flipped_LR = img.transpose(Image.FLIP_LEFT_RIGHT)
            for k in range(0, len(items)):
                item = items[k]
                bbox = [float(item[1]), float(item[2]), float(item[3]), float(item[4])] # input 라벨링
                bbox[0] = width-(bbox[0]*width) # 상대좌표 -> 절대좌표 변환 및 좌우대칭 좌표로 변환
                bbox[0] = bbox[0]/width # 절대좌표 -> 상대좌표 변환
                # lab 저장
                with open(f'.\\{INPUT_FOLDER_NAME[cur]}\\labels\\{fileNameImg}_LR.txt', "a") as f2: 
                    f2.write(f"{item[0]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                # 혹시나 bbox좌표도 그리고싶다면?? 아래 코드 사용(위에까진 yolov5 형식의 라벨링 좌표 상태)
                # bbox = [bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height] # 상대좌표 -> 절대좌표(중심점,너비,높이) 좌표로 변환
                # bbox = [bbox[0]-(bbox[2]/2),bbox[1]-(bbox[3]/2),bbox[0]+(bbox[2]/2),bbox[1]+(bbox[3]/2)] # 절대좌표(중심점,너비,높이) -> bbox leftTop, rightBottom 좌표로 변환
                # draw = ImageDraw.Draw(img_flipped_LR)
                # leftTop = (float(bbox[0]),float(bbox[1]))
                # rightBottom = (float(bbox[2]),float(bbox[3]))
                # draw.rectangle((leftTop,rightBottom), outline='green', width=3)
            # img 증강
            img_flipped_LR.save(f'.\\{INPUT_FOLDER_NAME[cur]}\\images\\{fileNameImg}_LR.jpg')

