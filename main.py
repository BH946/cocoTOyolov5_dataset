import os
import json
import yaml
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFile # 간단한 이미지 resize는 PIL 라이브러리 활용

###########################################

# 설정
INPUT_FOLDER_NAME = ["batch_01", "batch_44"] # 폴더 이름
JSON_NAME = ["data.json"]*len(INPUT_FOLDER_NAME) # json 파일 이름
RESIZE = 640 # 640 크기로 "resize"
USE_CATEGORY = ['General trash', 'Paper', 'Plastic', 'Metal'] # 사용할 카테고리
OUTPUT_FOLDER_NAME = 'trash' # 저장될 폴더명

def findCategory(category): # 사용 카테고리에 맞게 수정
    if(category==1 or category==8): category = 0 # 일반쓰레기
    elif(category==2 or category == 3): category = 1 # 종이
    elif(category==6): category = 2 # 플라스틱
    elif(category==4): category = 3 # 캔
    else: category = -1 # 레이블링 X
    return category

###########################################

# 저장될 폴더명 중복방지 : trash1, trash2...
unique = 1
while os.path.exists(f'.\\{OUTPUT_FOLDER_NAME}'):
    OUTPUT_FOLDER_NAME='trash'+str(unique) 
    unique+=1 

# 파일 총 개수 계산
fileTotalCount = 0 
for i in range(0, len(INPUT_FOLDER_NAME)):
    folder_path = INPUT_FOLDER_NAME[i]
    extensions = [".jpg", ".png", ".bmp", ".jpeg", "gif"] # 사용 이미지 파일 (없으면 꼭 추가)
    # 폴더 하위의 모든 파일을 검색하여 개수 카운트
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                fileTotalCount += 1
# test, valid, train -> 1:2:7 비율로 데이터셋
testCount = fileTotalCount*(1/10)
validCount = fileTotalCount*(3/10)
fileImgCount = 0
fileLabCount = 0

# BBox(경계박스) 찾는 함수
def findBBox(bbox, image):
    width = image['width'] # 이미지 너비, 높이
    height = image['height']
    centerX = bbox[0]+(bbox[2]/2) # 중심점X : 좌측상단X + 객체너비/2
    centerY = bbox[1]+(bbox[3]/2) # 중심점Y : 좌측상단Y + 객체너비/2
    bbox = [centerX/width, centerY/height, bbox[2]/width,bbox[3]/height] # 이미지에 대해 상대적인 값으로 수정
    return bbox

# 파일명 update 함수
def updateCountFun():
    global fileImgCount
    tempPath='test'
    fileImgCount = fileImgCount+1
    if(fileImgCount>validCount):
        tempPath = 'train'
    else:
        if(fileImgCount>testCount):
            tempPath = 'valid'
        else:
            tempPath = 'test'
    return tempPath

# 메인 실행 함수
def reImgLabFun(datas, cur, drawBBox):
    # start
    global fileImgCount
    imgCnt = len(datas['images']) # 변환할 img 개수
    bboxId = 0 # 복잡도 개선 위해
    for i in range(imgCnt):
        image = datas['images'][i]
        inputDir = image['file_name'] # ex:batch_01/0000.jpg
        fileName = inputDir.split('/')[-1] # ex:0000.jpg
        inputDir = f'.\\{INPUT_FOLDER_NAME[cur]}\\{fileName}'
        with Image.open(inputDir) as img:
            tempPath = updateCountFun()
            width = image['width']; height = image['height']
            ratio=1
            if height > RESIZE: # cal ratio
                ratio = RESIZE / float(height)
                width = int((float(width) * float(ratio)))
                height = int(RESIZE)
            else:
                ratio = 1
            # image resize
            try:
                # ImageFile.LOAD_TRUNCATED_IMAGES = True # 잘린 이미지 허용
                reImg = ImageOps.exif_transpose(img)
            except OSError as e:
                print(f"해당 {inputDir}는 잘린 이미지로 판단되므로 pass")
                global fileImgCount
                fileImgCount -= 1
                continue
            reImg = reImg.resize((width, height),resample=Image.LANCZOS)
            # image size update
            image['width'] = width
            image['height'] = height
            # ratio add
            image['ratio'] = ratio
            # bbox update
            reImg2 = reImg.copy()
            for j in range(bboxId, len(datas['annotations'])):
                detail = datas['annotations'][j]
                imageId = detail['image_id']
                if(imageId > image['id']): break
                if(imageId == image['id']):
                    rt = image['ratio']
                    bbox = detail['bbox']
                    bbox = [bbox[0]*rt, bbox[1]*rt, bbox[2]*rt, bbox[3]*rt] # 비율 곱해주기
                    detail['bbox'] = bbox # bbox update
                    bboxId = j # update
                    # start draw bbox
                    category = findCategory(detail['category_id']) # category 찾기
                    if(drawBBox==1 and category!=-1):
                        draw = ImageDraw.Draw(reImg2)
                        leftTop = (float(bbox[0]),float(bbox[1]))
                        rightBottom = (float(bbox[0]+bbox[2]),float(bbox[1]+bbox[3]))
                        draw.text(xy=(leftTop[0],leftTop[1]-30.0),text=USE_CATEGORY[category],font=ImageFont.truetype("arial.ttf",20))
                        draw.rectangle((leftTop,rightBottom), outline='green', width=10)
                    # output path - 디렉토리 없으면 자동 생성
                    outputDir3 = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\labels'
                    os.makedirs(os.path.dirname(outputDir3+'\\'), exist_ok=True)
                    # start labeling
                    with open(outputDir3+'\\'+str(fileImgCount)+'.txt', "a") as f2: 
                        if(category != -1):
                            bbox = findBBox(bbox,image) # bbox 값 찾기
                            f2.write(f"{category} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            # output path - 디렉토리 없으면 자동 생성
            outputDir1 = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\images'
            os.makedirs(os.path.dirname(outputDir1+'\\'), exist_ok=True) 
            if drawBBox == 1:
                # output path - 디렉토리 없으면 자동 생성
                outputDir2 = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\images_bbox'
                os.makedirs(os.path.dirname(outputDir2+'\\'), exist_ok=True) 
                # save img
                reImg2.save(outputDir2+'\\'+str(fileImgCount)+'.jpg')
                # reImg2.show() # img 띄워보기
            # save img
            reImg.save(outputDir1+'\\'+str(fileImgCount)+'.jpg')
            # reImg.show() # img 띄워보기
    return datas


# main
for i in range(0, len(INPUT_FOLDER_NAME)):
    f = open(f'.\\{INPUT_FOLDER_NAME[i]}\\{JSON_NAME[i]}', "r")
    datas = json.load(f)

    # img 640으로 "리사이즈" & labeling
    reImgLabFun(datas, i, 1) # 0:bbox not draw, 1:bbox draw

    f.close()

# data.yaml 생성
config = {
    'train': f'{OUTPUT_FOLDER_NAME}/train/images',
    'test': f'{OUTPUT_FOLDER_NAME}/test/images',
    'val': f'{OUTPUT_FOLDER_NAME}/valid/images',
    'names': USE_CATEGORY,
    'nc': len(USE_CATEGORY)
}
with open(f"./{OUTPUT_FOLDER_NAME}/data.yaml", "w") as f:
    yaml.safe_dump(config, f)