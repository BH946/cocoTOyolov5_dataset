import os
import json
import yaml
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFile # 간단한 이미지 resize는 PIL 라이브러리 활용

###########################################

# 설정 - naver COCO
# INPUT_FOLDER_NAME = ["batch_01", "batch_44"] # 폴더 이름
# JSON_NAME = ["data.json"]*len(INPUT_FOLDER_NAME) # json 파일 이름
# USED_CATEGORY = [[[2,-1,-1],[3,-1,-1],[-1,6,-1],[-1,-1,4]], # ex:2,3 카테고리 Paper(0)로 변경, 6->1, 4->2
#                  [[2,-1,-1],[3,-1,-1],[-1,6,-1],[-1,-1,4]]]

# 설정 - roboflow COCO
INPUT_FOLDER_NAME = ["metal.v1.coco\\test", "metal.v1.coco\\train", 
                     "metal.v1i.coco\\test", "metal.v1i.coco\\train", "metal.v1i.coco\\valid",
                     "metal.v1ii.coco\\test", "metal.v1ii.coco\\train","metal.v1ii.coco\\valid",
                     "metal.v1iii.coco\\train",
                     "plastic.v1i.coco\\test", "plastic.v1i.coco\\train", "plastic.v1i.coco\\valid",
                     "plastic.v3i.coco\\test","plastic.v3i.coco\\train","plastic.v3i.coco\\valid"] # 폴더 이름
JSON_NAME = ["_annotations.coco.json"]*len(INPUT_FOLDER_NAME) # json 파일 이름
USED_CATEGORY = [[[-1,-1,4]],[[-1,-1,4]],[[-1,-1,1]],[[-1,-1,1]],[[-1,-1,1]],[[-1,-1,1]],[[-1,-1,1]],[[-1,-1,1]],[[-1,-1,1]], # 폴더명 순, 설정된 Metal카테고리 전부 Metal(2)로 변경
                [[-1,1,-1]],[[-1,1,-1]],[[-1,1,-1]],[[-1,3,-1]],[[-1,3,-1]],[[-1,3,-1]]] # 폴더명 순, 설정된 Plastic카테고리 Plastic(1)로 변경
USE_CATEGORY = ['Paper', 'Plastic', 'Metal'] # 사용할 카테고리
RESIZE = 340 # 340 크기로 "resize"
OUTPUT_FOLDER_NAME = 'trash' # 저장될 폴더명

###########################################

# 다운한 데이터에 있는 카테고리와 우리가 사용하는 카테고리와 매핑 함수
def findCategory(category,cur): # USED_CATEGORY를 꼭 미리 잘 설정
    for l in range(len(USED_CATEGORY[cur])):
        for m in range(len(USE_CATEGORY)):
            if(category==USED_CATEGORY[cur][l][m]): return m
    return -1

# 저장될 폴더명 중복방지 : trash1, trash2...
unique = 1
while os.path.exists(f'.\\{OUTPUT_FOLDER_NAME}'):
    OUTPUT_FOLDER_NAME='trash'+str(unique) 
    unique+=1 

# 파일 총 개수 계산
testCount, validCount, fileTotalCount, fileImgCount = 0,0,0,0 # init
def fileCountFun(cur):
    global testCount, validCount, fileTotalCount, fileImgCount
    testCount, validCount, fileTotalCount, fileImgCount = 0,0,0,0 # init
    folder_path = INPUT_FOLDER_NAME[cur]
    extensions = [".jpg", ".png", ".bmp", ".jpeg", "gif"] # 사용 이미지 파일 (없으면 꼭 추가)
    # 폴더 하위의 모든 파일을 검색하여 개수 카운트
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                fileTotalCount += 1
    # test, valid, train -> 1:2:7 비율로 데이터셋
    testCount = fileTotalCount*(1/10)
    validCount = fileTotalCount*(3/10)

# 파일명 update 함수
fileNameCount = 0
def updateCountFun():
    global fileImgCount, fileNameCount
    tempPath='test'
    fileImgCount = fileImgCount+1
    fileNameCount = fileNameCount+1
    if(fileImgCount>validCount):
        tempPath = 'train'
    else:
        if(fileImgCount>testCount):
            tempPath = 'valid'
        else:
            tempPath = 'test'
    return tempPath

# BBox(경계박스) 찾는 함수
def findBBox(bbox, image):
    width = image['width'] # 이미지 너비, 높이
    height = image['height']
    centerX = bbox[0]+(bbox[2]/2) # 중심점X : 좌측상단X + 객체너비/2
    centerY = bbox[1]+(bbox[3]/2) # 중심점Y : 좌측상단Y + 객체너비/2
    bbox = [centerX/width, centerY/height, bbox[2]/width,bbox[3]/height] # 이미지에 대해 상대적인 값으로 수정
    return bbox


# 메인 실행 함수
def reImgLabFun(datas, cur, drawBBox):
    # file open => 사용안할 이미지 체크 위해
    notUseImgs = {} # key : fileName, val : inputDir
    with open('.\\notUseImgs.txt', "r") as f:
        while(f.readable()):
            item = f.readline().replace('\n','')
            if(item == ''): break
            index = item.split('/')[-1]
            notUseImgs[index] = [] # init
    with open('.\\notUseImgs.txt', "r") as f:
        while(f.readable()):
            item = f.readline().replace('\n','')
            if(item == ''): break
            index = item.split('/')[-1]
            notUseImgs[index].append(item) # append
    # start
    global fileImgCount, fileNameCount
    imgCnt = len(datas['images']) # 변환할 img 개수
    bboxId = 0 # 복잡도 개선 위해
    for i in range(0, imgCnt):
        needImg = True
        image = datas['images'][i]
        inputDir = image['file_name'] # ex:batch_01/0000.jpg or 0000.jpg
        fileName = inputDir.split('/')[-1] # ex:0000.jpg
        # 이미지 사용여부 결정
        try:
            notUseImgs[fileName]
        except:
            pass
        else:
            for val in notUseImgs[fileName]:
                if(f'{INPUT_FOLDER_NAME[cur]}/{fileName}' == val): 
                    needImg = False
        if(needImg is False): continue # 이미지 pass
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
                global fileImgCount, fileNameCount
                fileImgCount -= 1
                fileNameCount -= 1
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
                    category = findCategory(detail['category_id'],cur) # category 찾기
                    if(category==-1): 
                        print("category가 -1로 데이터가 잘못 구성되어있습니다. 해당 데이터를 수정 or notUseImgs.txt에 추가하고 다시 돌리세요")
                        print(fileName, " 설정된 클래스 : ", detail['category_id'])
                        break
                        # quit() # 종료
                    if(drawBBox==1):
                        draw = ImageDraw.Draw(reImg2)
                        leftTop = (float(bbox[0]),float(bbox[1]))
                        rightBottom = (float(bbox[0]+bbox[2]),float(bbox[1]+bbox[3]))
                        draw.text(xy=(leftTop[0],leftTop[1]-30.0),text=USE_CATEGORY[category],font=ImageFont.truetype("arial.ttf",20))
                        draw.rectangle((leftTop,rightBottom), outline='green', width=3)
                    # output path - 디렉토리 없으면 자동 생성
                    outputDir3 = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\labels'
                    os.makedirs(os.path.dirname(outputDir3+'\\'), exist_ok=True)
                    # start labeling
                    with open(outputDir3+'\\'+str(fileNameCount)+'.txt', "a") as f2: 
                        bbox = findBBox(bbox,image) # bbox 값 찾기
                        f2.write(f"{category} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            if(category==-1): continue
            # output path - 디렉토리 없으면 자동 생성
            outputDir1 = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\images'
            os.makedirs(os.path.dirname(outputDir1+'\\'), exist_ok=True) 
            if drawBBox == 1:
                # output path - 디렉토리 없으면 자동 생성
                outputDir2 = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\images_bbox'
                os.makedirs(os.path.dirname(outputDir2+'\\'), exist_ok=True) 
                # save img
                reImg2.save(outputDir2+'\\'+str(fileNameCount)+'.jpg')
                # reImg2.show() # img 띄워보기
            # save img
            reImg.save(outputDir1+'\\'+str(fileNameCount)+'.jpg')
            # reImg.show() # img 띄워보기


# (필요하다면) 선행 실행 함수
def notUseImgFun():
    notUseCategoryCount = 0
    for i in range(0, len(INPUT_FOLDER_NAME)):
        with open(f'.\\{INPUT_FOLDER_NAME[i]}\\{JSON_NAME[i]}', "r") as f:
            datas = json.load(f)
        f2 = open('.\\notUseImgs.txt','a')
        imgCnt = len(datas['images']) # 변환할 img 개수
        bboxId = 0
        for i in range(imgCnt):
            image = datas['images'][i]
            inputDir = image['file_name'] # ex:batch_01/0000.jpg
            fileName = inputDir.split('/')[-1]
            useCategory = False
            for j in range(bboxId, len(datas['annotations'])):
                detail = datas['annotations'][j]
                imageId = detail['image_id']
                if(imageId > image['id']): break
                if(imageId == image['id']):
                    bboxId = j # update
                    category = detail['category_id']
                    # 자신이 사용할 카테고리인지 확인
                    for c in range(len(USED_CATEGORY)):
                        for v in range(len(USED_CATEGORY[c])):
                            for b in range(len(USE_CATEGORY)):
                                if(category==USED_CATEGORY[c][v][b] and USED_CATEGORY[c][v][b]!=-1): 
                                  useCategory = True            
            if(useCategory is False):
                notUseCategoryCount+=1
                inputDir = f'{INPUT_FOLDER_NAME[i]}/{fileName}'
                f2.write(inputDir+'\n')
                print(inputDir) # 그냥 사용 카테고리 확인용
        f2.close()
    # 이에 따라 파일 총 개수 다시 연산필요
    global fileTotalCount
    global testCount
    global validCount
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
    fileTotalCount-=notUseCategoryCount
    testCount = fileTotalCount*(1/10)
    validCount = fileTotalCount*(3/10)


# main
# save not use img => 필요에 따라 사용
with open('.\\notUseImgs.txt', 'w'): pass # 기존 txt 초기화(생성)
# notUseImgFun() # 직접 txt에 사용안할 파일 적어도 됨 => 작성 형태 : '폴더명/파일명'

for i in range(0, len(INPUT_FOLDER_NAME)):
    # open COCO json
    f = open(f'.\\{INPUT_FOLDER_NAME[i]}\\{JSON_NAME[i]}', "r")
    datas = json.load(f)
    # 1:2:7 classification
    fileCountFun(i)
    # img 340으로 "리사이즈" & labeling
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