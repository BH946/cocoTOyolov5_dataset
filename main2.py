import os
import yaml
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFile # 간단한 이미지 resize는 PIL 라이브러리 활용

###########################################
# no json
# 설정
INPUT_FOLDER_NAME = ["metal_416x416", "plastic_"] # 폴더 이름
USED_CATEGORY = [[[-1,0,-1]], # ex:0 카테고리를 전부 Plastic(1)로 변경
                 [[-1,-1,0],[-1,-1,3]]] # ex:0,3 카테고리를 전부 Metal(2)로 변경
USE_CATEGORY = ['Paper', 'Plastic', 'Metal'] # 사용할 카테고리
RESIZE = 340 # 340 크기로 "resize"
OUTPUT_FOLDER_NAME = 'trash' # 저장될 폴더명

###########################################

# 다운한 데이터에 있는 카테고리와 우리가 사용하는 카테고리와 매칭 함수
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
    folder_path = INPUT_FOLDER_NAME[cur]+'\\images'
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


# img resize - no json & label change class
def reImgFun(cur, drawBBox):
    # file open
    folder_path = INPUT_FOLDER_NAME[cur]+'\\images'
    datas_img = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            datas_img = files
    folder_path = INPUT_FOLDER_NAME[cur]+'\\labels'
    datas_lab = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            datas_lab = files
    # start
    global fileImgCount, fileNameCount
    imgCnt = len(datas_img) # 변환할 img 개수
    for i in range(0, imgCnt):
        fileName = datas_img[i]
        inputDir = f'.\\{INPUT_FOLDER_NAME[cur]}\\images\\{fileName}'
        with Image.open(inputDir) as img:
            tempPath = updateCountFun()
            width, height = img.size
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
            # draw bbox & change class
            reImg2 = reImg.copy() 
            fileName = datas_lab[i]
            inputDir = f'.\\{INPUT_FOLDER_NAME[cur]}\\labels\\{fileName}'
            with open(inputDir, 'r') as f3:
                while(f3.readable()):
                    item = f3.readline()
                    if(item == ''): break
                    item = item.replace('\n','')
                    item = item.split(' ') # 배열 -> 0:class, 1~4:bbox
                    category = findCategory(int(item[0]),cur) # category 찾기
                    if(category==-1): 
                        print("category가 -1로 데이터가 잘못 구성되어있습니다. 해당 데이터를 수정 or 삭제 하고 다시 돌리세요")
                        print(fileName, " 설정된 클래스 : ", item[0])
                        break
                        # quit() # 종료
                    # start draw bbox
                    if(drawBBox==1):
                        draw = ImageDraw.Draw(reImg2)
                        leftTop = (float(item[1])*width,float(item[2])*height)
                        rightBottom = ((float(item[1])+float(item[3]))*width,(float(item[2])+float(item[4]))*height)
                        draw.text(xy=(leftTop[0],leftTop[1]-30.0),text=USE_CATEGORY[category],font=ImageFont.truetype("arial.ttf",20))
                        draw.rectangle((leftTop,rightBottom), outline='green', width=3)
                    # output path - 디렉토리 없으면 자동 생성
                    outputDir3 = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\labels'
                    os.makedirs(os.path.dirname(outputDir3+'\\'), exist_ok=True)
                    # start labeling - change class
                    with open(outputDir3+'\\'+str(fileNameCount)+'.txt', "a") as f2: 
                        f2.write(f"{category} {item[1]} {item[2]} {item[3]} {item[4]}\n")
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


# main
for i in range(0, len(INPUT_FOLDER_NAME)):
    # 1:2:7 classification
    fileCountFun(i)
    # img resize - no json & label change class
    reImgFun(i, 1) # 0:bbox not draw, 1:bbox draw

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