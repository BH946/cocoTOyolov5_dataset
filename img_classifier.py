import os
import shutil

###########################################

# 설정
INPUT_FOLDER_NAME = ["trash4_3(원본_3차분류_roboflow6개)\\test", 
                     "trash4_3(원본_3차분류_roboflow6개)\\train",
                     "trash4_3(원본_3차분류_roboflow6개)\\valid",
                     "trash4_4(원본_3차분류_네이버 전부)\\test",
                     "trash4_4(원본_3차분류_네이버 전부)\\valid"] # 분류할 데이터들 폴더 기입(각각 폴더 하위는 Yolo형태인 /images, /labels 여야 함)
OUTPUT_FOLDER_NAME = 'trash_real' # 저장될 폴더명

###########################################

# 저장될 폴더명 중복방지 : trash_real1, trash_real2...
unique = 1
while os.path.exists(f'.\\{OUTPUT_FOLDER_NAME}'):
    OUTPUT_FOLDER_NAME='trash_real'+str(unique) 
    unique+=1 

# 파일명 update 함수
def updateCountFun():
    global fileCurCount, testCount, validCount
    tempPath='test'
    fileCurCount = fileCurCount+1
    if(fileCurCount>validCount):
        tempPath = 'train'
    else:
        if(fileCurCount>testCount):
            tempPath = 'valid'
        else:
            tempPath = 'test'
    return tempPath

fileTotalCount = 0
fileCurCount = 0

for cur in range(len(INPUT_FOLDER_NAME)):
    # input
    images_path = INPUT_FOLDER_NAME[cur]+'\\images'
    datas_img = []
    for root, dirs, files in os.walk(images_path):
        for file in files:
            datas_img = files
    fileTotalCount += len(datas_img)
testCount = fileTotalCount*(1/10)
validCount = fileTotalCount*(3/10)

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
        tempPath = updateCountFun()
        fileNameImg = datas_img[i]
        fileName = fileNameImg.replace(".jpg", "")
        originalImg_file_path = f'.\\{images_path}\\{datas_img[i]}'
        originalLab_file_path = f'.\\{labels_path}\\{fileName}.txt'
        cloneImg_file_path = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\images\\{datas_img[i]}'
        cloneLab_file_path = f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\labels\\{fileName}.txt'
        os.makedirs(os.path.dirname(f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\labels\\'), exist_ok=True) # 폴더 생성
        os.makedirs(os.path.dirname(f'.\\{OUTPUT_FOLDER_NAME}\\{tempPath}\\images\\'), exist_ok=True) # 폴더 생성
        shutil.copyfile(originalImg_file_path, cloneImg_file_path)
        shutil.copyfile(originalLab_file_path, cloneLab_file_path)