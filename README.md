# Intro

**`Yolov5` 를 이용해서 오픈 데이터를 학습시켜야할 일이 생겼는데,**  

**`네이버 커넥트 재단` 에서 제공해주는 오픈 데이터는 `COCO` 형식의 데이터셋이라는 문제가 있다.**  

* **참고로 `RoboFlow` 의 경우 `Yolov5` 데이터셋과 `COCO` 형식 데이터셋 둘다 다운 받을 수 있다.**

<br>

**1. `COCO` -> `Yolov5` 데이터셋의 형태로 변환하는 함수를 소개한다.**

**2. `Yolov5` 데이터셋에 바로 resize, re labeling을 하는 함수도 소개한다.**

* 이 함수의 사용 목적은 **`Yolov5` 데이터셋에 이미지 resize와 labeiling의 클래스 분류 값 수정**
  * 참고 : Yolov5 라벨링 특성상 bbox 좌표값은 상대좌표이므로 이미지 resize와 무관하기 때문에 resize를 통한 라벨링을 추가계산 할 필요는 없다.

<br>

**3. 이미지 증강 기법 중 cutout, 좌우대칭 기법을 사용한 이미지 증강 함수를 소개한다.**

**4. 여러 폴더들에 흩어진 데이터들을 통합해 test,train,valid로 나눠주는 함수를 소개한다.**

<br><br>

# COCO, YOLO 데이터 분석

* **COCO의 bbox 경계값 : “좌측상단 좌표, bbox너비, bbox높이“**

* **YOLO의 bbox 경계값 : “중심점 좌표, bbox너비, bbox높이 단, 전부 상대좌표“**
* **COCO to YOLO 계산**
  * **좌측상단 좌표, bbox 너비와 높이**를 통해서 **중심점 좌표** 를 구할 수 있다.
    * **`중심점X = 좌측상단X + 객체너비/2` 와 `중심점Y = 좌측상단Y + 객체너비/2`**
  * **중심점 좌표, 이미지 크기** 를 통해서 **상대좌표** 를 구할 수 있다.
    * 예시로 이미지 크기 : **450x300**, 객체 중심점 좌표 : **(90,244),** 객체너비**100**, 객체높이**30** 를 가정
    * **`상대좌표 = 0 90/450 244/300 100/450 30/300`**
  * 만약 맨앞에 **클래스**를 붙이면, **실제 라벨링(txt)**을 구할 수 있다.
    * 예시로 **클래스 0** 으로 분류되었다 가정
    * **실제 라벨링(txt) : `0 90/450 244/300 100/450 30/300`**

<br><br>

## 1. COCO

**`네이버 커넥트 재단` 에서 받은 데이터 형태는 아래와 같다.**

```bash
naver-dataset
 ┣ batch_01
 ┃ ┣ 0000.jpg
 ┃ ┣ 0001.jpg
 ┃ ┣ 0002.jpg
 ┃ ┣ ...
 ┃ ┣ 0498.jpg
 ┃ ┣ 0499.jpg
 ┃ ┗ data.json
 ┗ batch_02
  ┃ ┗ ...
 ┗ batch_03
 ┗ batch_04
 ┗ ...
 ┗ batch_44
```

<br>

**각각의 이미지가 담긴 폴더에 `data.json` 파일이 존재하는데, `COCO` 데이터 형태로 이루어져 있다.**

**`data.json` 내부의 코드를 분석해서 `yolov5` 에 필요한 데이터셋의 형태로 바꿔줘야 한다.**

* 이미지 경로(파일명), 이미지 너비x높이, 카테고리 분류 id, bbox(경계값) 좌표 등등 정보들이 존재
* 여기서 bbox 좌표를 수학적으로(앞에서 언급) 계산후 yolov5 전용 bbox 좌표를 구해냄

<br><br>

## 2. YOLO

**Yolov5 전용 데이터 형식으로 변경한 모습이다.**

**참고로 `images_bbox` 는 경계박스 그린 이미지 따로 보려고 생성한것이라서 실제 데이터셋에서는 전혀 필요없으므로 없다고 봐도 무방하다.**

```bash
trash
 ┣ test
 ┃ ┣ images
 ┃ ┃ ┣ 1.jpg
 ┃ ┃ ┣ ...
 ┃ ┃ ┣ 81.jpg
 ┃ ┣ images_bbox
 ┃ ┃ ┣ 1.jpg
 ┃ ┃ ┣ ...
 ┃ ┃ ┣ 81.jpg
 ┃ ┗ labels
 ┃ ┃ ┣ 1.txt
 ┃ ┃ ┣ ...
 ┃ ┃ ┣ 81.txt
 ┣ train
 ┃ ┣ images
 ┃ ┣ images_bbox
 ┃ ┗ labels
 ┣ valid
 ┃ ┣ images
 ┃ ┣ images_bbox
 ┃ ┗ labels
 ┗ data.yaml
```

<br><br>

# 1. COCO to YOLO - main.py

**`CoCo to Yolo 데이터셋` 뿐만 아니라 `PIL` 라이브러리를 활용해서 `Img resize` 까지 함께 동작**

**단, `data.json` 구조나 폴더명, 파일명들은 사람마다 가진 데이터 마다 다를수도 있기 때문에 필요에 따라 적절히 코드 수정을 권장**

* **예시**
  * `네이버` 의 경우 batch폴더 하위에 img들 및 json
  * `RoboFlow` 의 경우 `Yolo`가 아닌 `COCO`형태로 다운받을때 test, train, valid폴더 하위 img들 및 json

<br><br>

## 설정

* **반드시 `main.py` 를 `INPUT_FOLDER_NAME ` 와 같은 계층에 두고 실행**
* **`INPUT_FOLDER_NAME = ["batch_01", "batch_44"]` : 폴더 이름 설정**
  * 해당 폴더 하위에 꼭 `img`가 존재하는 곳으로 설정
  * 왜냐하면 `Roboflow`처럼 하위에 `test,train,valid` 폴더로 또 구성되는 경우도 있기 때문
    * 이때, 올바른 설정 예시 : **`INPUT_FOLDER_NAME=["metal.v1.coco\\test", "metal.v1.coco\\train", "metal.v1.coco\\valid"]`**

* **`JSON_NAME = ["data.json"]*len(INPUT_FOLDER_NAME)` : json 파일 이름 설정**
  * `json` 이름들이 다르다면 `JSON_NAME = ["data_01.json", "data_02.json",]` 형태로 설정
* **`USED_CATEGORY` : COCO 데이터에서 사용중인 카테고리를 `USE_CATEGORY` 와 맞게끔 index로 매핑**
  * **설정 - naver COCO 의 예시**
    * `INPUT_FOLDER_NAME = ["batch_01", "batch_44"]` 라 가정
    * `USE_CATEGORY = ['Paper', 'Plastic', 'Metal']` 라 가정 했을때,
    * `USED_CATEGORY = [[[2,-1,-1],[3,-1,-1],[-1,6,-1],[-1,-1,4]],   
      [[2,-1,-1],[3,-1,-1],[-1,6,-1],[-1,-1,4]]]` 로 설정
      * `[?,?,?]` 의 각 자리의 의미는 `USE_CATEGORY` 의 paper(0), plastic(1), metal(2) 의미
      * `[[2,-1,-1],[3,-1,-1],[-1,6,-1],[-1,-1,4]]` : bacth_01 폴더의 데이터들 카테고리가 2,3 -> 0(paper), 6 -> 1(plastic), 4 -> 2(metal) 로 매핑을 의미
      * `[[2,-1,-1],[3,-1,-1],[-1,6,-1],[-1,-1,4]]` : batch_44 폴더의 데이터들 카테고리 매핑이며 위와 매핑 값은 동일

* **`USE_CATEGORY = ['Paper', 'Plastic', 'Metal']`  : 우리가 사용할 카테고리 설정**
  * 예로 원본 데이터의 카테고리가 11개 였는데, 필요에 따라 3개로 줄임
* **`RESIZE = 340` : 340 크기로 "resize"(원하는 크기로 설정)**
* **`OUTPUT_FOLDER_NAME = 'trash'` : 저장될 폴더명 설정**
  * 동일한 폴더명 존재시 `trash1, trash2, ...` 로 자동으로 폴더명 변경함

<br>

**참고**

* **카테고리 설정을 잘했으면, `findCategory()` 함수로 자동으로 카테고리 매핑 값 반환**

* **`test, valid, train -> 1:2:7 비율` 로 데이터셋을 진행**
  * `fileCountFun(), updateCountFun()` 함수를 사용

* **`notUseImgFun()` 란 카테고리를 'Paper' 만 사용할 경우 해당 카테고리가 없는 이미지는 제외하기 위해서 만든 함수**
  * 해당 함수를 실행하면, 자신이 사용하는 카테고리가 없는 이미지를 구해줌
  * `notUseImgs.txt` 를 생성해서 기입하므로, 이곳에 직접 사용안할 이미지를 적어도 됨
    * 단, ''폴더명/파일명'' 형태로 기입

* **`reImgLabFun(datas, i, 1) # 0:bbox not draw, 1:bbox draw` 란 함수 인자로 0을 주면 경계박스 없고, 1을 주면 경계박스 없는 사진 + 경계박스 그린 사진까지 폴더 구분해서 만듬**
* **`data.yaml` 을 자동으로 생성**

<br><br>

## 동작

1. `notUseImgFun()` 함수는 필요에 따라 사용
1. `INPUT_FOLDER_NAME` 에 폴더명 넣은 개수만큼 전체 반복(주석 상태)
   1. `data.json` 파일 로드
   1. `fileCountFun()` 함수 사용 - 1:2:7 로 이미지 분류
   2. `reImgLabFun() 함수 실행`
      1. `notUseImgs.txt` 에 기록된 데이터 먼저 dict형태로 load
      2. `data.json` 에 있는 전체 이미지를 가져와 해당 개수만큼 반복
         1. `notUseImgs` 인 이미지는 pass
         2. 각각 이미지 `Image.open(이미지경로)` 로 오픈
         3. 이미지를 설정한 `높이` 로 resize 진행
         4. `bbox` 정보와 `이미지` 정보가 **매칭**되는 경우들 전부 접근
            1. `bbox` 를 `yolo` 형태로 계산 및 설정한 `카테고리` 를 구함
            2. **`카테고리` 유무에 따라 `라벨링(txt)` 파일에 요소 추가**
            3. `drawBBox` 유무에 따라 `이미지` 에 경계박스를 그림
         5. **이미지 추가** 및 `drawBBox` 유무에 따라 **경계박스 추가한 이미지도 추가**
2. `data.yaml` 를 생성

<br><br>

# 2. YOLO to YOLO - main2.py

이번 함수는 `Yolov5` 형태의 데이터셋에 **이미지 resize와 labeiling의 클래스 분류 값 커스텀**을 하는 함수이다.

* 라벨링(bbox 좌표)을 직접 계산 안하는 점을 제외하고는 `main.py`와 비슷하다.
* 또한 `notUseImgFun()` 는 굳이 넣지 않았다.
* 만들 생각이 없었는데, json없이 YOLO 데이터 때 사용하려고 만든거라서 YOLO형태가 아닌건 직접 커스텀이 필요하다.

<br>

**따라서 해당 파일은 그냥 참고용으로만 사용!**

<br><br>

# 3. img_aumentation.py

**제목 그대로 이미지 증강기법을 의미하며, `INPUT_FOLDER_NAME` 에 적어둔 경로의 하위 모든 이미지에 적용을 한다.**

**이때, `Yolo` 형식을 기준으로 코드 작성을 했으며 해당 경로에 바로 이미지를 추가 저장하였다. (라벨링도 포함)**

* 기법은 **좌우대칭**과 **cutout** 사용

<br><br>

# 4. img_classifier.py

**이미지 증강을 "여러 폴더"들에 하고나니 "이미지 통합" 및 "다시 test, train, valid 재분류" 작업이 필요했다.**

**이를 진행하는 함수이다.**

<br><br>

# Outro

**COCO 데이터를 YOLO 데이터 형태로 변환 및 이미지 리사이즈나 클래스명 커스텀 지원이 주 목적이었으며, 부가적인 함수들은 필요에 따라 사용**

**실제 이 코드를 프로젝트에 사용한 모습을 보여주기 위해서  
이전에 진행했던 Yolo - Trash Detect 프로젝트 링크 제공**

* **[trash detect](https://github.com/BH946/trash-detection)**

