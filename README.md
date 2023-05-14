# Intro

**`Yolov5` 를 이용해서 오픈 데이터를 학습시켜야할 일이 생겼는데,**

**`네이버 커넥트 재단` 에서 제공해주는 오픈 데이터는 `CoCo` 형식의 데이터셋이라는 문제가 있다.**

**따라서 `CoCo` -> `Yolov5` 데이터셋의 형태로 변환하는 함수를 소개한다.**

<br><br>

# COCO, YOLO 데이터 비교

* **COCO의 bbox 경계값 : “좌측상단 좌표, bbox너비, bbox높이“**

* **YOLO의 bbox 경계값 : “중심점 좌표, bbox너비, bbox높이 단, 전부 상대좌표“**
* **COCO to YOLO 계산**
  * **좌측상단 좌표, bbox 너비와 높이**를 통해서 **중심점 좌표** 를 구할 수 있다.
    * **`중심점X = 좌측상단X + 객체너비/2` 와 `중심점Y = 좌측상단Y + 객체너비/2`**
  * **중심점 좌표, 이미지 크기** 를 통해서 **상대좌표** 를 구할 수 있다.
    * 예시로 이미지 크기 : 450x300, 객체 중심점 좌표 : (90,244), 객체너비100, 객체높이30 를 가정
    * **`상대좌표 = 0 90/450 244/300 100/450 30/300`**
  * 만약 맨앞에 **클래스**를 붙이면, **실제 라벨링(txt)**을 구할 수 있다.
    * 예시로 클래스 0 으로 분류되었다 가정
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

**각각의 이미지가 담긴 폴더에 `data.json` 파일이 존재하는데, `CoCo` 데이터 형태로 이루어져 있다.**

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

# COCO to YOLO

**`CoCo to Yolo 데이터셋` 뿐만 아니라 `PIL` 라이브러리를 활용해서 `Img resize` 까지 함께 동작**

**단, `data.json` 구조가 사람마다 가진 데이터 마다 다를수도 있기 때문에 필요에 따라 적절히 코드 수정을 권장**

<br><br>

## 설정

* **반드시 `main.py` 를 `INPUT_FOLDER_NAME ` 와 같은 계층에 두고 실행**

* **`INPUT_FOLDER_NAME = ["batch_01", "batch_44"]` : 폴더 이름 설정**
* **`JSON_NAME = ["data.json"]*len(INPUT_FOLDER_NAME)` : json 파일 이름 설정**
  * `json` 이름들이 다르다면 `JSON_NAME = ["data_01.json", "data_02.json",]` 형태로 설정
* **`RESIZE = 640` : 640 크기로 "resize" (보통 640으로 많이함)**
* **`USE_CATEGORY = ['General trash', 'Paper', 'Plastic', 'Metal']`  : 사용할 카테고리 설정**
  * 본인은 원본 데이터의 카테고리가 11개 였는데, 필요에 따라 줄였음
* **`def findCategory(category)` : 위에 설정한 카테고리에 알맞게 수정 필요**
* **`OUTPUT_FOLDER_NAME = 'trash'` : 저장될 폴더명 설정**
  * 실제로 동일한 폴더명 존재시 `trash1, trash2, ...` 이런식으로 자동으로 폴더명 수정

<br>

**참고**

* **`test, valid, train -> 1:2:7 비율` 로 데이터셋을 진행**
* **`reImgLabFun(datas, i, 1) # 0:bbox not draw, 1:bbox draw` 란 함수 인자로 0을 주면 경계박스 없고, 1을 주면 경계박스 없는 사진 + 경계박스 그린 사진까지 폴더 구분해서 만듬**
* **`data.yaml` 을 자동으로 생성**

<br><br>

## 동작

1. `INPUT_FOLDER_NAME` 에 폴더명 넣은 개수만큼 전체 반복
   1. `data.json` 파일 로드
   2. `reImgLabFun() 함수 실행`
      1. `data.json` 에 있는 전체 이미지를 가져와 해당 개수만큼 반복
         1. 각각 이미지 `Image.open(이미지경로)` 로 오픈
         2. 이미지를 설정한 `높이` 로 resize 진행
         3. `bbox` 정보와 `이미지` 정보가 **매칭**되는 경우들 전부 접근
            1. `bbox` 를 `yolo` 형태로 계산 및 설정한 `카테고리` 를 구함
            2. **`카테고리` 유무에 따라 `라벨링(txt)` 파일에 요소 추가**
            3. `drawBBox` 유무에 따라 `이미지` 에 경계박스를 그림
         4. **이미지 추가** 및 `drawBBox` 유무에 따라 **경계박스 추가한 이미지도 추가**
2. `data.yaml` 를 생성

<br><br>

# Outro

**COCO 데이터를 YOLO 데이터 형태로 변환 및 이미지 리사이즈 하는 작업의 예시정도로 참고 바람**

**실제 이 코드를 프로젝트에 사용한 모습을 보여주기 위해서  
현재 진행중인 Yolov5 - 쓰레기 분류 프로젝트가 끝나게 된다면, 추가로 깃에 정리하겠다.**