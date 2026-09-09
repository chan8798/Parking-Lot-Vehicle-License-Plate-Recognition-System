import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gc

def detect_plate(image_path):

    # -----------------------------
    # 이미지 읽기
    # -----------------------------
    img = cv2.imread(image_path)

    if img is None:
        print("이미지 읽기 실패")
        return

    # -----------------------------
    # 원본 이미지 출력
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # -----------------------------
    # 차량 검출 모델
    # -----------------------------
    vehicle_model = YOLO("yolo26x.pt")

    vehicle_results = vehicle_model(img)

    vehicle_boxes = []

    vehicle_draw = cv2.cvtColor(
        img.copy(),
        cv2.COLOR_BGR2RGB
    )

    # COCO 클래스
    vehicle_classes = [2, 5, 7]  # car, bus, truck

    # -----------------------------
    # 차량 탐지
    # -----------------------------
    for box in vehicle_results[0].boxes:

        cls = int(box.cls[0])

        if cls not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        conf = float(box.conf[0])

        area = (x2 - x1) * (y2 - y1)

        vehicle_boxes.append({
            "box": [x1, y1, x2, y2],
            "area": area,
            "conf": conf
        })

        cv2.rectangle(
            vehicle_draw,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

    # -----------------------------
    # 차량 검출 결과 출력
    # -----------------------------
    plt.figure(figsize=(12, 8))
    plt.imshow(vehicle_draw)
    plt.title("Vehicle Detection")
    plt.axis("off")
    plt.show()

    if len(vehicle_boxes) == 0:
        print("차량 검출 실패")
        return

    # -----------------------------
    # 가장 큰 차량 선택
    # -----------------------------
    best_vehicle = max(
        vehicle_boxes,
        key=lambda x: x["area"]
    )

    x1, y1, x2, y2 = best_vehicle["box"]

    print("선택된 차량:")
    print(best_vehicle)

    # -----------------------------
    # 차량 Crop
    # -----------------------------
    vehicle_crop = img[y1:y2, x1:x2]

    plt.figure(figsize=(10, 6))
    plt.imshow(
        cv2.cvtColor(
            vehicle_crop,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("in/out Vehicle Crop")
    plt.axis("off")
    plt.show()

    # -----------------------------
    # 번호판 모델
    # -----------------------------
    plate_model = YOLO("best.pt")

    plate_results = plate_model(vehicle_crop)

    plate_boxes = []

    plate_draw = cv2.cvtColor(
        vehicle_crop.copy(),
        cv2.COLOR_BGR2RGB
    )

    # -----------------------------
    # 번호판 탐지
    # -----------------------------
    for box in plate_results[0].boxes:

        px1, py1, px2, py2 = map(
            int,
            box.xyxy[0]
        )

        conf = float(box.conf[0])

        plate_boxes.append({
            "box": [px1, py1, px2, py2],
            "conf": conf
        })

        cv2.rectangle(
            plate_draw,
            (px1, py1),
            (px2, py2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            plate_draw,
            f"{conf:.2f}",
            (px1, max(py1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    # -----------------------------
    # 번호판 검출 결과 출력
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(plate_draw)
    plt.title("Plate Detection In in/out Vehicle")
    plt.axis("off")
    plt.show()

    if len(plate_boxes) == 0:
        print("번호판 검출 실패")
        return

    # -----------------------------
    # 최고 confidence 번호판 선택
    # -----------------------------
    best_plate = max(
        plate_boxes,
        key=lambda x: x["conf"]
    )

    px1, py1, px2, py2 = best_plate["box"]

    print("선택된 번호판:")
    print(best_plate)

    # -----------------------------
    # 번호판 Crop
    # -----------------------------
    plate_crop = vehicle_crop[
        py1:py2,
        px1:px2
    ]

    # -----------------------------
    # 최종 번호판 출력
    # -----------------------------
    plt.figure(figsize=(8, 3))
    plt.imshow(
        cv2.cvtColor(
            plate_crop,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Final Plate")
    plt.axis("off")
    plt.show()

    # -----------------------------
    # 저장
    # -----------------------------
    cv2.imwrite(
        "result.jpg",
        plate_crop
    )

    print("번호판 저장 완료")

    # -----------------------------
    # 메모리 정리
    # -----------------------------
    del vehicle_model
    del plate_model
    del vehicle_results
    del plate_results

    gc.collect()

# 실행
detect_plate(
    "C:/Users/chan1/cvp/test_fail/e2.jpg"
)