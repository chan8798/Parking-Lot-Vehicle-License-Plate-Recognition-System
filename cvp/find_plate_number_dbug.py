import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# OCR
# -------------------------
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="korean",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)

# -------------------------
# 이미지 로드
# -------------------------
img = cv2.imread("result.jpg")
img_orign = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vis = img_orign.copy()
plt.imshow(vis)
plt.title("original image")
plt.axis("off")
plt.show()

# -------------------------
# 1차 OCR
# -------------------------
result = ocr.predict(input=img)
boxes = result[0]['dt_polys']
text = result[0]['rec_texts']
print("text",text)

if boxes is None or len(boxes) == 0:
    print("None")

if text is None or len(text) == 0:
    print("None")


# 시각화
vis = img_orign.copy()
for box in boxes:
    pts = np.array(box, dtype=np.int32)
    cv2.polylines(vis, [pts], True, (255,0,0), 1)

plt.imshow(vis)
plt.title("1st OCR Boxes")
plt.axis("off")
plt.show()

# -------------------------
# 점 정렬
# -------------------------
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# -------------------------
# 직선 계산
# -------------------------
def line_slope_intercept(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        return None, x1

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

top_lines, bottom_lines = [], []

print("boxes",boxes)

for box in boxes:
    pts = order_points(box)
    tl, tr, br, bl = pts
    
    t=min(tl[1],tr[1])
    b=max(bl[1],br[1])
    
    m_top, b_top = line_slope_intercept(tl, tr)
    m_bottom, b_bottom = line_slope_intercept(bl, br)

    top_lines.append((m_top, b_top, t))
    bottom_lines.append((m_bottom, b_bottom, b))
    
print("top_line",top_lines)
print("bottom_line",bottom_lines)

top_line = min(top_lines, key=lambda x: x[2])
bottom_line = max(bottom_lines, key=lambda x: x[2])

m_top, b_top,_ = top_line
m_bottom, b_bottom,_= bottom_line

b_top-=10
b_bottom+=10
# -------------------------
# 직선 시각화
# -------------------------
h, w = img.shape[:2]
vis_lines = img_orign.copy()

# Top line
if m_top is not None:
    x1, x2 = 0, w
    y1 = int(m_top * x1 + b_top)
    y2 = int(m_top * x2 + b_top)
    cv2.line(vis_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)

# Bottom line
if m_bottom is not None:
    x1, x2 = 0, w
    y1 = int(m_bottom * x1 + b_bottom)
    y2 = int(m_bottom * x2 + b_bottom)
    cv2.line(vis_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)

# 출력
plt.imshow(vis_lines)
plt.title("Top / Bottom Lines")
plt.axis("off")
plt.show()

def mb_to_abc(m, b):
    return -m, 1, b

top_line_abc = mb_to_abc(m_top, b_top)
bottom_line_abc = mb_to_abc(m_bottom, b_bottom)

_, w = img.shape[:2]
x_min=0
x_max=w

car_plate_number=None
number_candidate=[]
#번호판 글 잘림 방지를 위해 안전하게 0~13까지 반복
for i in range(0,14):
    
    left_line = (1, 0, x_min+i)
    right_line = (1, 0, x_max-i)
    
    # Left vertical line
    cv2.line(vis_lines,
             (int(x_min), 0),
             (int(x_min), h),
             (0,255,0),
             2)
    
    # Right vertical line
    cv2.line(vis_lines,
             (int(x_max), 0),
             (int(x_max), h),
             (255,255,0),
             2)
    
    # 다시 출력
    plt.figure(figsize=(12,6))
    plt.imshow(vis_lines)
    plt.title("Top / Bottom / Left / Right Lines")
    plt.axis("off")
    plt.show()
    
    def intersect(line1, line2):
        A = np.array([[line1[0], line1[1]],
                      [line2[0], line2[1]]], dtype=np.float32)
        C = np.array([line1[2], line2[2]], dtype=np.float32)
        return np.linalg.solve(A, C)
    
    tl = intersect(top_line_abc, left_line)
    tr = intersect(top_line_abc, right_line)
    bl = intersect(bottom_line_abc, left_line)
    br = intersect(bottom_line_abc, right_line)
    
    new_box = np.array([tl, tr, br, bl], dtype=np.float32)
    
    vis_intersection = img_orign.copy()
    
    # 교점 좌표
    points = {
        "TL": tl,
        "TR": tr,
        "BR": br,
        "BL": bl
    }
    
    # 점 표시
    for name, p in points.items():
    
        x, y = int(p[0]), int(p[1])
    
        cv2.circle(vis_intersection,
                   (x, y),
                   1,
                   (255,0,0),
                   -1)
    
    # 최종 사각형
    pts = np.int32(new_box)
    
    cv2.polylines(vis_intersection,
                  [pts],
                  True,
                  (255,255,0),
                  3)
    
    plt.figure(figsize=(12,6))
    plt.imshow(vis_intersection)
    plt.title("Intersection Points & Final Box")
    plt.axis("off")
    plt.show()
    
    # -------------------------
    # warp
    # -------------------------
    def warp(image, pts):
        (tl, tr, br, bl) = pts
    
        w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        h = int(min(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    
        dst = np.array([
            [0, 0],
            [w , 0],
            [w , h],
            [0, h ]
        ], dtype=np.float32)
    
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(image, M, (w, h))
    
    warped = warp(img, new_box)
    warped_vis= cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    warped_height,warped_width =warped.shape[:2]
    print("warped_height", warped_height)
    print("warped_width", warped_width)
    
    plt.imshow(warped_vis)
    plt.title("Warped")
    plt.axis("off")
    plt.show()

    result2 = ocr.predict(input=warped)
    boxes2 = result2[0]['dt_polys']
    texts = result2[0]['rec_texts']
    
    print("texts",texts)
    
    if boxes2 is None or len(boxes2) == 0:
        print("None")

    if texts is None or len(texts) == 0:
        print("None")
        
    # 시각화
    vis2 = warped_vis.copy()
    
    for box in boxes2:
    
        pts = np.array(box, dtype=np.float32)
    
        # 최소 직사각형 계산
        rect = cv2.minAreaRect(pts)
    
        # 직사각형 4점 얻기
        rect_pts = cv2.boxPoints(rect)
    
        rect_pts = np.int32(rect_pts)
    
        # 그리기
        cv2.polylines(vis2,
                      [rect_pts],
                      True,
                      (255,0,0),
                      2)
    
    plt.figure(figsize=(12,5))
    plt.imshow(vis2)
    plt.title("2nd OCR MinAreaRect")
    plt.axis("off")
    plt.show()
    
    # -------------------------
    # 중심 계산
    # -------------------------
    centers = [None] * len(boxes2)
    for i, box in enumerate(boxes2):
        
        pts = order_points(box)  # 점 순서 정렬 (좌상, 우상, 우하, 좌하)
        
        # 최소 직사각형
        rect = cv2.minAreaRect(pts)
        box_rect = cv2.boxPoints(rect)
        box_rect = np.int32(box_rect)
        
        # 다시 정렬 (선택)
        rect_pts = order_points(box_rect)
        (tl, tr, br, bl) = rect_pts
        
        # 세로 길이 (좌/우 중 큰 값)
        width = np.linalg.norm(tl - tr)
        print("widthA",width)
        
        # 세로 길이 (좌/우 중 큰 값)
        height = np.linalg.norm(tr - br)
        print("heightA",height)
        
        
        if height <= (warped_height)*0.15 or width <= (warped_width)*0.15:
            continue
        
        # 중심 계산 (평균)
        cx = np.mean(rect_pts[:, 0])
        cy = np.mean(rect_pts[:, 1])
        centers[i]=(i, cx, cy)
    
    print(centers)
    
    # -------------------------
    # 그룹화 (Union-Find)
    # -------------------------
    n = len(centers)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    v_ref = np.array([1.0, 0])
    
    for i in range(n):
        
        if centers[i] is None:
            continue
        
        _, x1, y1 = centers[i]
    
        for j in range(i + 1, n):
            
            if centers[j] is None:
                continue
            
            _, x2, y2 = centers[j]
    
            dx = x2 - x1
            if dx == 0:
                continue
            
            v_ij = np.array([1, abs((y2 - y1)/dx)])
            cos_sim = cosine_sim(v_ref, v_ij)
            print("cos_sim",cos_sim)
            if abs(cos_sim) >= 0.99:
                print("수평")
                union(i, j)
    
    groups = {}
    for i in range(n):
        
        if centers[i] is None:
            continue
    
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # -------------------------
    # 8. 그룹 내부 정렬 (좌 → 우)
    # -------------------------
    for g in groups.values():
        g.sort(key=lambda i: centers[i][1])
    
    # -------------------------
    # 9. 그룹 간 정렬 (위 → 아래)
    # -------------------------
    group_list = list(groups.values())
    group_list.sort(key=lambda g: min([centers[i][2] for i in g]))
    
    data = list(groups.values())   
    print("data",data)
    
    vis_group = vis2.copy()
    
    colors = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (255,255,0),
    ]
    
    for gi, group in enumerate(group_list):
    
        color = colors[gi % len(colors)]
    
        for idx in group:
    
            _, cx, cy = centers[idx]
    
            cv2.circle(vis_group,
                       (int(cx), int(cy)),
                       6,
                       color,
                       -1)
    
        for i in range(len(group)-1):
    
            _, x1, y1 = centers[group[i]]
            _, x2, y2 = centers[group[i+1]]
    
            cv2.line(vis_group,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     color,
                     3)
    
    plt.figure(figsize=(12,5))
    plt.imshow(vis_group)
    plt.title("Character Grouping")
    plt.axis("off")
    plt.show()
    
    import re
    merged_text=""
    
    for idx,group in enumerate(data):
    
        row_text = ''.join(texts[num] for num in group)
        
        if len(data)==2:
            
            row_text = row_text.replace("|", "1")
            row_text = row_text.replace("ㅇ", "0").replace("o", "0").replace("O", "0")
            row_text  = re.sub(r'[^0-9가-힣]', '',row_text)
            
            if idx==0:
                pattern = r'서울|경기|인천|강원|대전|충북|충남|세종|광주|전북|전남|부산|대구|울산|경북|경남|제주'
                match = re.search(pattern, row_text)
            
                if match:
                    region = match.group()
                    pos = match.start()
                    row_text = row_text[pos + len(region):]
                    
                    if len(row_text) > 2:
                        row_text = row_text[:-1]
                    
                else:
                    
                    match = re.search(r'(\d*)([가-힣])(\d*)', row_text)
    
                    if match:
                        
                        left = match.group(1)
                        korean = match.group(2)
                        right = match.group(3)
                    
                        # 오른쪽 숫자 제거
                        right = ""
                    
                        # 왼쪽이 3자리면 맨 앞 제거(나사 문자 오인식)
                        if len(left) == 3:
                            left = left[1:]
                    
                        row_text = left + korean + right
                        
                    else:
                        row_text=""
                        
            # merged_text에 추가
            merged_text += row_text
            
        elif len(data)==1:
            
            merged_text += row_text
            merged_text = re.sub(r'\s+', '', merged_text)
            merged_text = re.sub(r'[^0-9가-힣]', '', merged_text)
            merged_text=re.sub(r'^[^0-9]+', '', merged_text)
            if len(merged_text)==7 or len(merged_text)==8:
                if merged_text[-5] == "4":
                    merged_text = ( merged_text[:-5]+ "나"+ merged_text[-4:])
        else:
            merged_text=""
            
    number_candidate.append(merged_text)
    
    #2줄 자동차 번호 일 때
    if len(data)==2:
        pattern = r'^\d{2}[가-힣]\d{4}$'
        if re.match(pattern, merged_text):
            car_plate_number = merged_text
            break  # 찾았으므로 다음 루프를 돌지 않고 즉시 종료!
            
    #1줄 자동차 번호 일 때
    elif len(data)==1:
        pattern = r'^\d{2,3}[가-힣]\d{4}$'
        if re.match(pattern, merged_text):
            car_plate_number = merged_text
            break  # 찾았으므로 다음 루프를 돌지 않고 즉시 종료!

print("number_candidate",number_candidate)
print("car_number:",car_plate_number)