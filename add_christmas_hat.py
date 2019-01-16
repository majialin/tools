
# coding: utf-8

# In[43]:

import cv2
import dlib
import os


# In[44]:

hat_path = os.path.join(os.path.abspath('.'), 'christmas-hat.png')
hat = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)


# In[45]:

detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_5_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)


# In[46]:

face_path = 'face.jpg'
face = cv2.imread(face_path)
dets = detector(face,1) # 检测出的人脸矩形集合
det = dets[0]
shape = predictor(face, det) # 检测出人脸关键点


# In[47]:

# 绘制标识出人脸和五个关键点
def draw_mark(face, dets):
    if len(dets)>0:
        for d in dets:
            x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()
            cv2.rectangle(face,(x,y),(x+w,y+h),(255,0,0))  # 绘制人脸矩形
            shape = predictor(face, d)      # 输入人脸矩形，输出关键点
            for point in shape.parts():
                cv2.circle(face,(point.x, point.y),2,color=(0,255,0))   # 绘制关键点
            cv2.imshow('face',face)
            cv2.waitKey()


# In[48]:

# 调整帽子大小
factor = 0.8*(det.right()-det.left())/hat.shape[1]
resized_hat_w = round(factor*hat.shape[1])
resized_hat_h = round(factor*hat.shape[0])
resized_hat = cv2.resize(hat, (resized_hat_w, resized_hat_h))
# cv2.imshow('resized_hat', resized_hat)
# cv2.waitKey()


# In[49]:

# 算出要修改的目标框位置
px = shape.part(0).x + shape.part(2).x - shape.part(4).x # 以眼角和鼻尖找出成菱形的点
py = shape.part(0).y + shape.part(2).y - shape.part(4).y # 以此点为目标框的下中点
x1 = int(px - resized_hat_w/2)    # x1,x2,y1,y2为目标框的位置
x2 = x1 + resized_hat_w
y1 = py - resized_hat_h
y2 = py


# In[50]:

# 透明部分处理
alpha_h = resized_hat[:, :, 3] / 255
alpha_f = 1 - alpha_h
# 按3个通道合并
for c in range(0,3):
    face[y1:y2, x1:x2, c] = resized_hat[:, :, c] * alpha_h + face[y1:y2, x1:x2, c] * alpha_f

cv2.imshow('face',face)
cv2.waitKey()
# cv2.imwrite('face_with_hat.png', face)


# In[ ]:



