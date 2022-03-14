import cv2

mov_num = 168
mov_type = 6
file_path = 'posedatas/{0}({1}).mov'.format(mov_num, mov_type)
mv = cv2.VideoCapture(file_path)
frame_count = int(mv.get(cv2.CAP_PROP_FRAME_COUNT))
size = (1920, 1080)
alpha = 0.5
frame_rate = int(mv.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
save = cv2.VideoWriter('posedatas/result.mp4', fourcc, frame_rate, size)
mv_data = []
for i in range(frame_count):
    ch, frame = mv.read()
    if ch==True:
        width = frame.shape[1]
        height = frame.shape[0]
        frame = frame[0:height,int(width*0.2):int(width*0.8)]
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hsv[:,:,(2)] = frame_hsv[:,:,(2)]*alpha
        frame_bgr = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
        frame = cv2.resize(frame_bgr,size)
        mv_data.append(frame)

for i in range(len(mv_data)):
    frame=mv_data[i]
    save.write(frame)

save.release()
cv2.destroyAllWindows()