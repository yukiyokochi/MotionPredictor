import glob, cv2, os

# mvs = glob.glob('videodatas/MAH02914.MP4')
mvs = glob.glob('videodatas/*.MP4')


for mv_name in mvs:
    mv = cv2.VideoCapture(mv_name)
    mv_file_name = mv_name[11:]
    frame_rate = int(mv.get(cv2.CAP_PROP_FRAME_COUNT))
    for j in range(frame_rate):
        ch, frame = mv.read()
        if ch:
            frame = cv2.resize(frame,(320,240))
            path = 'devidedvideodatas/{0}/{1}.jpg'.format(mv_file_name,j)
            # path = '../labelImg/data/seculity/{}_case.jpg'.format(j)
            os.makedirs('devidedvideodatas/{0}'.format(mv_file_name),exist_ok=True)
            cv2.imwrite(path, frame)