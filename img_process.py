import numpy as np
from numpy.linalg import norm
import cv2

cap=cv2.VideoCapture(0)
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
video=cv2.VideoWriter()
video.open('train.mp4',fourcc,10,sz,True)
save_=False
next_=False
esc_=False
delete_=False
yes_=False
def recall(event,x,y,flags,param):
    global  save_,next_,esc_,delete_,yes_
    if  event==cv2.EVENT_LBUTTONUP:
        if y>5 and y<58:
            if   x>5   and x<95:
                save_=True
            if   x>105 and x<195:
                next_=True
            if   x>205 and x<295:
                esc_=True
            if   x>305 and x<395:
                delete_=True
            if   x>405 and x<495:
                yes_=True

cv2.namedWindow('record')
cv2.setMouseCallback('record', recall)
choice_label=np.zeros((60,500,3)).astype(np.uint8)
cv2.rectangle(choice_label,(5,5),(95,58),(0,255,255),thickness=2)
cv2.rectangle(choice_label,(105,5),(195,58),(0,255,255),thickness=2)
cv2.rectangle(choice_label,(205,5),(295,58 ),(0,255,255),thickness=2)
cv2.rectangle(choice_label,(305,5),(395,58 ),(0,255,255),thickness=2)
cv2.rectangle(choice_label,(405,5),(495,58 ),(0,255,255),thickness=2)
cv2.putText(choice_label,'SAVE',(8,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
cv2.putText(choice_label,'NEXT',(108,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
cv2.putText(choice_label,'ESC',(220,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
cv2.putText(choice_label,'DEL',(320,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
cv2.putText(choice_label,'YES',(420,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)


#当鼠标按下时为True
drawing = False
ix1,iy1,ix2,iy2 = -1,-1,-1,-1

#####创建回调函数
def draw_circle(event,x,y,flags,param):
    global ix1,iy1,ix2,iy2,drawing
    #当按下左键时返回起始位置坐标
    if   event==cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix1,iy1=x,y
        # img2 = img.copy()
        # cv2.rectangle(img2, (ix1, iy1), (x, y), (0, 255, 0), 2)
        # cv2.imshow('img', img2)
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            img2 = img.copy()
            cv2.rectangle(img2,(ix1,iy1),(x,y),(0,255,255),3)
            cv2.imshow('img', img2)
    #当鼠标松开时停止绘图
    elif event ==cv2.EVENT_LBUTTONUP:
        drawing = False
        ix2, iy2 = x, y

# # # ##以下是录制视频的代码
cnt=0
# print('When camera is ready\nclik (SAVE) to record and (ESC) to quit!!')
show_label1=np.zeros((80,500,3)).astype(np.uint8)
cv2.line(show_label1,(0,5),(500,5),(0,255,0),thickness=3)
cv2.putText(show_label1,'When camera ready,clik SAVE',(5,35),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
cv2.putText(show_label1,'to record ; ESC to quit',(5,70),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
# cv2.imshow('show',show_label1)
choice_label1=np.concatenate([choice_label,show_label1],axis=0)
while True:
    _,frame=cap.read()
    cv2.imshow('img',frame)
    if save_:
        video.write(frame)
        print(cnt)
        cnt+=1
    cv2.imshow('record', choice_label1)
    k=cv2.waitKey(500)
    if esc_:
        cv2.destroyWindow('img')
        cv2.destroyWindow('show')
        break

video.release()
# save_=False
# next_=False
# esc_  =False
# ##结束录制

cap.open('train.mp4')
cnt=1
# print('FPS:{0}'.format(cap.get(cv2.CAP_PROP_FPS)))
# print('FRAME COUNT:{0}'.format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)

train_img=[]
chars_label=[]
draw_circle_flag=True
# print('please draw the region of interest')

show_label1=np.zeros((120,500,3)).astype(np.uint8)
cv2.line(show_label1,(0,5),(500,5),(0,255,0),thickness=3)
cv2.putText(show_label1,'Draw fixed ROI,then comfirm:',(5,35),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
cv2.putText(show_label1,'Join in train set?(SAVE)',(40,70),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
cv2.putText(show_label1,'Not Join(NEXT),Quit(ESC)',(40,110),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
choice_label1=np.concatenate([choice_label,show_label1],axis=0)


for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT ))):
    _,img=cap.read()
    # print('cnt:{0}'.format(cnt))
    img2 = img.copy()
    cv2.rectangle(img2, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
    cv2.imshow('img',img2)

    # print('join the train set?(SAVE) and next image(NEXT) or break(ESC) ')
    while True:
        save_ = False
        next_ = False
        esc_  = False
        cv2.imshow('record', choice_label1)
        cv2.waitKey(100)

        if save_ :
            if ix1==-1 or iy1==-1 or ix2==-1 or iy2==-1:
                cue = np.zeros((40, 400, 3)).astype(np.uint8)
                cv2.putText(cue, 'Draw ROI firstly', (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('bnh', cue)
                cv2.waitKey(1000)
                cv2.destroyWindow('bnh')
                break
            img2=img[iy1:iy2,ix1:ix2].copy()
            train_img.append(img2)
            chars_label.append(cnt)
            print('comfirm the label cnt:{0}'.format(cnt) )
            cnt+=1
            break
        elif next_ :
            break
        if esc_ :
            break
    if esc_ :
        break
cv2.destroyWindow('img')
print('{0},{1},{2},{3}'.format(ix1, iy1, ix2, iy2), file=open('location.txt', 'w'))

show_label1=np.zeros((80,500,3)).astype(np.uint8)
cv2.line(show_label1,(0,5),(500,5),(0,255,0),thickness=3)
cv2.putText(show_label1,'Delete this image(DEL)',(5,35),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
cv2.putText(show_label1,'Not delete(NEXT)',(40,70),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
choice_label1=np.concatenate([choice_label,show_label1],axis=0)

cap.release()
numdelete=[]
thresh_img=[]
for imglistnum in range(len(chars_label)):
    print(chars_label[imglistnum])
    cv2.imshow('img',train_img[imglistnum])

    ##分量提取并作后续工作
    B = train_img[imglistnum][:, :, 0]  # .copy()
    G = train_img[imglistnum][:, :, 1]  # .copy()
    R = train_img[imglistnum][:, :, 2]  # .copy()

    Bhist = cv2.calcHist([B],[0],None,[256],[0,256]).flatten()
    Ghist = cv2.calcHist([G],[0],None,[256],[0,256]).flatten()
    Rhist = cv2.calcHist([R],[0],None,[256],[0,256]).flatten()

    N=50
    weight=np.ones(N)/N
    Bhist_conv =np.concatenate([np.zeros(1), np.convolve(weight, Bhist )[N - 1:-N + 1].astype(int),np.zeros(1)])
    Ghist_conv =np.concatenate([np.zeros(1),  np.convolve(weight, Ghist )[N - 1:-N + 1].astype(int),np.zeros(1)])
    Rhist_conv =np.concatenate([np.zeros(1),  np.convolve(weight, Rhist )[N - 1:-N + 1].astype(int),np.zeros(1)])

    Bcnt=[0,]
    Bcnt_r=0
    for i in range(Bhist_conv.shape[0]-1):
        if Bhist_conv[i+1]>Bhist_conv[i]:
            Bcnt_r =i+1
        if Bhist_conv[i+1]<Bhist_conv[i] and Bcnt[-1]!=Bcnt_r:
            Bcnt.append(Bcnt_r)

    Gcnt=[0,]
    Gcnt_r=0
    for i in range(Ghist_conv.shape[0]-1):
        if Ghist_conv[i+1]>Ghist_conv[i]:
            Gcnt_r =i+1
        if Ghist_conv[i+1]<Ghist_conv[i] and Gcnt[-1]!=Gcnt_r:
            Gcnt.append(Gcnt_r)

    Rcnt=[0,]
    Rcnt_r=0
    for i in range(Rhist_conv.shape[0]-1):
        if Rhist_conv[i+1]>Rhist_conv[i]:
            Rcnt_r =i+1
        if Rhist_conv[i+1]<Rhist_conv[i] and Rcnt[-1]!=Rcnt_r:
            Rcnt.append(Rcnt_r)

    Bmax_diff = Bcnt[-1] - Bcnt[1]
    Gmax_diff = Gcnt[-1] - Gcnt[1]
    Rmax_diff = Rcnt[-1] - Rcnt[1]

    # print('Bmax_diff:{0}'.format(Bmax_diff))
    # print('Gmax_diff:{0}'.format(Gmax_diff))
    # print('Rmax_diff:{0}'.format(Rmax_diff))
    flag = False
    if Bmax_diff > 150 and (Gmax_diff > 150 or Rmax_diff > 150):
        flag = True
    elif Gmax_diff > 150 and Rmax_diff > 150:
        flag = True

    img_thresh = cv2.cvtColor(train_img[imglistnum], cv2.COLOR_BGR2GRAY)
    thresh_loc = 127
    if not flag:
        choose = np.argmax([Bmax_diff, Gmax_diff, Rmax_diff])
        if choose == 0:
            print('B')
            img_thresh = B
            thresh_loc = int((Bcnt[-1] + Bcnt[1]) / 2)
        #     bigrate_pot=np.argmax(Bhist)
        elif choose == 1:
            print('G')
            img_thresh = G
            thresh_loc = int((Gcnt[-1] + Gcnt[1]) / 2)
        #     bigrate_pot=np.argmax(Ghist )
        elif choose == 2:
            print('R')
            img_thresh = R
            # bigrate_pot=np.argmax(Rhist )#计算最大值在哪里，一次确定背景颜色
            thresh_loc = int((Rcnt[-1] + Rcnt[1]) / 2)


    _,img_thresh=cv2.threshold(img_thresh,thresh_loc+10,255,cv2.THRESH_BINARY)
    cv2.imshow('img_thresh', img_thresh)
    thresh_img.append(img_thresh)
    # print('clik (DEL) to delete this image\nclik (NEXT) to next one !!')

    while True:
        delete_ = False
        next_ = False
        cv2.imshow('record', choice_label1)
        cv2.waitKey(100)
        if delete_:
            numdelete.append(imglistnum)
            cue = np.zeros((40, 300, 3)).astype(np.uint8)
            cv2.putText(cue, 'Delete Done', (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('del',cue)
            cv2.waitKey(500)
            cv2.destroyWindow('del')
            # print('delete done')
            break
        elif next_:
            break


for num in reversed( numdelete):
    train_img.pop(num)
    chars_label.pop(num)
    thresh_img.pop(num)
key_choose_thresh= False#input('do u want use thresh img to produce train model?(save_/esc_)')

show_label1 = np.zeros((80, 500, 3)).astype(np.uint8)
cv2.line(show_label1,(0,5),(500,5),(0,255,0),thickness=3)
cv2.putText(show_label1, 'Use Binarization image to ', (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
cv2.putText(show_label1, 'make model file?(YES/ESC)', (5, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
choice_label1=np.concatenate([choice_label,show_label1],axis=0)

# print('Use thresh img to produce train model?(YES/ESC)')
while True:
    esc_ = False
    yes_ = False
    cv2.imshow('record', choice_label1)
    cv2.waitKey(100)
    if yes_:
        key_choose_thresh = True
        break
    elif esc_:
        break

print('%d'%key_choose_thresh , file=open('location.txt', 'a'))

cv2.destroyAllWindows()
chars_img=[]
if key_choose_thresh:#=='yes' or key_choose_thresh=='YES':
    chars_img=thresh_img
else :
    chars_img=train_img


###训练开始
# print('prepare data welly and begin training! ')
cue = np.zeros((80, 400, 3)).astype(np.uint8)
cv2.putText(cue, 'prepare data welly', (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
cv2.putText(cue, 'begin training!', (5, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
cv2.imshow('name',cue)
cv2.waitKey(500)
cv2.destroyWindow('name')


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=100, gamma=0.3):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()
winSize=(50,50)
blockSize=(20,20)
blockStride=(10,10)
cellSize=(10,10)
nbin=9
hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbin)

####训练模型代码
model=SVM()
chars_train = []
for digit_img in chars_img:
    if not key_choose_thresh:
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    digit_img=cv2.resize(digit_img,(50,50))
    # hog.compute(digit_img)
    chars_train.append(hog.compute(digit_img).flatten())
# chars_train = list(map(deskew, chars_train))
# chars_train = preprocess_hog(chars_train)
chars_label = np.array(chars_label)
model.train(np.array(chars_train) , chars_label)
model.save("svm.dat")







#
# def deskew(img):
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11'] / m['mu02']
#     M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
#     img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
#     return img
#
#
# def preprocess_hog(digits):
#     samples = []
#     for img in digits:
#         gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#         gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#         mag, ang = cv2.cartToPolar(gx, gy)
#         bin_n = 16
#         bin = np.int32(bin_n * ang / (2 * np.pi))
#         bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
#         mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
#         hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
#         hist = np.hstack(hists)
#
#         eps = 1e-7
#         hist /= hist.sum() + eps
#         hist = np.sqrt(hist)
#         hist /= norm(hist) + eps
#
#         samples.append(hist)
#     return np.float32(samples)
