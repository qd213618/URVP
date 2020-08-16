import cv2
import glob

files=glob.glob("/home/lyb/datasets/vp/new4class_line/*/*.jpg")
for f in files:
    im=cv2.imread(f)
    kernel_size = (5, 5);
    sigma = 2.5;
    imm = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    #img = cv2.imread(imgName);
    img = cv2.GaussianBlur(im, kernel_size, sigma);
    imgg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.add(imgg,imm)
    #new_imgName = "New_" + str(kernel_size[0]) + "_" + str(sigma) + "_" + imgName;
    cv2.imwrite(f, img);
    # cv2.imshow('im',img)
    # cv2.waitKey(0)
