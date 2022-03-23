import cv2


if __name__ == '__main__':
    img_path = 'images/bodys3.jpeg'
    save_path = 'images/bodys3_3.png'
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(img_raw.shape)
    # cv2.imwrite(save_path, img_raw, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])  # 0-9,default 3,higher num leads to compress deeper
    cv2.imwrite(save_path, img_raw)
