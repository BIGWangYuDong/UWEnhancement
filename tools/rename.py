import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self):
        self.listpath = os.listdir('/home/dong/python-project/Dehaze/DATA/# O-HAZY NTIRE 2018/GT')
        self.path = '/home/dong/python-project/Dehaze/DATA/# O-HAZY NTIRE 2018/GT'  # 表示需要命名处理的文件夹

    def rename(self):
        filelist =  self.listpath
        # 获取文件路径
        total_num = len(filelist) # 获取文件长度（个数）
        i = 1  # 表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.JPG'):  # 初始的图片的格式为png格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                src = os.path.join(os.path.abspath(self.path), item)
                item_0= item[:-4]
                item_0 = item_0.split('_')[0]
                item_0 = 'OHaze_' + item_0 + '.png'
                dst = os.path.join(os.path.abspath(self.path), item_0)
                # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')    这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
    # path_1 = os.path.realpath(__file__)
    # path_1, _ = os.path.split(path_1)
    # path_1, _ = os.path.split(path_1)
    # path_1, _ = os.path.split(path_1)
    # path_1, _ = os.path.split(path_1)
    # path_2 = 'dong/fog/JPEGImages00'
    # path = os.path.join(path_1, path_2)
    # filelist = os.listdir(path)
    # # /home/dong/fog/JPEGImages
    # print(filelist)

