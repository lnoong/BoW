import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn import metrics
# 划分训练集和测试集，图像+标签
class DatasetProcessor:
    def __init__(self, path, trainSetNum):
        self.path = path
        self.trainSetNum = trainSetNum
        self.imgTrainSet = []
        self.imgTestSet = []
        self.labels = []
        self.divTrainAndTest()

    def divTrainAndTest(self):
        # dirs=['00','01',...]
        dirs = os.listdir(self.path)
        self.labels = dirs
        for dir in dirs:
            # files = 1.png,2.png...
            files = os.listdir(self.path + "/" + dir)
            for i in range(self.trainSetNum):
                # filePaths = dataset/00/1.jpg
                filePath = self.path + "/" + dir + "/" + files[i]
                img = {'image': cv2.imread(filePath), 'label': dir,'sift': [], 'histogram': []}
                self.imgTrainSet.append(img)
            for i in range(self.trainSetNum, len(files)):
            # for i in range(self.trainSetNum, self.trainSetNum+1):
                filePath = self.path + "/" + dir + "/" + files[i]
                img = {'image': cv2.imread(filePath), 'label': dir,'sift': [], 'histogram': []}
                self.imgTestSet.append(img)
# 特征提取image[i]["image"]，getSiftFeature使用SIFT对训练集特征提取，normalizeSIFT归一化特征
class FeatureExtractor:
    def __init__(self, imgTrainSet):
        self.imgTrainSet = imgTrainSet
        self.features = []
        self.getSiftFeature()
        self.normalizeSIFT()

    def getSiftFeature(self):
        sift = cv2.SIFT_create()
        nums = 0
        for i in range(len(self.imgTrainSet)):
            gray = cv2.cvtColor(self.imgTrainSet[i]["image"], cv2.COLOR_BGR2GRAY)
            keypoints, feature = sift.detectAndCompute(gray, None)
            # print(len(feature)) # 特征点个数
            # print(len(feature[0])) # 0号点128维
            # image_with_keypoints = cv2.drawKeypoints(gray, keypoints, None)
            # cv2.imshow('Image with Keypoints', image_with_keypoints)
            # nums += len(feature)
            # print(nums)
            # cv2.waitKey(0)
            self.imgTrainSet[i]['sift'] = feature
            self.features.extend(feature)



    def normalizeSIFT(self):
        for i in range(len(self.features)):
            norm = np.linalg.norm(self.features[i])
            if norm > 1:
                self.features[i] /= float(norm)


class FeatureCluster:
    def __init__(self, features):
        self.features = features

    def createBOWByKMeans(self):
        wordCnt = 100
        kmeans = MiniBatchKMeans(n_clusters=wordCnt, random_state=3, batch_size=200).fit(self.features)
        centers = kmeans.cluster_centers_
        return centers


if __name__ == '__main__':
    dataset = DatasetProcessor("dataset", 150)
    imgTestSet = dataset.imgTestSet
    imgTrainSet = dataset.imgTrainSet
    print(dataset.labels)
    print(len(dataset.imgTrainSet), len(dataset.imgTestSet), len(dataset.imgTrainSet) + len(dataset.imgTestSet))

    trainsift = FeatureExtractor(imgTrainSet)
    testsift = FeatureExtractor(imgTestSet)
    kmeans = FeatureCluster(trainsift.features)
    centers = kmeans.createBOWByKMeans()

    # 训练集投射到词袋上的直方图
    for i in range(len(imgTrainSet)):
        featVec = np.zeros((1, 100))
        features = imgTrainSet[i]['sift']
        for feature in features:
            diffMat = np.tile(feature, (100, 1)) - centers
            # axis=1按行求和，即求特征到每个中心点的距离
            sqSum = (diffMat ** 2).sum(axis=1)
            dist = sqSum ** 0.5
            # 升序排序
            sortedIndices = dist.argsort()
            # 取出最小的距离，即找到最近的中心点
            idx = sortedIndices[0]
            # 该中心点对应+1
            featVec[0][idx] += 1
        imgTrainSet[i]['histogram'] = featVec

    for i in range(len(imgTestSet)):
        featVec = np.zeros((1, 100))
        features = imgTestSet[i]['sift']
        for feature in features:
            diffMat = np.tile(feature, (100, 1)) - centers
            # axis=1按行求和，即求特征到每个中心点的距离
            sqSum = (diffMat ** 2).sum(axis=1)
            dist = sqSum ** 0.5
            # 升序排序
            sortedIndices = dist.argsort()
            # 取出最小的距离，即找到最近的中心点
            idx = sortedIndices[0]
            # 该中心点对应+1
            featVec[0][idx] += 1
        imgTestSet[i]['histogram'] = featVec

    trainData = []
    testData = []
    trainLabels = []
    testLabels = []
    for i in range(len(imgTestSet)):
        lable = imgTestSet[i]['label']
        testLabels.append(lable)

    for i in range(len(imgTrainSet)):
        lable = imgTrainSet[i]['label']
        trainLabels.append(lable)

    print("********************")
    print(testLabels)
    for img in imgTrainSet:
        trainData.append(img['histogram'][0])
        print(img['histogram'][0])
        # trainData = img['histogram']
    for img in imgTestSet:
        testData.append(img['histogram'][0])
        print(img['histogram'][0])

    clf = svm.SVC(kernel='rbf', C=1000, decision_function_shape='ovo')
    clf.fit(trainData, trainLabels)
    prediction = clf.predict(testData)

    print(prediction)
    report = metrics.classification_report(testLabels, prediction)
    confuse_matrix = confusion_matrix(testLabels, prediction)



    # print(centers)
    # print(len(centers))

    print(report)
    print(confuse_matrix)

    matrix = confuse_matrix.astype(float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center', fontsize=6)
    ax.set_xticklabels([''] + dataset.labels, rotation=90)
    ax.set_yticklabels([''] + dataset.labels)
    plt.savefig("confuse_matrix.png")

