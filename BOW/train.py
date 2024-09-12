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
            files = os.listdir(self.path + '/' + dir)
            for i in range(self.trainSetNum):
                # filePaths = dataset/00/1.jpg
                filePath = self.path + '/' + dir + '/' + files[i]
                img = {'image': cv2.imread(filePath), 'label': dir}
                self.imgTrainSet.append(img)
            for i in range(self.trainSetNum, len(files)):
                filePath = self.path + '/' + dir + '/' + files[i]
                img = {'image': cv2.imread(filePath), 'label': dir}
                self.imgTestSet.append(img)
        return self.imgTrainSet,self.imgTestSet

# 特征提取image[i]["image"]，getSiftFeature使用SIFT对训练集特征提取，normalizeSIFT归一化特征
class FeaturesProcessor:
    def __init__(self, imgSet):
        self.imgSet = imgSet # [[img,label],[]]
        self.featureSet = []
        self.features = []
        self.centers = []
        self.dataset = []
        self.labelset = []

    def getFeaturesBySIFT(self):
        sift = cv2.SIFT_create()
        for i in range(len(self.imgSet)):
            gray = cv2.cvtColor(self.imgSet[i]['image'], cv2.COLOR_BGR2GRAY)
            # gray = self.imgSet[i]['image']
            keypoints, feature = sift.detectAndCompute(gray, None)
            self.featureSet.append({'feature': feature, 'label': self.imgSet[i]['label']})
            self.features.extend(feature)

    def normalizeFeatures(self):
        for i in range(len(self.features)):
            norm = np.linalg.norm(self.features[i])
            if norm > 1:
                self.features[i] /= float(norm)

    def createBoWByKMeans(self, wordnum ,random_state, batch_size):
        kmeans = MiniBatchKMeans(n_clusters=wordnum, random_state=random_state, batch_size=batch_size).fit(self.features)
        centers = kmeans.cluster_centers_
        self.centers = centers
        return centers

    def featuresToBoW(self, centers):
        for i in range(len(self.featureSet)):
            featVec = np.zeros(50)
            features = self.featureSet[i]['feature']
            for feature in features:
                diffMat = np.tile(feature, (50, 1)) - centers
                # axis=1按行求和，即求特征到每个中心点的距离
                sqSum = (diffMat ** 2).sum(axis=1)
                dist = sqSum ** 0.5
                # 升序排序
                sortedIndices = dist.argsort()
                # 取出最小的距离，即找到最近的中心点
                idx = sortedIndices[0]
                # 该中心点对应+1
                featVec[idx] += 1

            self.dataset.append(featVec)
            self.labelset.append(self.featureSet[i]['label'])



class Trainer:
    def __init__(self, imgTrainset):
        self.imgSet = imgTrainset # [[img,label],[]]
        self.FeaturesProcessor = FeaturesProcessor(self.imgSet)
    def train(self, wordnum,random_state, batch_size):
        self.FeaturesProcessor.getFeaturesBySIFT()
        self.FeaturesProcessor.normalizeFeatures()
        self.FeaturesProcessor.createBoWByKMeans(wordnum=wordnum,random_state=random_state, batch_size=batch_size)
        self.FeaturesProcessor.featuresToBoW(self.FeaturesProcessor.centers)
        clf = svm.SVC(kernel='rbf', C=1000, decision_function_shape='ovo')
        clf.fit(self.FeaturesProcessor.dataset, self.FeaturesProcessor.labelset)
        return clf, self.FeaturesProcessor.centers


if __name__ == '__main__':

    dataset = DatasetProcessor('dataset', 150)
    imgTrainset = dataset.imgTrainSet
    imgTestset = dataset.imgTestSet
    trainFeaturesProcessor = FeaturesProcessor(imgTrainset)
    trainFeaturesProcessor.getFeaturesBySIFT()
    trainFeaturesProcessor.normalizeFeatures()
    trainFeaturesProcessor.createBoWByKMeans(wordnum=50,random_state=3, batch_size=200)
    trainFeaturesProcessor.featuresToBoW(trainFeaturesProcessor.centers)



    testFeaturesProcessor = FeaturesProcessor(imgTestset)
    testFeaturesProcessor.getFeaturesBySIFT()
    testFeaturesProcessor.normalizeFeatures()
    testFeaturesProcessor.featuresToBoW(trainFeaturesProcessor.centers)

    print(trainFeaturesProcessor.dataset)
    print(testFeaturesProcessor.labelset)

    clf = svm.SVC(kernel='rbf', C=1000, decision_function_shape='ovo')
    clf.fit(trainFeaturesProcessor.dataset, trainFeaturesProcessor.labelset)
    prediction = clf.predict(testFeaturesProcessor.dataset)

    print(prediction)
    report = metrics.classification_report(testFeaturesProcessor.labelset, prediction)
    confuse_matrix = confusion_matrix(testFeaturesProcessor.labelset, prediction)

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