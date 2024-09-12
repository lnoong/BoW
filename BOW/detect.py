import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from train import FeaturesProcessor
class Detector:
    def __init__(self, imgTestset, labels):
        self.labels = labels
        self.imgSet = imgTestset # [[img,label],[]]
        self.FeaturesProcessor = FeaturesProcessor(self.imgSet)

    def detect(self, model):
        self.FeaturesProcessor.getFeaturesBySIFT()
        self.FeaturesProcessor.normalizeFeatures()
        self.FeaturesProcessor.featuresToBoW(self.FeaturesProcessor.centers)
        result = model.predict(self.FeaturesProcessor.dataset)
        return result

    def evaluate(self, result, savename):
        report = metrics.classification_report(self.FeaturesProcessor.labelset, result)
        confuse_matrix = confusion_matrix(self.FeaturesProcessor.labelset, result)

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
        ax.set_xticklabels([''] + self.labels, rotation=90)
        ax.set_yticklabels([''] + self.labels)
        plt.savefig(savename)
