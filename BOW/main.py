from train import Trainer, DatasetProcessor
from detect import Detector

if __name__ == '__main__':
    dataset = DatasetProcessor('dataset', 150)
    imgTrainset = dataset.imgTrainSet
    imgTestset = dataset.imgTestSet
    trainer = Trainer(imgTrainset)
    model, bow = trainer.train(wordnum=50, random_state=3, batch_size=200)
    print(len(trainer.FeaturesProcessor.features))
    detector = Detector(imgTestset, dataset.labels)
    detector.FeaturesProcessor.centers = trainer.FeaturesProcessor.centers
    result = detector.detect(model)

    detector.evaluate(result, "confuse_matrix")
