import DesicionTree


if __name__ == '__main__':
    
    dataset, labels = DesicionTree.createDataSet()
    featLabels = []
    myTree = DesicionTree.createTree(dataset, labels, featLabels)
    DesicionTree.createPlot(myTree)