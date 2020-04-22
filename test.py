from sklearn.ensemble import RandomForestClassifier
import numpy as np

domainlist = []
class Domain:
    def __init__(self,_name,_label,_length):
        self.name = _name
        self.label = _label
        self.length = _length

    def returnData(self):
        return [self.length]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length = len(name)
            domainlist.append(Domain(name,label,length))

def main():
    initData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state = 0)
    clf.fit(featureMatrix,labelList)
    
    testfile = 'test.txt'
    resultfile = 'result.txt'
    with open(resultfile, 'a') as resultf:
        with open(testfile) as testf:
            for line in testf:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                tokens = line.split(",")
                testname = tokens[0]
                testlen = [len(testname)]
                testlabel = clf.predict([testlen])
                teststr = ""
                if testlabel == 0:
                    teststr = "notdga"
                else:
                    teststr = "dga"
                resultf.write(str(testname) + str(",") + str(teststr) + '\n')


if __name__ == '__main__':
    main()
