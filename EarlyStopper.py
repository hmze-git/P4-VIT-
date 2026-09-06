class EarlyStopping:

    def __init__(self,patience=5,delta=0):
        self.patience=patience
        self.delta=delta
        self.bestLoss=float('inf')
        self.noImprovementCount=0
        self.stopTraining=False

    def stopEarly(self,validationLoss):
        if validationLoss<self.bestLoss:
            self.noImprovementCount=0
            self.bestLoss=validationLoss
        elif validationLoss>(self.bestLoss+self.delta):
            self.noImprovementCount+=1
            if self.patience<=self.noImprovementCount:
                self.stopTraining=True

        return self.stopTraining