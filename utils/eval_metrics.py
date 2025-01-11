from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt
def display_confusion_matrix(predicted,true_output,labels,logger):

    cm=confusion_matrix(true_output,predicted)

    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    
    disp.plot(cmap="Blues")
    plt.show()

        

