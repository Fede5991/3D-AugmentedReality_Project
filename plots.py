# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 08:20:01 2019

@author: Fede
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix,accuracy_score

def plot_dataset_situation(classes,lengths,screens):
    plt.figure(figsize=(10,4))
    plt.bar(classes,lengths)
    plt.title('Models per class distribution')
    plt.xlabel('Class')
    plt.ylabel('Models')
    plt.xticks(rotation=90)
    plt.savefig("Models_per_class_distribution")
    plt.show()


    plt.figure(figsize=(10,4))
    plt.bar(classes,screens)
    plt.title('Screnshots per class distribution')
    plt.xlabel('Class')
    plt.ylabel('Screenshots')
    plt.xticks(rotation=90)
    plt.savefig("Screenshots_per_class_distribution")
    plt.show()

    
def plot_IAHOS(y,ogp,ogp2,tgp,tgp2):
    fig = make_subplots(rows=2, cols=2,subplot_titles=("Mean train. accuracy first round",
                                                   "Mean valid accuracy first round",
                                                  "Mean train accuracy last round",
                                                  "Mean valid accuracy last round"))

    x = np.linspace(0,len(tgp[0])-1,len(tgp[0]))
    Colorscale = [[0, '#FF0000'],[0.5, '#F1C40F'], [1, '#00FF00']]
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp2, colorscale = Colorscale),row=1,col=1)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp,colorscale=Colorscale),row=1,col=2)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp2, colorscale = Colorscale),row=2,col=1)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp,colorscale=Colorscale),row=2,col=2)
    fig.update_layout(height=600, width=800)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/IAHOS_'+str(model)+'.png')
    fig.show()

def plot_model(volume):    
    x=[]
    y=[]
    z=[]
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                if volume[i,j,k]>0:
                    x.append(i)
                    y.append(j)
                    z.append(k)
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers',
                          marker=dict(size=5,color=z,opacity=1))])
    fig.update_layout(width=800,height=400,autosize=True)
    fig.show()

def plot_training_accuracy(training_accuracy,optimizers,model):
    plt.figure(figsize=(12,4))
    for i in range(len(optimizers)):
        plt.plot(training_accuracy[i]) 
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(optimizers)
    plt.ylim(0, 1)
    plt.grid(True)
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/Training_accuracy'+str(model))
    plt.show()

def plot_validation_accuracy(validation_accuracy,optimizers,model):
    plt.figure(figsize=(12,4))
    for i in range(len(optimizers)):
        plt.plot(validation_accuracy[i]) 
    plt.title('Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(optimizers)
    plt.ylim(0, 1)
    plt.grid(True)
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/Validation_accuracy'+str(model))
    plt.show()
    
def plot_test_scores(scores,y,model):
    fig = go.Figure(data=[go.Bar(name='radam', x=scores, y=y[0],text=y[0]),
                      go.Bar(name='sgd', x=scores, y=y[1],text=y[1]),
                      go.Bar(name='rmsprop', x=scores, y=y[2],text=y[2]),
                      go.Bar(name='adagrad', x=scores, y=y[3],text=y[3]),
                      go.Bar(name='adadelta', x=scores, y=y[4],text=y[4]),
                      go.Bar(name='adam', x=scores, y=y[5],text=y[5]),
                      go.Bar(name='adamax', x=scores, y=y[6],text=y[6]),
                      go.Bar(name='nadam', x=scores, y=y[7],text=y[7])])
    fig.update_layout(barmode='group',width=800)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/test_'+str(model)+'.png')
    fig.show()
    
def plot_confusion_matrix(new_test_labels,y_pred,words_name,model):
    cm = confusion_matrix(y_true=new_test_labels,y_pred=y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.jet)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(len(words_name))
    plt.xticks(tick_marks,words_name,rotation=90)
    plt.yticks(tick_marks,words_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/confusion_matrix_'+str(model)+'.png')
    plt.show()
    
def plot_output_NN(words_name,classifier,sample):
    plt.figure(figsize=(14,4))
    plt.bar(words_name,classifier.predict(sample)[0][0:55])
    plt.title('Probability distribution per class')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(rotation=90)
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/output.png')
    plt.show()