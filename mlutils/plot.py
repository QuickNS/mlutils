import matplotlib.pyplot as plt 
from pandas.api.types import is_string_dtype, is_numeric_dtype
import seaborn as sns
from pdpbox import pdp

def set_plot_sizes(sml, med, big):
    plt.rc('font', size=sml)          # controls default text sizes
    plt.rc('axes', titlesize=sml)     # fontsize of the axes title
    plt.rc('axes', labelsize=med)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sml)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title

def plot_hist(df,c, title=None, figsize=(12,6)):
    _, ax = plt.subplots(figsize=figsize)
    if is_numeric_dtype(df[c]):
        df[c].plot.hist()
    else:
        sns.countplot(y=c, data=df, ax=ax)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_scatter(df, x, y, hue=None, title=None, size=8, fit_reg=False):
    sns.lmplot(x=x,y=y,data=df, hue=hue, fit_reg=fit_reg, height=size)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_joint(df, x, y, title=None, size=8, alpha=1):
    sns.jointplot(x=x,y=y,data=df, alpha=alpha, height=size)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_line(df, x, y, hue=None, title=None, figsize=(12,6), alpha=1, sort=True, xticks_rotation=0):
    _, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=x,y=y,data=df, hue=hue, alpha=alpha, sort=sort)
    plt.xticks(rotation=xticks_rotation)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_bar(df, x, y, hue=None, title=None, figsize=(12,6), alpha=1):
    _, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x,y=y,data=df, hue=hue, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_label_distribution_by_category(data, categorical_feature, label):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    sns.boxplot(x=categorical_feature,y=label, data=data,ax=ax)
    plt.xticks(rotation=90)
    ax2 = fig.add_subplot(122)
    sns.pointplot(x=categorical_feature,y=label, data=data,ax=ax2)
    plt.xticks(rotation=90)
    plt.show()

def plot_corr_matrix(matrix):   
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, ax=ax, center=0)
    tick_marks = [i for i in range(len(matrix.columns))]
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.columns)

def plot_confusion_matrix(confusionMatrix, labels):
 
    #plotting the confusion matrix
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(confusionMatrix, annot=True,annot_kws={"size": 12} ,linewidths=2, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted', labelpad=20)
    ax.set_ylabel('Truth', labelpad=20)

    """
    xtick_labels = list()
    ytick_labels = list()
    xtick_totals = np.sum(confusionMatrix, axis=0)
    ytick_totals = np.sum(confusionMatrix, axis=1)

    for i in range(len(labels)):
        xtick_labels.append("{0}\nTotal: {1}".format(labels[i], xtick_totals[i]))
        ytick_labels.append("{0}\nTotal: {1}".format(labels[i], ytick_totals[i]))
        
    ax.set_xticklabels(xtick_labels, rotation='horizontal', ha='center', size=12)
    ax.set_yticklabels(ytick_labels, rotation='horizontal', va='center', size=12)
    """

    plt.show()

def plot_classification_report(classification_report, title='Classification report', cmap='RdBu'):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    t = lines[-2].strip().split()
    scores = t[3:-1]
    total_samples = int(t[-1])
   
        
    xlabel = 'Metrics'
    ylabel = 'Classes'
    metrics = ['Precision', 'Recall', 'F1-score']
    xticklabels = ['{0}\n({1})'.format(metrics[idx], score) for idx, score  in enumerate(scores)]
    yticklabels = ['{0}\n({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    
    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(np.array(plotMat), annot=True,annot_kws={"size": 12} ,linewidths=2, fmt=".2f", vmin=0, vmax=1, cmap=cmap, ax=ax)
    ax.set_title("%s\nTotal: %d" % (title, total_samples))
    ax.set_xlabel(xlabel,labelpad=20)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_xticklabels(xticklabels, rotation='horizontal', ha='center')
    ax.set_yticklabels(yticklabels, rotation='horizontal', va='center')
   


def plot_partial_dependence(m, df, feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, df, df.columns, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)

def plot_keras_history(h, figsize=(16,6)):
    val_present:bool = False

    if 'val_acc' in h.history.keys():
        val_present = True

    accuracy = h.history['acc']
    loss = h.history['loss']
    if val_present:
        val_loss = h.history['val_loss']
        val_accuracy = h.history['val_acc']
    
    epochs = range(len(accuracy))

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(121)
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    if val_present:
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    ax2 = f.add_subplot(122)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    if val_present:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()