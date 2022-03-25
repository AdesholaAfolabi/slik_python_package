import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def plot_nan(data):

    
    """
    
    Plot the top values from a value count in a dataframe.

    Parameters
    -----------
    data: DataFrame or name Series.
        Data set to perform plot operation on.
        
    Returns: A bar plot
        The bar plot of top n values.
    
    """
    plot = data.sort_values(ascending=False)[:30]
    
    # Figure Size 
    fig, ax = plt.subplots(figsize =(16, 9)) 

    # Horizontal Bar Plot 
    ax.barh(plot.index, plot.values) 

    # Remove axes splines 
    for s in ['top', 'bottom', 'left', 'right']: 
        ax.spines[s].set_visible(False) 
    # Remove x, y Ticks 
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 

    # Add padding between axes and labels 
    ax.xaxis.set_tick_params(pad = 5) 
    ax.yaxis.set_tick_params(pad = 10) 

    # Add x, y gridlines 
    ax.grid(b = True, color ='grey', 
            linestyle ='-.', linewidth = 0.5, 
            alpha = 0.2) 

    # Show top values  
    ax.invert_yaxis() 

    # Add annotation to bars 
    for i in ax.patches: 
        plt.text(i.get_width()+0.2, i.get_y()+0.5,  
                    str(round((i.get_width()), 2)), 
                    fontsize = 10, fontweight ='bold', 
                    color ='grey') 
    # Add Plot Title 
    ax.set_title('Chart showing the top 30 missing values in the dataset', 
                    loc ='left', ) 

    # Add Text watermark 
    fig.text(0.9, 0.15, 'Alvaro', fontsize = 12, 
                color ='grey', ha ='right', va ='bottom', 
                alpha = 0.7) 

    # Show Plot 
    plt.show() 

sns.set()

DPI = 150


def label_share(share, fp):

    '''
    
    The distribution of label in a data set.

    Parameters
    -----------
    share: label distribution
        
    fp: the file path to save the figure.

    Returns
    -------
    A bar plot of the label distribution

    '''
    share_norm = share / share.sum()
    fig, ax = plt.subplots()
    bar = sns.barplot(share_norm.index, share_norm.values)
    for idx, p in enumerate(bar.patches):
        bar.annotate('{:.2f}\n({})'.format(share_norm[idx], share[idx]),
                     (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                     ha='center', va='center', color='white', fontsize='large')
    ax.set_xlabel('Label')
    ax.set_ylabel('Share')
    ax.set_title('Label Share')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def corr_matrix(corr, fp):
    fig, ax = plt.subplots()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, vmin=-1, vmax=1, mask=mask,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                linewidths=0.5, cbar=True, square=True, ax=ax)
    ax.set_title('Correlation Matrix')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def confusion_matrix(cm, fp, norm_axis=1):
    
    """
    [TN, FP]
    [FN, TP]

    The confusion matrix after validating the model on a test set.

    Parameters
    -----------
    cm: confusion matrix
        
    fp: the file path to save the figure.

    Returns
    -------
    A bar plot of the confusion matrix

    """

    cm_norm = cm / cm.sum(axis=norm_axis, keepdims=True)
    TN, FP, FN, TP = cm.ravel()
    TN_norm, FP_norm, FN_norm, TP_norm = cm_norm.ravel()
    annot = np.array([
        [f'TN: {TN}\n({TN_norm:.3f})', f'FP: {FP}\n({FP_norm:.3f})'],
        [f'FN: {FN}\n({FN_norm:.3f})', f'TP: {TP}\n({TP_norm:.3f})']
    ])

    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=1,
                annot=annot, fmt='s', annot_kws={'fontsize': 'large'},
                linewidths=0.2, cbar=True, square=True, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def metric(metrics, fp):
    fig, ax = plt.subplots()
    for idx, data in enumerate(metrics):
        line = ax.plot(data['values'], label='fold{}'.format(idx), zorder=1)[0]
        ax.scatter(data['best_iteration'], data['values'][data['best_iteration'] - 1],
                   s=60, c=[line.get_color()], edgecolors='k', linewidths=1, zorder=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(metrics[0]['name'])
    ax.set_title('Metric History (marker on each line represents the best iteration)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def feature_importance(features, feature_importances, title, fp):

    '''
    The feature importance indicating features that contribute the
    most to the predictive power of the model.

    Parameters
    -----------
    features: features of the data set

    feature_importances: the importances of the features that contributes
                         the most to the predictive power of the model

    title: title of the feature importance chart
        
    fp: the file path to save the figure.

    Returns
    -------
    A bar plot of the feature importances
    '''
    fig, ax = plt.subplots()
    idxes = np.argsort(feature_importances)[::-1][:25]
    y = np.arange(len(idxes))
#     y = np.arange(len(feature_importances))
    ax.barh(y, feature_importances[idxes][::-1], align='center', height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(features[idxes][::-1])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def scores(scores, fp):

    '''
    The average classification score for model validation.

    Parameters
    -----------
    scores: test data scores
        
    fp: the file path to save the figure.

    Returns
    -------
    A bar plot of the average classification scores
    '''
    array = np.array([v for v in scores.values()]).reshape((2, 2))
    annot = np.array(['{}: {:.3f}'.format(k, v) for k, v in scores.items()]).reshape((2, 2))
    fig, ax = plt.subplots()
    sns.heatmap(array, cmap='Blues', vmin=0, vmax=1,
                annot=annot, fmt='s', annot_kws={'fontsize': 'large'},
                linewidths=0.1, cbar=True, square=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Average Classification Scores')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def roc_curve(fpr, tpr, auc, fp):

    '''
    The roc_curve for model validation.

    Parameters
    -----------
    fpr: false positive rate

    tpr: true positive rate

    auc: area under the curve
        
    fp: the file path to save the figure.

    Returns
    -------
    The ROC AUC curve
    
    '''
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], 'k:')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'ROC Curve (AUC: {auc:.3f})')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)


def pr_curve(pre, rec, auc, fp):

    '''
    The precision-recall curve for model validation.

    Parameters
    -----------
    pre: precision

    rec: recall

    auc: area under the curve
        
    fp: the file path to save the figure.

    Returns
    -------
    The precision-recall curve
    '''
    fig, ax = plt.subplots()
    ax.plot(pre, rec)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Presision')
    ax.set_title(f'Precision-Recall Curve (AUC: {auc:.3f})')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
