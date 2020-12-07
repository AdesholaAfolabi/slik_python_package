import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def plot_nan(nan_values):
        
        """
        
        Here is where top 30 missing values are plotted before and after NaN
        values are detected using separate conditions
        
        """
        plot = nan_values.sort_values(ascending=False)[:30]
        
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
        ax.set_title('   Chart showing the top 30 missing values in the dataset', 
                     loc ='left', ) 

        # Add Text watermark 
        fig.text(0.9, 0.15, 'Alvaro', fontsize = 12, 
                 color ='grey', ha ='right', va ='bottom', 
                 alpha = 0.7) 

        # Show Plot 
        plt.show() 