"""
Customized class for monitoring classification performance during model training/validation

Author: Ziping Liu
Date: Apr 19, 2024
"""



from collections import defaultdict
import numpy as np


class MetricMonitor():
    
    
    def __init__(self, float_precision=3):
        
        self.float_precision = float_precision
        self.reset()


    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "ct": 0})


    def update(self, metric_name, val, ct):
        
        metric = self.metrics[metric_name] # e.g. metric = self.metrics["iou"] = {"val": 0, "ct": 0}
        metric['val'] += val
        metric['ct'] += ct

    
    def value(self, metric_name):
        
        try:
            return (self.metrics[metric_name]['val'] / self.metrics[metric_name]['ct'])
        except:
            return np.nan
