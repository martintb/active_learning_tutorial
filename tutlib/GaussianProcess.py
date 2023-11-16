import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities.traversal import leaf_components

from tutlib.util import ternary_to_xy,xy_to_ternary

class GaussianProcess:
    def __init__(self,dataset,kernel=None):
        self.reset_GP(dataset,kernel)
        
    def construct_data(self):
        if 'labels_ordinal' not in self.dataset:
            raise ValueError('Must have labels_ordinal variable in Dataset before making GP!')

        labels = self.dataset['labels_ordinal'].values
        if len(labels.shape)==1:
            labels = labels[:,np.newaxis]
            
        domain = self.transform_domain()

        # X = tf.Variable(domain,shape=(None,None),trainable=False)
        # Y = tf.Variable(labels,shape=(None,None),trainable=False)
        # data = (X,Y)
        data = (domain,labels)
        return data
        
    def transform_domain(self,components=None):
      if components is None:
        components = self.dataset.attrs['components']
      if not (len(self.dataset.attrs['components'])==3):   
          raise ValueError("Ternary domain transform specified but len(components)!=3") 
      comp = self.dataset[components].to_array('component').transpose(...,'component')
      domain = ternary_to_xy(comp.values)
      return domain
            
    def reset_GP(self,dataset,kernel):
        self.dataset = dataset
        self.n_classes = dataset.attrs['n_phases']

        data = self.construct_data()
            
        invlink = gpflow.likelihoods.RobustMax(self.n_classes)  
        likelihood = gpflow.likelihoods.MultiClass(self.n_classes, invlink=invlink)  
        self.model = gpflow.models.VGP(
            data=data, 
            kernel=kernel, 
            likelihood=likelihood, 
            num_latent_gps=self.n_classes
        ) 
        self.loss = self.model.training_loss_closure(compile=True)
        self.trainable_variables = self.model.trainable_variables
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        

    def optimize(self,N,progress_bar=False):
        if progress_bar:
            for i in tqdm.tqdm(tf.range(N),total=N):
                self._step(i)
        else:
            for i in tf.range(N):
                self._step(i)
            
    @tf.function
    def _step(self,i):
        self.optimizer.minimize(self.loss,self.trainable_variables) 
    
    def predict(self,components):
        domain = self.transform_domain(components=components)
        self.y = self.model.predict_y(domain)
        self.y_mean = self.y[0].numpy() 
        self.y_var = self.y[1].numpy() 
        return {'mean':self.y_mean,'var':self.y_var}

