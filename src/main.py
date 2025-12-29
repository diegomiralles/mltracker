from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
import pandas as pd
from typing import Dict, List, Self, Tuple,Callable,Iterable
from dataclasses import dataclass,field,asdict,is_dataclass
from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score,accuracy_score
from xgboost import XGBClassifier
from datetime import datetime
import json

classification_metrics = {roc_auc_score,recall_score,precision_score,f1_score,accuracy_score}

def load_data()->Tuple[pd.DataFrame, pd.Series]:
    
    return load_breast_cancer(return_X_y=True,as_frame=True)
    

def split_data(X:pd.DataFrame,y:pd.Series,
               test_size:float=0.2,random_state:int=42
               )->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    
    return train_test_split(X,y,test_size=test_size,random_state=random_state)



@dataclass
class MLData:
    X_train:pd.DataFrame
    X_test:pd.DataFrame
    y_train:pd.Series
    y_test:pd.Series

@dataclass
class MLEvalData:
    y_true:pd.Series
    y_pred:pd.Series
    y_proba:pd.Series

@dataclass
class MLExperiment:
    name:str 
    algorithim:str
    params:Dict
    metrics_vals:Dict
    timestamp:datetime = field(default_factory=datetime.now)

@dataclass
class MLTracker:
    experiments:Dict[str,MLExperiment] = field(default_factory=dict)

    def add_experiment(self,experiment:MLExperiment)->Self:
        '''Adds the experiment to the dictionary and an ID '''
        experiment_id = str(len(self.experiments))
        self.experiments[experiment_id] = experiment
        return self
    def read(self,str_path:str)->Self:
        try:
            with open(str_path,'r') as json_file:
                data = json.load(json_file)
            for k,v in data.items():
                if 'timestamp' in v:
                    v['timestamp'] = datetime.fromisoformat(v['timestamp'])
                self.experiments[k] = MLExperiment(**v)
        except FileNotFoundError:
            pass
        return self
        
    def write(self,str_path:str)->Self:
        '''Writes the experiments attribute as a JSON file'''
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if is_dataclass(obj):
                return asdict(obj)
            if hasattr(obj, 'item'):
                return obj.item()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(str_path,'w') as json_file:
            json.dump(self.experiments,fp=json_file,indent=4,default=default_serializer)
        return self
    
    def summary(self):
        names = [exp.name for exp in self.experiments.values()]

        #TODO
        pass
    
        


@dataclass
class MLTrainer:
    model:Callable
    data: MLData

    def train(self)-> Self:
        self.model.fit(self.data.X_train,self.data.y_train)
        return self
    
    def _predict(self,X,y)->MLEvalData:
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:,1]
        return MLEvalData(y,y_pred,y_proba)
    
    def predict_train(self)->MLEvalData:
        return self._predict(self.data.X_train,self.data.y_train)
    
    def predict_test(self)->MLEvalData:
        return self._predict(self.data.X_test,self.data.y_test)
    

@dataclass
class MLEvaluator:
    data: MLEvalData
    metrics:Iterable[Callable] 

    def evaluate(self):
        results = {}
        for metric in self.metrics:
            if metric.__name__ == 'roc_auc_score':
                args = [self.data.y_true,self.data.y_proba]
            else:
                args = [self.data.y_true,self.data.y_pred]

            results[metric.__name__]=metric(*args)
        return results
    
@dataclass
class MLExecutor:
    trainer:MLTrainer
    evaluator:MLEvaluator

    def execute(self):
        self.trainer.train()
        self.evaluator.data = self.trainer.predict_test()
        return self.evaluator.evaluate()
    




if __name__ == "__main__":
    tracker = MLTracker()
    
    X,y = load_data()
    X_train,X_test,y_train,y_test = split_data(X,y)
    data = MLData(X_train,X_test,y_train,y_test)
    model = XGBClassifier()
    model.get_params()

    trainer = MLTrainer(model,data)
    
    predictions =trainer.train().predict_test()
    evaluator = MLEvaluator(predictions,metrics=classification_metrics)
    executor = MLExecutor(trainer,evaluator)
    experiment_metadada = MLExperiment(name='xgboost',algorithim='xgboost_classifier'
                                       ,params=model.get_params()
                                       ,metrics_vals=evaluator.evaluate())
    tracker.read('experiments.json')
    tracker.add_experiment(experiment_metadada)
    tracker.write('experiments.json')
                                     

    # print(X.describe(include='object'))
    # print(X.describe(include='number'))
    # print(y.value_counts())
