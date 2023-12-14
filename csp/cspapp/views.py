from django.shortcuts import render,redirect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


def demo(request):
    return render(request,'cspapp/predict.html')

def index(request):
    if request.method =="POST":
        val1 = float(request.POST['n1'])
        val2 = float(request.POST['n2'])
        val3 = float(request.POST['n3'])
        val4 = float(request.POST['n4'])
        val5 = float(request.POST['n5'])
        res = normalize_user_input(val1,val2,val3,val4,val5)
       
        if request.POST['algo'] == 'ml' :

            input = np.array([[1,res[0],res[1],res[2],res[3],res[4]]])
            filename = 'linear_model.pkl'
            theta = pickle.load(open(filename,'rb'))
            y_pred = np.dot(input, theta)[0]
            total_runs = round((int(263-55) * y_pred[0]) + 55)
        
           
            context = {'total_runs':total_runs,'val1':val1,'val2':val2,'val3':val3,'val4':val4,'val5':val5,'ml': 'ml'}
           
            return render(request,'cspapp/predict.html',context)

        else:
            filename = 'forest.pkl'
            forest = pickle.load(open(filename,'rb'))
            mydict = [{'current_score': res[0], 'balls_left': res[1], 'wickets_left': res[2], 'crr': res[3],'last_five' : res[4]}]
            input = pd.DataFrame(mydict)
            prediction = random_forest_predictions(input,forest)
            total_runs = round((int(263-55) * prediction[0]) + 55)
    
            context = {'total_runs':total_runs,'val1':val1,'val2':val2,'val3':val3,'val4':val4,'val5':val5,'rf':'rf'}
            return render(request,'cspapp/predict.html',context)

    return render(request,'cspapp/predict.html',{"ml":'ml'})


def normalize_user_input(val1,val2,val3,val4,val5):
    current_score = (val1 - 8) / (263 - 8)
    balls_left = (val2- 0) / (98 - 0)
    wickets_left = (val3 - 0) / (10 - 0)
    crr = (val4 - 1.6) / (16.6 - 1.6)
    last_five = (val5 - 8) / (89 - 8)
    return current_score,balls_left,wickets_left,crr,last_five


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mean(axis=1)
    
    return random_forest_predictions


def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions

def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)