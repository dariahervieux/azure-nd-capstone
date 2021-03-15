import joblib

# loading the trained model downloaded via Azure ML studio
model = joblib.load('../output/hd/best_model.joblib')

# preparing the data to score, the data is cleaned and has the same transformations as for training
score_input = [[29,8000,32000,32,8,32,200,
              0, 1, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0]]
           
result = model.predict(score_input)

print(f'prediction result: {result}')

