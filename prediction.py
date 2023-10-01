import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
import pickle
import os
import json
import requests


def handle_keypress(event):
    """Print the character associated to the key pressed"""
    print(event.char)


def auto_fill(datas, entries, text_result, num_result):
    text_result.configure(text='')
    num_result.configure(text='')
    ndatas = len(datas)
    rand = np.random.randint(ndatas)
    for entry, dat in zip(entries, datas[rand]):
        entry.delete(0, tk.END)
        entry.insert(0, dat)


def make_predict(parms, user_inputs, text_result, num_result, pred_local):
    print("PREDICTION ON GOING...")
    df_dict = {}
    for npar, par in enumerate(parms):
        df_dict[par] = [user_inputs[npar].get()]
    df = pd.DataFrame(df_dict)
    df = df.replace('', np.nan)
    if df.isna().sum().sum() > 0:
        text_string = 'INPUT DATA IS EMPTY'
    else:
        fcat = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        fnum = [x for x in df.columns.values.tolist() if x not in fcat]
        df = df.astype('string')
        df[fcat] = df[fcat].astype('category')
        df[fnum] = df[fnum].astype('float64')
        prediction = model_predict(df, pred_local)
        text_result.configure(text="The prediction for the input data is:")
        if prediction == 0:
            text_string = 'NO CARDIOVASCULAR DISEASE'
        elif prediction == 1:
            text_string = 'CARDIOVASCULAR DISEASE DETECTED'
        else:
            text_string = 'ERROR IN PREDICTION'
    num_result.configure(text=text_string)


def model_predict(input_data, pred_local):
    if pred_local:
        # Read model from parent folder and make prediction
        mycwd = os.getcwd()
        os.chdir('..')
        with open(os.getcwd() + '\\model.pkl', "rb") as file:
            model = pickle.load(file)
        os.chdir(mycwd)
        print('Probability prediction: {}'.format(model.predict_proba(input_data)[0]))
        prediction = model.predict(input_data)[0]
    else:
        json_str = input_data.to_json(orient="records")  # Transform input data from dataframe to json string
        json_obj = json.loads(json_str)  # Transform the json string to json object
        response = requests.post("http://127.0.0.1:8000/predict", json=json_obj[0])  # Launch prediction request
        prediction = response.json()[0]  # Take the first value in the json object response
    print('Prediction: {}'.format(prediction))
    return prediction


# MAIN
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_rows', None)  # Enable option to display all dataframe rows
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
pd.set_option('display.max_seq_items', None)  # Enable printing the whole sequence content

local = False
params = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
data = [[17333.0, 2, 174.0, 88.0, 150.0, 90.0, 1, 1, 0, 0, 1],
        [18121.0, 1, 154.0, 70.0, 110.0, 70.0, 1, 1, 0, 0, 1],
        [20351.0, 1, 170.0, 72.0, 120.0, 65.0, 2, 3, 0, 0, 1]]

nparams = len(params)
window = tk.Tk()
title = tk.Label(master=window, text="CARDIOVASCULAR DISEASE PREDICTION", font=('Arial', 15))
title.grid(row=0, column=0, columnspan=2, sticky='NW', padx=10, pady=10)

img = Image.open("container.jpg")
resized_img = ImageTk.PhotoImage(img.resize((250, 120)))
label = tk.Label(master=window, image=resized_img)
label.grid(row=1, column=0, columnspan=2, sticky='N', padx=5, pady=5)

subtitle = tk.Label(master=window, text="Introduce the input data:", font=('Arial', 13))
subtitle.grid(row=2, column=0, sticky='', padx=2, pady=2)

autofill = tk.Button(master=window, relief=tk.RIDGE, borderwidth=5, text="AUTOFILL", width=15, font=('Arial', 12))
autofill.grid(row=2, column=1, sticky='', padx=10, pady=10)

param = []
gaps = []
for p in range(nparams):
    param.append(tk.Label(master=window, relief=tk.RAISED, borderwidth=5, text=params[p], width=20, font=('Arial', 12)))
    param[p].grid(row=p + 3, column=0, sticky='', padx=2, pady=2)
    gaps.append(tk.Entry(master=window, relief=tk.SUNKEN, borderwidth=5, font=('Arial', 12)))
    gaps[p].grid(row=p + 3, column=1, sticky='E', padx=2, pady=2)

button = tk.Button(master=window, relief=tk.RIDGE, borderwidth=5, text="PREDICT", width=20, font=('Arial', 15))
button.grid(row=nparams+3, column=0, columnspan=2, sticky='N', padx=10, pady=10)

res_label = tk.Label(master=window, font=('Arial', 13))
res_label.grid(row=nparams+4, column=0, columnspan=2, sticky='N')
res_number = tk.Label(master=window, font=('Arial', 13), fg='red')
res_number.grid(row=nparams+5, column=0, columnspan=2, sticky='N', padx=10, pady=10)

# Bind left click on button PREDICT
button.bind("<Button-1>",
            lambda event, p1=params, p2=gaps, p3=res_label, p4=res_number, p5=local: make_predict(p1, p2, p3, p4, p5))
# Bind left click on button AUTOFILL
autofill.bind("<Button-1>",
              lambda event, p1=data, p2=gaps, p3=res_label, p4=res_number: auto_fill(p1, p2, p3, p4))
# Bind keypress event to handle_keypress()
window.bind("<Key>", handle_keypress)
window.mainloop()
