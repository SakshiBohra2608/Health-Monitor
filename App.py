from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import pandas as pd
import random

app=Flask(__name__)
#scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('classifier.pkl','rb'))
model1=pickle.load(open('linear1.pkl', 'rb'))

@app.route('/')
def welcome_page():
    return render_template('welcomePage.htm')\
    

@app.route('/home',methods=['GET','POST'])   
def homef():
    print('going home')
    return render_template('home.htm')

@app.route('/about')   
def aboutus():
    return render_template('about.htm')

@app.route('/faq')   
def faq():
    return render_template('faq.htm')

@app.route('/diet')   
def diet():
    return render_template('diet.html')


@app.route('/predict',methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    prg = request.form['prg']
    glc = request.form['gl']
    bp = request.form['bp']
    skt = request.form['sk']
    ins = request.form['ins']
    bmi = request.form['BMI']
    dpf = request.form['ped']
    age = request.form['age']

    prg = int(prg)
    glc = int(glc)
    bp = int(bp)
    skt = int(skt)
    ins = int(ins)
    bmi = float(bmi)
    dpf = float(dpf)
    age = int(age)
#   int_features = [int(x) for x in request.form.values()]
    final_features = np.array([(prg, glc, bp, skt, ins, bmi, dpf, age)])
    sc=pickle.load(open('scaler.pkl','rb'))
    final_features=sc.transform(final_features)

    prediction = model.predict(final_features)
    #output = round(prediction[0], 2)
    return render_template("result.htm", pred = prediction)

@app.route('/predict1',methods=['POST'])
def predict1():
    age = request.form['age']
    height = request.form['height']
    weight = request.form['weight']
    age = int(age)
    height = float(height)
    weight = int(weight)
    
    h_value = (height*12*0.0254)
    h_value = h_value*h_value
    bmi = weight/h_value
    print(bmi)
    
    df = pd.read_csv("a.csv")
    male = df[df.gender== 'M']
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    polyreg=PolynomialFeatures(degree=3)
    linear1=LinearRegression()
    X_male = male.iloc[:,0:1].values
    linear1.fit(X_male,male.calorie)
    Xmale_poly = polyreg.fit_transform(X_male)
    linearregmale=LinearRegression()
    linearregmale.fit(Xmale_poly,male.calorie)
    out=[0]
    c = ([age],)
    c = polyreg.fit_transform(c)
    out = linearregmale.predict(c)
    if bmi<18.5:
        req_cal = out[0]+200
    elif bmi>18.5 and bmi<24.9:
        req_cal = out[0]
    elif bmi>=25 and bmi<29.9:
        req_cal = out[0]-500
    else:
        req_cal = out[0]-1000
    
    print(req_cal)
    final_features1 = np.array([(age, height, weight)])
    # sc=pickle.load(open('scaler.pkl','rb'))
    # final_features=sc.transform(final_features)

    # calorie count
    breakfast_calorie = 15/100*req_cal
    lunch_calorie = 40/100*req_cal
    snack_calorie = 10/100*req_cal
    dinner_calorie = 35/100*req_cal
    
    # breakfast
    df_bf = pd.read_csv("Breakfast1.csv")
    #print(df_bf)
    b_carb = df_bf[df_bf.Type == 'Carbohydrate']
    b_protein = df_bf[df_bf.Type == 'Protein']
    b_fiber = df_bf[df_bf.Type == 'Fiber']
    listff = b_fiber.iloc[:,0].values
    listcf = b_fiber.iloc[:,2].values
    listfc = b_carb.iloc[:,0].values
    listcc = b_carb.iloc[:,2].values
    listfp = b_protein.iloc[:,0].values
    listcp = b_protein.iloc[:,2].values
    obtcal=0
    sumcal = 0
    listy=[]
    
    while sumcal < breakfast_calorie: 
        ran_p = random.choice(listfp)
        ran_c = random.choice(listfc)
        ran_f = random.choice(listff)
        
        result_p = np.where(listfp == ran_p)
        result_c = np.where(listfc == ran_c)
        result_f = np.where(listff == ran_f)
        
        cp = listcp[result_p]
        cf = listcf[result_f]
        cc = listcc[result_c]
        
        sumca= int(cp[0])+int(cf[0])+int(cc[0])
        print (sumca, breakfast_calorie)
        if sumca<(breakfast_calorie+30):
            sumcal=sumca 
            if sumcal>breakfast_calorie:
                listy.append(ran_p)
                listy.append(ran_c)
                listy.append(ran_f)
                
    obtcal=obtcal+sumcal
    
    df_lu = pd.read_csv("Lunch.csv")

    l_carb = df_lu[df_lu.type == 'Carbohydrate']
    l_protein = df_lu[df_lu.type == 'Protein']
    l_drink = df_lu[df_lu.type == 'Drink']
    l_veg = df_lu[df_lu.type == 'Vegetable']
    list_l_df = l_drink.iloc[:,0].values
    list_l_dc = l_drink.iloc[:,2].values
    list_l_cf = l_carb.iloc[:,0].values
    list_l_cc = l_carb.iloc[:,2].values
    list_l_pf = l_protein.iloc[:,0].values
    list_l_pc = l_protein.iloc[:,2].values
    list_l_vf = l_veg.iloc[:,0].values
    list_l_vc = l_veg.iloc[:,2].values
    sumcalL = 0
    listl=[]
    while sumcalL < lunch_calorie: 
        ran_p = random.choice(list_l_pf)
        ran_c = random.choice(list_l_cf)
        ran_c2 = random.choice(list_l_cf)
        ran_d = random.choice(list_l_df)
        ran_v = random.choice(list_l_vf)
        result_p = np.where(list_l_pf == ran_p)
        result_c = np.where(list_l_cf == ran_c)
        result_c2 = np.where(list_l_cf == ran_c2)
        result_d = np.where(list_l_df == ran_d)
        result_v = np.where(list_l_vf == ran_v)
        cp = list_l_pc[result_p]
        cd = list_l_dc[result_d]
        cc = list_l_cc[result_c]
        cc2 = list_l_cc[result_c2]
        cv = list_l_vc[result_v]
        sumca= int(cp[0])+int(cd[0])+int(cc[0])+int(cv[0])+int(cc2[0])
        if sumca<(lunch_calorie+50):
            sumcalL=sumca 
            if sumcalL>lunch_calorie:
                listl.append(ran_p)
                listl.append(ran_c)
                listl.append(ran_d)
                listl.append(ran_v)                         
    obtcal=obtcal+sumcalL
    
    df_sn = pd.read_csv("Snacks.csv")
    l_fiber = df_sn[df_sn.type == 'Fiber']
    l_protein = df_sn[df_sn.type == 'Protein']
    list_s_ff = l_fiber.iloc[:,0].values
    list_s_fc = l_fiber.iloc[:,2].values
    list_s_pf = l_protein.iloc[:,0].values
    list_s_pc = l_protein.iloc[:,2].values
    sumcal = 0
    lists=[]
    while sumcal < snack_calorie: 
        ran_p = random.choice(list_s_pf)
        ran_f = random.choice(list_s_ff)
        result_p = np.where(list_s_pf == ran_p)
        result_f = np.where(list_s_ff == ran_f)
        cp = list_s_pc[result_p]
        cf = list_s_fc[result_f]
        sumca= int(cp[0])+int(cf[0])
        if sumca<(snack_calorie + 20):
            sumcal=sumca 
            if sumcal>snack_calorie:
                lists.append(ran_p)
                lists.append(ran_f)
    obtcal=obtcal+sumcal
    
    df_di = pd.read_csv("Dinner.csv")
    s_carb = df_di[df_di.type == 'Carbohydrate']
    s_protein = df_di[df_di.type == 'Protein']
    s_drink = df_di[df_di.type == 'Drink']
    s_veg = df_di[df_di.type == 'Vegetable']
    s_fruit = df_di[df_di.type == 'Fruit']

    list_s_df = s_drink.iloc[:,0].values
    list_s_dc = s_drink.iloc[:,2].values
    list_s_cf = s_carb.iloc[:,0].values
    list_s_cc = s_carb.iloc[:,2].values
    list_s_pf = s_protein.iloc[:,0].values
    list_s_pc = s_protein.iloc[:,2].values
    list_s_vf = s_veg.iloc[:,0].values
    list_s_vc = s_veg.iloc[:,2].values
    list_s_ff = s_fruit.iloc[:,0].values
    list_s_fc = s_fruit.iloc[:,2].values
    sumcal = 0
    listd=[]
    while sumcal < dinner_calorie: 
        ran_p = random.choice(list_s_pf)
        ran_c = random.choice(list_s_cf)
        ran_d = random.choice(list_s_df)
        ran_v = random.choice(list_s_vf)
        ran_f = random.choice(list_s_ff)
        result_p = np.where(list_s_pf == ran_p)
        result_c = np.where(list_s_cf == ran_c)
        result_d = np.where(list_s_df == ran_d)
        result_v = np.where(list_s_vf == ran_v)
        result_f = np.where(list_s_ff == ran_f)
        cp = list_s_pc[result_p]
        cd = list_s_dc[result_d]
        cc = list_s_cc[result_c]
        cv = list_s_vc[result_v]
        cf = list_s_fc[result_f]
        sumca= int(cp[0])+int(cd[0])+int(cc[0]+int(cv[0])+int(cf[0]))
        if sumca<(dinner_calorie + 20):
            sumcal=sumca 
            if sumcal>dinner_calorie:
                listd.append(ran_p)
                listd.append(ran_c)
                listd.append(ran_d)
                listd.append(ran_v)
                listd.append(ran_f)
    obtcal=obtcal+sumcal

    arr = np.array([req_cal])
    newarr=arr.reshape(-1,1)
    pred1 = model1.predict(newarr)
    #output = round(prediction[0], 2)
    #listy=['breads', 'jam ', 'yogurt']
    return render_template("result1.html", pred = pred1, listy=listy, listl=listl, lists=lists, 
    listd=listd)

if __name__=="__main__":
        app.run(debug=True)