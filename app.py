import pickle
import streamlit as st
import pandas as pd
import numpy as np

# import the model
model = pickle.load(open('laptop_price_model.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
st.title('Laptop predictor')

#1.brand
brand = st.selectbox('Company',df['brand'].unique())
#2.processor_brand
processor_brand = st.selectbox('Processor Brand',df['processor_brand'].unique())
#3. processor_name
processor_name = st.selectbox('Processor Name',df['processor_name'].unique())
#4. processor_gnrtn
processor_gnrtn = st.selectbox('Processor Generation',df['processor_gnrtn'].unique())
#5. ram_gb
ram_gb = st.selectbox('RAM',df['ram_gb'].unique())
#6. ram_type
ram_type = st.selectbox('Type Of RAM',df['ram_type'].unique())
#7. ssd
ssd = st.selectbox('SSD',sorted(df['ssd'].unique()))
#8. hdd
hdd = st.selectbox('HDD',sorted(df['hdd'].unique()))
#9. os
os = st.selectbox('Operating System',df['os'].unique())
#10. os_bit
os_bit = st.selectbox('Operating System bit',df['os_bit'].unique())
#11. graphic_card_gb
graphic_card_gb = st.selectbox('Graphic Card GB',df['graphic_card_gb'].unique())
#12. weight
weight = st.selectbox('Weight',df['weight'].unique())
# 13.warranty
warranty = st.selectbox('warranty',df['warranty'].unique())
#14. Touchscreen
Touchscreen = st.selectbox('Touchscreen',df['Touchscreen'].unique())

if st.button('Predict price'):
    # query
    query = np.array([brand,processor_brand,processor_name,processor_gnrtn,
                      ram_gb,ram_type,ssd,hdd,os,os_bit,graphic_card_gb,weight,warranty
                      ,Touchscreen])
    query = query.reshape(1,14)
    st.title("The price for this configration is " + str(round(np.exp(model.predict(query)[0]),2)))
