# https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly_express as px



listUserImage={"F":'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80',"M":'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=987&q=80'}

@st.cache
def load_data():
   df1=pd.read_csv('./ResultatsFinal1.csv')
   df2=pd.read_csv('./ResultatsFinal2.csv')
   df3=pd.read_csv('./ResultatsFinal3.csv')
   df4=pd.read_csv('./ResultatsFinal4.csv')
   df5=pd.read_csv('./ResultatsFinal5.csv')
   df6=pd.read_csv('./ResultatsFinal6.csv')
   df7=pd.read_csv('./ResultatsFinal7.csv')
   df8=pd.read_csv('./ResultatsFinal8.csv')
   df1_2=pd.concat([df1,df2])
   df3_4=pd.concat([df3,df4])
   df5_6=pd.concat([df5,df6])
   df7_8=pd.concat([df7,df8])
   df1_2_3_4=pd.concat([df1_2,df3_4])
   df5_6_7_8=pd.concat([df5_6,df7_8])
   data=pd.concat([df1_2_3_4,df5_6_7_8])
   return data
df = load_data()

st.markdown(
"""
<style>
htlm{color:#F00;font-family: Verdana,Geneva,sans-serif; }
.reportview-container {
    background: #FFF;
    font-color:#FFF;
}
.sidebar .sidebar-content {
    
}
div.stButton > button:first-child { width:100%;}
</style>
""",
unsafe_allow_html=True
)
clients_list = df['SK_ID_CURR'].to_list()
clients_list.insert(0,"")
sideb=st.sidebar
choosen_client = sideb.selectbox('Which client?',clients_list[:40])
btnProfil=sideb.button('Profil')
btnLoan=sideb.button('Loan')
btnStats=sideb.button('Stats')
sideb.write("Help support:")
sideb.image("helpsupport.png")

def showDashData(choosen_client):
    if choosen_client == "" :
        currentStr = f"""
             <h1 style="color:#353b48;text-align:center;">Dashboard <span style="border:3px solid black;border-radius:5px;">PRET A DEPENSER</span></h1>
            <h3 style="color:#353b48;text-align:center;">Please select a Client in the sidebar in the left</h3>
            <hr />
            """
        components.html(currentStr)
        st.image("logo.png")
    else:
        currUser= df[(df['SK_ID_CURR']==choosen_client)].index
        if (df.loc[currUser]['CODE_GENDER'].tolist()[0]== 'F'):
            currImage="F"
        else:
            currImage="M"
        currentStr = f'''
            <h1>Dashboard client n°{choosen_client} - <img src="{listUserImage[currImage]}" style="width:4em;height:4em;float:right;"/></h1>
            <h3 style="color:#FFF;">Your personal data:</h3>
            <hr />
            <div style="width:100%;height:40%;background:#F2F2F2;">choose an action in the sidebar in the left</div>
            '''
        components.html(currentStr)
        if btnProfil:
            ProfilStr =f'''<h3>Your profil:</h3><li>name: </li><li>firstname: </li><li>sex:{df.loc[currUser]['CODE_GENDER'].tolist()[0]}</li><li>Loan Type:{df.loc[currUser]['NAME_CONTRACT_TYPE'].tolist()[0]} </li>'''
            components.html(ProfilStr)
        if btnLoan:
            if df.loc[currUser]['TARGET'].tolist()[0] >0.6:
                refundLoanVal="Good"
                LoanStr =f''' <p>The Probability that you can refund you loan is <span style="color:#0F0;">{refundLoanVal}</span></p><br/>'''
            else:
                refundLoanVal="Bad"
                LoanStr =f''' <p>The Probability that you can refund you loan is <span style="color:#F00;">{refundLoanVal}</span></p><br/>'''
            components.html(LoanStr)
            # Data
            targetVal = [1-df.loc[currUser]['TARGET'].tolist()[0],df.loc[currUser]['TARGET'].tolist()[0]]
            names = ['other', choosen_client]
            st.plotly_chart( px.pie(values=targetVal, names=names))
        if btnStats:
            components.html(f''' your income is : {df.loc[currUser]['AMT_INCOME_TOTAL'].tolist()[0]}''')
            fig1 = px.box(df, x='AMT_INCOME_TOTAL', y=clients_list[1:], color='CODE_GENDER', notched=True,)
            boxplot_chart = st.plotly_chart(fig1)
            components.html(f''' your credit is : {df.loc[currUser]['AMT_CREDIT'].tolist()[0]}''')
            fig2 = px.box(df, x='AMT_CREDIT', y=clients_list[1:], color='CODE_GENDER', notched=True,)
            boxplot_chart = st.plotly_chart(fig2)
            components.html(f''' your annuity is : {df.loc[currUser]['AMT_ANNUITY'].tolist()[0]}''')
            fig3 = px.box(df, x='AMT_ANNUITY', y=clients_list[1:], color='CODE_GENDER', notched=True,)
            boxplot_chart = st.plotly_chart(fig3)
        
        
        

showDashData(choosen_client)


