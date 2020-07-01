import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score



def main():
    st.title("Classification Model ")
    st.subheader("by Shallum Israel")
    st.sidebar.title("Classification models")
    st.markdown("Classify any dataset through various Algorithms")



    @st.cache(persist=True)
    def load_data():
           data = pd.read_csv("mushrooms.csv")
           label = LabelEncoder()
           for col in data.columns:
               data[col] = label.fit_transform(data[col])
           return data

    @st.cache(persist=True)

    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metric_list):
        if 'Confusion Matrix' in metric_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test)
            st.pyplot()
        if 'ROC Curve' in metric_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if 'Precission-Recall Curve' in metric_list:
            st.subheader("Precission Recall")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()











    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_name = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("SVM", "LR", "RF"))
    if classifier == 'SVM':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regulization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("kernel", ("rbf","linear" ), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficent)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precission-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("SVM Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy", round(accuracy, 2))
            st.write("Precision", precision_score(y_test, y_pred, labels=class_name))
            st.write("Recall", recall_score(y_test, y_pred, labels=class_name))

            plot_metrics(metrics)

    if classifier == 'LR':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regulization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maxiumum iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precission-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("LR Results")
            model = LogisticRegression(C=C, max_iter= max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy", round(accuracy, 2))
            st.write("Precision", precision_score(y_test, y_pred, labels=class_name))
            st.write("Recall", recall_score(y_test, y_pred, labels=class_name))

            plot_metrics(metrics)

    if classifier == 'RF':
        st.sidebar.subheader("Model Hyperparameters")

        n_est = st.sidebar.number_input("The no of trees", 100, 500, step=10, key='n_est')
        n_depth = st.sidebar.number_input("The depth of tree", 1, 20, step=1, key='n_depth')
        n_boots = st.sidebar.radio("Bootstrap sample", ("True", "False"), key='n_boots')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ("Confusion Matrix", "ROC Curve", "Precission-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Rf Results")
            model = RandomForestClassifier(n_estimators=n_est, max_depth=n_depth, bootstrap=n_boots, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy", round(accuracy, 2))
            st.write("Precision", precision_score(y_test, y_pred, labels=class_name))
            st.write("Recall", recall_score(y_test, y_pred, labels=class_name))

            plot_metrics(metrics)



    if st.sidebar.checkbox("Show data", False):
        st.subheader("Raw data")
        st.write(df.head())





if __name__ == '__main__':
         main()




## Run this command on terminal streamlit run MLwebApp.py
