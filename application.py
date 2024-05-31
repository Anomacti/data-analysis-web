import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def clean_data(data):
    numeric_data = data.select_dtypes(exclude=["object"]).columns
    data = data[numeric_data]

    if data.empty:
        st.write("No numeric features found in the dataset.")
    else:
        # Handle missing values
        data = data.fillna(data.mean())
        return data

def standardize_data(data):
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    return standardized_data

def apply_pca(data):
    standardized_data = standardize_data(data)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(standardized_data)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    return pca_df


def apply_tsne(data):
    standardized_data = standardize_data(data)

    # Perform TSNE
    tsne = TSNE(n_components=2, random_state=1)
    x_embedded = tsne.fit_transform(standardized_data)

    # Create a DataFrame with the principal components
    tsne_df = pd.DataFrame(data=x_embedded, columns=['PC1', 'PC2'])

    return tsne_df 

def apply_k_nearest(data, target):
    X = data.drop(columns=[target])
    y = data[target]

    standardized_data = standardize_data(X)
    std_X = pd.DataFrame(standardized_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(std_X, y, random_state=2)

    # Initialize the k-NN classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = knn.predict(X_test)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, accuracy, report, precision, recall, f1, cm


def apply_random_forests(data, target):
    X = data.drop(columns=[target])
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    # Initialize the RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2)

    # Fit the model on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_classifier.predict(X_test)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, accuracy, report, precision, recall, f1, cm


def apply_k_means(data):
    standardized_data = standardize_data(data)

    # Create a KMeans instance with 2 clusters
    kmeans = KMeans(n_clusters=2)

    # Fit the model to the data
    kmeans.fit(data)
    # kmeans.fit(standardized_data)

    # Get the cluster labels
    labels = kmeans.labels_

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    return labels, centers


def apply_dbscan(data):
    dbscan = DBSCAN(eps=0.3, min_samples=4)

    labels = dbscan.fit_predict(data)

    return 1, 2


# Function to display content for the "Home" tab
def show_home(data, cleaned_data):
    st.title('Home page')

    st.title("Testing/Debugging")

    st.subheader("Normal data")
    st.dataframe(data)

    st.subheader("Cleaned data")
    st.write(cleaned_data)

    st.subheader('Normal Data types')
    st.write(data.dtypes)

    st.subheader('Cleaned Data types')
    st.write(cleaned_data.dtypes)



# Function to display content for the "2D Visualization" tab
def show_2D_visual(data):
    st.title("2D Visualization Tab")
    st.write("This is the 2D Visualization Tab.")

    # Title of the Streamlit app
    st.title('Exploratory Data Analysis')

    # Display the dataframe
    st.subheader('Data')
    st.write(data.head())

    # Display basic information
    st.subheader('Data Information')
    buffer = data.info(buf=None)
    st.text(buffer)

    # Display statistical summary
    st.subheader('Statistical Summary')
    st.write(data.describe())


    # Display Data scatter chart
    st.scatter_chart(data)

    # Display Standardized data scatter chart
    data_columns = data.columns.tolist()
    standardized_data = standardize_data(data)
    std_data = pd.DataFrame(standardized_data)
    std_data.columns = data_columns
    st.scatter_chart(std_data)

    st.sidebar.write("This is the 2D Visualization Tab sidebar.")
    option = st.sidebar.selectbox(
        "Select 2D visualization algorithm",
        ["-- SELECT --", "PCA", "TSNE"])

    if option == "PCA":
        st.title('PCA')
        pca_df = apply_pca(data)

        # Display PCA result
        st.write("PCA Result")
        st.write(pca_df)

        # Plot PCA result
        st.write("PCA Plot")
        st.scatter_chart(pca_df)
    elif option == "TSNE":
        st.title('TSNE')
        tsne_df = apply_tsne(data)

        # Display TSNE result
        st.write("TSNE Result")
        st.write(tsne_df)

        # Plot TSNE result
        st.write("TSNE Plot")
        st.scatter_chart(tsne_df)



def show_ml_classification(data):
    st.title("Machine Learning classification tab")
    st.write("This is the Machine Learning classification tab.")
    st.sidebar.title("Classification parameters")
    st.sidebar.write("This is the Machine Learning classifiction sidebar.")

    results = {
        "k-nearest": {},
        "random forests": {}
    }

    features = data.columns.tolist()
    features.insert(0, "--- SELECT ---")

    target = st.sidebar.selectbox(
        "Select target feature",
        features
    )

    st.header("K-Nearest Neighbors")

    if target != "--- SELECT ---":
        # y_pred, accuracy, report, precision, recall, f1, cm = apply_k_nearest(data, target)

        results["k-nearest"]['y_pred'], results["k-nearest"]['accuracy'], results["k-nearest"]['report'], results["k-nearest"]['precision'], results["k-nearest"]['recall'], results["k-nearest"]['f1'], results["k-nearest"]['cm'] = apply_k_nearest(data, target)

        # st.header("Predicted values:")
        # st.write(y_pred)

        tab1, tab2, tab3, tab4, tab5, tab6= st.tabs([
            "Accuracy",
            "Report",
            "Precision",
            "Recall",
            "F1 Score",
            "Confusion Matrix"
        ])

        with tab1:
            st.subheader("Accuracy score:")
            st.write(results["k-nearest"]["accuracy"])
        with tab2:
            st.subheader("Report:")
            st.write(results["k-nearest"]["report"])
        with tab3:
            st.subheader("Precision:")
            st.write(results["k-nearest"]["precision"])
        with tab4:
            st.subheader("Recall:")
            st.write(results["k-nearest"]["recall"])
        with tab5:
            st.subheader("F1 Score:")
            st.write(results["k-nearest"]["f1"])
        with tab6:
            st.subheader("Confusion Matrix:")
            st.write(results["k-nearest"]["cm"])


    st.header("Random Forests")

    if target != "--- SELECT ---":
        # y_pred, accuracy, report, precision, recall, f1, cm = apply_random_forests(data, target)

        results["random forests"]['y_pred'], results["random forests"]['accuracy'], results["random forests"]['report'], results["random forests"]['precision'], results["random forests"]['recall'], results["random forests"]['f1'], results["random forests"]['cm'] = apply_random_forests(data, target)

        # st.header("Predicted values:")
        # st.write(y_pred)

        tab1, tab2, tab3, tab4, tab5, tab6= st.tabs([
            "Accuracy",
            "Report",
            "Precision",
            "Recall",
            "F1 Score",
            "Confusion Matrix"
        ])

        with tab1:
            st.subheader("Accuracy score:")
            st.write(results["random forests"]["accuracy"])
        with tab2:
            st.subheader("Report:")
            st.write(results["random forests"]["report"])
        with tab3:
            st.subheader("Precision:")
            st.write(results["random forests"]["precision"])
        with tab4:
            st.subheader("Recall:")
            st.write(results["random forests"]["recall"])
        with tab5:
            st.subheader("F1 Score:")
            st.write(results["random forests"]["f1"])
        with tab6:
            st.subheader("Confusion Matrix:")
            st.write(results["random forests"]["cm"])

    return results



def show_ml_clustering(data):
    st.title("Machine Learning clusteing tab")
    st.write("This is the Machine Learning clusteing tab. Get in touch with us here.")
    st.sidebar.title("Clustering parameters")
    st.sidebar.write("This is the Machine Learning clusteing sidebar.")

    option = st.sidebar.selectbox(
        "Select classification algorithm",
        ("--- SELECT ---", "K-means", "DBSCAN"))

    if option == "K-means":
        labels, centers = apply_k_means(data)
        data['Cluster'] = labels
        st.write(data)
        st.write("Cluster centers: ", centers)
    elif option == "DBSCAN":
        _, _ = apply_dbscan(data)




def show_result():
    st.title("Result and Comparison tab")
    st.write("This is where the results will be displayed and compared")

def show_info():
    st.title("Info tab")
    st.write("Show info about the project")


def main():
    # Sidebar with tabs
    st.sidebar.title("File Upload")
    uploaded_file = st.sidebar.file_uploader('Upload your tabular data here', type=['txt', 'csv', 'xls', 'xlsx'])

    dataset_has_index = st.sidebar.checkbox("Check this if your dataset has an index column ?")

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Home", "2D Visualization", "Machine Learning: Classification", "Machine Learning: Clustering", "Result", "Info"])

    is_file_uploaded = False
    cleaned_data = None
    performace_results = {
        "classification": {},
        "clustering": {}
    }

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the uploaded file as a DataFrame using pandas, assuming no header
        if dataset_has_index:
            data = pd.read_csv(uploaded_file, encoding='latin1', index_col=0)
            data = data.reset_index(drop=True)
        else:
            data = pd.read_csv(uploaded_file, encoding='latin1')
            data = data.reset_index(drop=True)

        is_file_uploaded = True
        cleaned_data = clean_data(data)

    # Main content changes based on selected tab
    if is_file_uploaded:
        if tab == "Home":
            show_home(data, cleaned_data)
        elif tab == "2D Visualization":
            show_2D_visual(cleaned_data)
        elif tab == "Machine Learning: Classification":
            performace_results["classification"] = show_ml_classification(cleaned_data)
        elif tab == "Machine Learning: Clustering":
            show_ml_clustering(cleaned_data)
        elif tab == "Result":
            show_result()
        elif tab == "Info":
            show_info()
    else:
        st.write("Upload data first")


    # st.header("War crimes continued")
    # st.write(performace_results)


if __name__ == "__main__":
    main()


# TODO:
# 1. Think about if you want to give user only discrete values
#     when selecting target values for ML
# 2. Try to find a good way to visualize K-Means
# 3. Standardize all data before doing ML algorithms
# 4. either use  '' or "" for strings
# 5. MAYBE use dictionary for "results" variable

# TODO: 
# 1. Clean the unnessasarry text
# 2. Remove any debugging writes/prints
# 3. Finish the structure of "Home" and "2D Visual" tabs
# 4. Proceed to the ML tabs
