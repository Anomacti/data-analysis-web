import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def clean_data(data):
    numeric_data = data.select_dtypes(exclude=["object"]).columns
    data = data[numeric_data]

    if data.empty:
        st.write("No numeric features found in the dataset.")
    else:
        data = data.fillna(data.mean())
        return data

def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    return standardized_data

def apply_pca(data):
    standardized_data = standardize_data(data)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(standardized_data)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

    return pca_df


def apply_tsne(data):
    standardized_data = standardize_data(data)

    tsne = TSNE(n_components=2, random_state=1)
    x_embedded = tsne.fit_transform(standardized_data)

    # Create a DataFrame with the principal components
    tsne_df = pd.DataFrame(data=x_embedded, columns=["PC1", "PC2"])

    return tsne_df 

def apply_k_nearest(data, target):
    X = data.drop(columns=[target])
    y = data[target]

    standardized_data = standardize_data(X)
    std_X = pd.DataFrame(standardized_data)

    X_train, X_test, y_train, y_test = train_test_split(std_X, y, random_state=2)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "cm": confusion_matrix(y_test, y_pred)
    }

    return y_pred, performance_metrics


def apply_random_forests(data, target):
    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "cm": confusion_matrix(y_test, y_pred)
    }

    return y_pred, performance_metrics


def apply_k_means(data):
    standardized_data = standardize_data(data)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    performance_metrics = {
        "inertia": kmeans.inertia_,
        "silhouette": silhouette_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels)
    }

    return labels, centers, performance_metrics


def apply_dbscan(data):
    dbscan = DBSCAN(eps=0.3, min_samples=4)

    labels = dbscan.fit_predict(data)

    performance_metrics = {
        "silhouette": silhouette_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels)
    }

    return labels, dbscan, performance_metrics


def show_home():
    st.title("Home page")

def show_2D_visual(data):
    st.title("2D Visualization Tab")
    st.write("In this tab you can visualize different aspects of your data and perform Dimensionality reduction.")

    st.title("Exploratory Data Analysis")

    st.subheader("Data")
    st.write(data.head())

    st.subheader("Statistical Summary")
    st.write(data.describe())

    st.subheader("Non standardized data")
    st.write("In order for the algorithms to work, non-numeric features where removed")
    st.scatter_chart(data)

    st.subheader("Standardized data")
    st.write("In order for the algorithms to work, non-numeric features where removed")
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
        st.title("PCA")
        pca_df = apply_pca(data)

        # Display PCA result
        st.write("PCA Result")
        st.write(pca_df)

        # Plot PCA result
        st.write("PCA Plot")
        st.scatter_chart(pca_df)
    elif option == "TSNE":
        st.title("TSNE")
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
        results["k-nearest"]["y_pred"], results["k-nearest"]["performance_metrics"] = apply_k_nearest(data, target)

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
            st.write(results["k-nearest"]["performance_metrics"]["accuracy"])
        with tab2:
            st.subheader("Report:")
            st.write(results["k-nearest"]["performance_metrics"]["report"])
        with tab3:
            st.subheader("Precision:")
            st.write(results["k-nearest"]["performance_metrics"]["precision"])
        with tab4:
            st.subheader("Recall:")
            st.write(results["k-nearest"]["performance_metrics"]["recall"])
        with tab5:
            st.subheader("F1 Score:")
            st.write(results["k-nearest"]["performance_metrics"]["f1"])
        with tab6:
            st.subheader("Confusion Matrix:")
            st.write(results["k-nearest"]["performance_metrics"]["cm"])


    st.header("Random Forests")
    if target != "--- SELECT ---":
        results["random forests"]["y_pred"], results["random forests"]["performance_metrics"] = apply_random_forests(data, target)

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
            st.write(results["random forests"]["performance_metrics"]["accuracy"])
        with tab2:
            st.subheader("Report:")
            st.write(results["random forests"]["performance_metrics"]["report"])
        with tab3:
            st.subheader("Precision:")
            st.write(results["random forests"]["performance_metrics"]["precision"])
        with tab4:
            st.subheader("Recall:")
            st.write(results["random forests"]["performance_metrics"]["recall"])
        with tab5:
            st.subheader("F1 Score:")
            st.write(results["random forests"]["performance_metrics"]["f1"])
        with tab6:
            st.subheader("Confusion Matrix:")
            st.write(results["random forests"]["performance_metrics"]["cm"])



        st.header("Results and comparisons")

        st.subheader("Accuracy")
        if results["k-nearest"]["performance_metrics"]["accuracy"] > results["random forests"]["performance_metrics"]["accuracy"]:
            st.write("K-nearest has better accuracy")
        else:
            st.write("Random forrest has better accuracy")

        st.subheader("Precision")
        if results["k-nearest"]["performance_metrics"]["precision"] > results["random forests"]["performance_metrics"]["precision"]:
            st.write("K-nearest has better precision")
        else:
            st.write("Random forrest has better precision")

        st.subheader("Recall")
        if results["k-nearest"]["performance_metrics"]["recall"] > results["random forests"]["performance_metrics"]["recall"]:
            st.write("K-nearest has better recall")
        else:
            st.write("Random forrest has better recall")

        st.subheader("F1")
        if results["k-nearest"]["performance_metrics"]["f1"] > results["random forests"]["performance_metrics"]["f1"]:
            st.write("K-nearest has better f1")
        else:
            st.write("Random forrest has better f1")


def show_ml_clustering(data):
    st.title("Machine Learning clusteing tab")
    st.write("This is the Machine Learning clusteing tab. Get in touch with us here.")
    st.sidebar.title("Clustering parameters")
    st.sidebar.write("This is the Machine Learning clusteing sidebar.")

    results = {
        "k-means": {},
        "DBSCAN": {}
    }


    st.header("K-means")
    results["k-means"]["labels"], results["k-means"]["centers"], results["k-means"]["performance_metrics"] = apply_k_means(data)

    tab1, tab2, tab3 = st.tabs([
        "inertia",
        "silhouette",
        "calinski_harabasz"
    ])

    with tab1:
        st.subheader("Inertia score:")
        st.write(results["k-means"]["performance_metrics"]["inertia"])
    with tab2:
        st.subheader("Silhouette score:")
        st.write(results["k-means"]["performance_metrics"]["silhouette"])
    with tab3:
        st.subheader("Precision:")
        st.write(results["k-means"]["performance_metrics"]["calinski_harabasz"])


    st.header("DBSCAN")
    results["DBSCAN"]["labels"], results["DBSCAN"]["dbscan"], results["DBSCAN"]["performance_metrics"] = apply_dbscan(data)

    tab1, tab2 = st.tabs([
        "silhouette",
        "calinski_harabasz"
    ])

    with tab1:
        st.subheader("Silhouette score:")
        st.write(results["DBSCAN"]["performance_metrics"]["silhouette"])
    with tab2:
        st.subheader("Precision:")
        st.write(results["DBSCAN"]["performance_metrics"]["calinski_harabasz"])


    st.header("Results and comparisons")

    st.subheader("Silhouette Score")
    if results["k-means"]["performance_metrics"]["silhouette"] > results["DBSCAN"]["performance_metrics"]["silhouette"]:
        st.write("K-nearest has better silhouette")
    else:
        st.write("Random forrest has better silhouette")

    st.subheader("Calinski Harabasz")
    if results["k-means"]["performance_metrics"]["calinski_harabasz"] > results["DBSCAN"]["performance_metrics"]["calinski_harabasz"]:
        st.write("K-nearest has better calinski_harabasz")
    else:
        st.write("Random forrest has better calinski_harabasz")

def show_info():
    st.title("Info tab")


def main():
    st.sidebar.title("File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your tabular data here", type=["txt", "csv", "xls", "xlsx"])

    dataset_has_index = st.sidebar.checkbox("Check if provided dataset has index column")

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Choose the tab you want to see", ["Home", "2D Visualization", "Machine Learning: Classification", "Machine Learning: Clustering", "Info"])

    is_file_uploaded = False
    cleaned_data = None

    if uploaded_file is not None:
        if dataset_has_index:
            data = pd.read_csv(uploaded_file, encoding="latin1", index_col=0)
            data = data.reset_index(drop=True)
        else:
            data = pd.read_csv(uploaded_file, encoding="latin1")
            data = data.reset_index(drop=True)

        is_file_uploaded = True
        cleaned_data = clean_data(data)

    # Main content changes based on selected tab
    if is_file_uploaded:
        if tab == "Home":
            show_home()
        elif tab == "2D Visualization":
            show_2D_visual(cleaned_data)
        elif tab == "Machine Learning: Classification":
            show_ml_classification(cleaned_data)
        elif tab == "Machine Learning: Clustering":
            show_ml_clustering(cleaned_data)
        elif tab == "Info":
            show_info()
    else:
        st.write("Upload data first")


if __name__ == "__main__":
    main()
