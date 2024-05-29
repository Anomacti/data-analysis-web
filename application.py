import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def standardize_data(data):
    # Select only the numeric features
    numeric_data = data.select_dtypes(include=['number'])

    if numeric_data.empty:
        st.write("No numeric features found in the dataset.")

    # Handle missing values
    numeric_data = numeric_data.fillna(numeric_data.mean())

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data)

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

def apply_k_nearest(data):
    st.write("Under construction")


# Function to display content for the "Home" tab
def show_home(data):
    st.title('Data upload')

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

# Function to display content for the "2D Visualization" tab
def show_2D_visual(data):
    st.title("2D Visualization Tab")
    st.write("This is the 2D Visualization Tab tab. Learn more about us here.")
    st.sidebar.write("This is the 2D Visualization Tab sidebar.")

    option = st.sidebar.selectbox(
        "Select 2D visualization algorithm",
        ("-- SELECT --", "PCA", "TSNE"))

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


# Function to display content for the "Machine Learning" tab
def show_machine_learning_classification(data):
    st.title("Machine Learning classification tab")
    st.write("This is the Machine Learning classification tab. Get in touch with us here.")
    st.sidebar.write("This is the Machine Learning classifiction sidebar.")

    option = st.sidebar.selectbox(
        "Select 2D visualization algorithm",
        ("-- SELECT --", "K-Nearest Neighbors"))

    # if option == "k-Nearest Neighbors":
        # Use algorithm

def show_machine_learning_clustering(data):
    st.title("Machine Learning clusteing tab")
    st.write("This is the Machine Learning clusteing tab. Get in touch with us here.")
    st.sidebar.write("This is the Machine Learning clusteing sidebar.")

def show_result():
    st.title("Result and Comparison tab")
    st.write("This is where the results will be displayed and compared")

def show_info():
    st.title("Info tab")
    st.write("Show info about the project")


def main():
    # Sidebar with tabs
    st.sidebar.title("File Upload")
    uploaded_file = st.sidebar.file_uploader('Upload your tabular data here', type=['txt', 'csv', 'xlsx'])

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Home", "2D Visualization", "Machine Learning: Classification", "Machine Learning: Clustering", "Result", "Info"])

    is_file_uploaded = False

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the uploaded file as a DataFrame using pandas, assuming no header
        data = pd.read_csv(uploaded_file, encoding='latin1')
        is_file_uploaded = True


    # Main content changes based on selected tab
    if tab == "Home":
        if is_file_uploaded:
            show_home(data)
        else:
            st.write("Upload data first")
    elif tab == "2D Visualization":
        if is_file_uploaded:
            show_2D_visual(data)
        else:
            st.write("Upload data first")
    elif tab == "Machine Learning: Classification":
        if is_file_uploaded:
            show_machine_learning_classification(data)
        else:
            st.write("Upload data first")
    elif tab == "Machine Learning: Clustering":
        if is_file_uploaded:
            show_machine_learning_clustering(data)
        else:
            st.write("Upload data first")
    elif tab == "Result":
        show_result()
    elif tab == "Info":
        show_info()


if __name__ == "__main__":
    main()
