import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
            # show_machine_learning_classification(data)
            st.write("TEMP: Do machine learning")
        elif tab == "Machine Learning: Clustering":
            # show_machine_learning_clustering(data)
            st.write("TEMP: Do machine learning")
        elif tab == "Result":
            show_result()
        elif tab == "Info":
            show_info()
    else:
        st.write("Upload data first")

if __name__ == "__main__":
    main()



# TODO: 
# 1. Clean the unnessasarry text
# 2. Remove any debugging writes/prints
# 3. Finish the structure of "Home" and "2D Visual" tabs
# 4. Proceed to the ML tabs
