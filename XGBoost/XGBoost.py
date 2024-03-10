import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_tree, plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import logging

# Capture the logs from model.fit()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



STYLES = {
    "container": {"margin": "0px !important", "padding": "0!important", "align-items": "stretch", "background-color": "#fafafa"},
    "icon": {"color": "black", "font-size": "20px"}, 
    "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected": {"background-color": "lightblue", "font-size": "20px", "font-weight": "normal", "color": "black", },
}


@st.cache_data
def load_diabetes_dataset():
    diabetes = load_diabetes()
    df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df_diabetes['target'] = diabetes.target
    return df_diabetes

@st.cache_data
def load_iris_dataset():
    # URL of the Iris dataset on the UCI repository
    iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Define column names for the dataset
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # Load the Iris dataset into a Pandas DataFrame
    iris_df = pd.read_csv(iris_url, header=None, names=column_names)
    return iris_df

@st.cache_data
def load_custom_dataset(dataset):
    df = pd.read_csv(dataset)
    return df


class XGBoost(object):
    def __init__(self, title) -> None:
        self._title = title
        self._menu = {
            'title': self._title,
            'items': { 
                'Gradient Boosting' : {
                    'action': self._gradient_boosting, 'item_icon': 'lightbulb-fill'
                },
                'XGBoost-Hello World' : {
                    'action': self._xgboost, 'item_icon': 'card-heading'
                },
                'Data Prep' : {
                    'action': self._data_prep, 'item_icon': 'bricks'
                },
                'Evaluation' : {
                    'action': self._evaluate, 'item_icon': 'building-fill-up'
                },
                'Visualization' : {
                    'action': self._visualize, 'item_icon': 'file-bar-graph-fill'
                },
                'Feature Importance' : {
                    'action': self._feature_importance, 'item_icon': 'file-bar-graph-fill'
                },
                'Monitor & Multi-Threading' : {
                    'action': self._monitor, 'item_icon': 'file-bar-graph-fill'
                },
                'HyperParameter Tuning' : {
                    'action': self._tuning, 'item_icon': 'file-bar-graph-fill'
                },
            },
            'menu_icon': 'house',
            'default_index': 0,
            'orientation': 'vertical',
            'styles': STYLES
        }
        st.set_page_config(page_title=f'{self._title} series', layout='wide', page_icon=":rocket:")

        custom_css = """
            <style>
                body {
                    color: white;
                    background-color: #1E1E1E;  /* Dark background color */
                }
                .stApp {
                    color: white;
                    background-color: #1E1E1E;  /* Dark background color */
                }
            </style>
        """

        # Apply custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)
        self._setup_sidebar()

    def _setup_sidebar(self):
        kwargs = {
            'menu_title': self._menu['title'] ,
            'options': list(self._menu["items"].keys()),
            'icons': [v['item_icon'] for _k, v in self._menu['items'].items()],
            'menu_icon': self._menu['menu_icon'],
            'default_index': self._menu['default_index'],
            'orientation': self._menu['orientation'],
            # 'styles': self._menu['styles']
        }
        with st.sidebar:
            menu_selection = option_menu(**kwargs)
        self._menu["items"][menu_selection]['action']()

    def _gradient_boosting(self):
        st.markdown('<h3 style="color: orange;">Boosting</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <p>Boosting is converting a relatively <span style="color: green;">weak learners</span> into a very <span style="color: darkgreen;">good learners</span>(weak learners performances are slightly better than random chance)</p>
                <p>The idea is to use several weak learners to succession, where each weak learners are focussed on previous which found it
                 difficult and misclassified</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="color: orange;">Ada Boost</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <p>Simple terms <span style="color: green;">Decision Trees</span> with single split </p>
                <p>The idea is to use several weak learners to succession, where each weak learners are focussed on previous which found it
                 difficult and misclassified</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="color: orange;">Ada Boost - Working</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <ol type="1">
                    <li>Loss function<p>Depends on problem. Log loss(classification), squarred error(regression)</p></li>
                    <li>Weak learner<p>Decision Tree, specifically regression trees for real output values, where outputs can be added together
                    allowing subsequent model outputs to be added and correct the residuals in the predictions</p></li>
                    <li>Additive model<p>Trees are added sequentially, a gradient descent procedure is used to minimze the loass when adding trees</p></li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="color: orange;">Ada Boost - Improvements</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <ol type="1">
                    <li>Tree constraints
                        <ol type="A">
                            <li> No of trees (add until mno improvement is observed)</li>
                            <li> No of nodes</li>
                            <li> No of observations persplit</li>
                            <li> Minimum improvement</li>
                        </ol>
                    </li>
                    <li>Weighted updates<p>Contribution of each tree to this sum is weighted to slow down the learning rate (typically 0.1-0.3 or even lesser)</p></li>
                    <li>Stochastic gradient boosting
                        <p> 
                            At each iteration a subsample of training data is drawn at random from full training dataset. This randomly selected subsample are used to fit base learner
                            <ol type="A">
                                <li> Subsample rows before creating tree</li>
                                <li> Subsample columns before creating tree</li>
                                <li>Subsample columns before creating each split</li>
                                <li> Minimum improvement</li>
                            </ol>
                        </p>
                    </li>
                    <li>Penalty
                        <ol type="A">
                            <li> L1</li>
                            <li> L2</li>
                        </ol>
                    </li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    def _basic_iris_dataset(self):
        # Load the iris dataset
        df_iris = load_iris_dataset()
        st.write(df_iris.head())
        # Select features (X) and target variable (Y)
        X = df_iris.drop('class', axis=1)
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(df_iris['class'])
        y = label_encoder.transform(df_iris['class'])

        with st.expander("Data Types Summary"):
            st.dataframe(df_iris.dtypes)

        with st.expander("Descriptive Summary"):
            st.dataframe(df_iris.describe())

        with st.expander("Class Distribution"):
            st.dataframe(df_iris['class'].value_counts())

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a RandomForestRegressor model
        model = XGBClassifier()
        
        # Boolean variable to check if model.fit is run
        # Train the model and capture the logs
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

        st.write(accuracy_score(y_test, predictions))
        return model

    def _basic_diabetes_dataset(self, show_feature_imp=False):
        # Load the diabetes dataset
        df_diabetes = load_diabetes_dataset()
        st.write(df_diabetes.head())
        # Select features (X) and target variable (Y)
        X = df_diabetes.drop('target', axis=1)
        # Binarize the target variable
        threshold = 150  # You can adjust the threshold based on your criteria
        df_diabetes['target_class'] = (df_diabetes['target'] > threshold).astype(int)
        y = df_diabetes['target_class']

        with st.expander("Data Types Summary"):
            st.dataframe(df_diabetes.dtypes)

        with st.expander("Descriptive Summary"):
            st.dataframe(df_diabetes.describe())

        with st.expander("Class Distribution"):
            st.dataframe(df_diabetes['target_class'].value_counts())

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a RandomForestRegressor model
        st.session_state.diabetes_model = XGBClassifier()
        
        if 'diabetes_model_predict' not in st.session_state:
            st.session_state.diabetes_model_predict = None

        # Boolean variable to check if model.fit is run
        if st.session_state.diabetes_model_predict is None:
            if st.button("Fit and Predict"): 
                # Train the model and capture the logs
                with st.spinner("Training the model..."):
                    st.session_state.diabetes_model.fit(X_train, y_train)
                st.session_state.diabetes_model_predict = True

        if st.session_state.diabetes_model_predict:
            predictions = st.session_state.diabetes_model.predict(X_test)

            st.write(accuracy_score(y_test, predictions))

        if show_feature_imp and st.session_state.diabetes_model_predict:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_importance(st.session_state.diabetes_model, ax=ax)
            st.pyplot(fig)

    def _diabetes_model(self, show_feature_imp=False):
        if 'diabetes_model' not in st.session_state:
            st.session_state.diabetes_model = None

        if st.button("Diabetes") or st.session_state.diabetes_model:

            self._basic_diabetes_dataset(show_feature_imp=show_feature_imp)

            with st.expander('Code', expanded=False):
                code = '''
                #imports
                from xgboost import XGBClassifier, plot_importance
                from sklearn.feature_selection import SelectFromModel
                from sklearn.datasets import load_diabetes
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score

                #get dataset
                def load_diabetes_dataset():
                    diabetes = load_diabetes()
                    df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
                    df_diabetes['target'] = diabetes.target
                    return df_diabetes

                df_diabetes = load_diabetes_dataset()
                # Select features (X) and target variable (Y)
                X = df_diabetes.drop('target', axis=1)
                # Binarize the target variable
                threshold = 150  # You can adjust the threshold based on your criteria
                df_diabetes['target_class'] = (df_diabetes['target'] > threshold).astype(int)
                y = df_diabetes['target_class']

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create a XGBClassifier model
                model = XGBClassifier()

                #Fit and make predictions on test set
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                #print accuracy
                print(accuracy_score(y_test, predictions))
                '''
                if show_feature_imp:
                    code +="""
                    plot_importance(model)
                    thresholds = sort(model.feature_importances_)
                    for thresh in thresholds:
                    # select features using threshold
                    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                    select_X_train = selection.transform(X_train)
                    # train model
                    selection_model = XGBClassifier()
                    selection_model.fit(select_X_train, y_train)
                    # eval model
                    select_X_test = selection.transform(X_test)
                    predictions = selection_model.predict(select_X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    print(f"Thresh{thresh}, n={select_X_train.shape[1]}, Accuracy: {accuracy*100.0}%%")
                    """
                st.code(code, language='python')


    def _xgboost(self):
        st.markdown('<h3 style="color: orange;">XGBoost</h3>', unsafe_allow_html=True)
        st.write("Subset of gradient boosting")

        st.markdown('<h3 style="color: orange;">Features</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <ol type="1">
                <li>Gradient Boosting algorithm</li>
                <li>Stochastic gradient boosting with subsampling at row, column and column per split</li>
                <li>Regularised gradient boosting(L1 and L2)</li>
                <li>Parallelization of tree constructions</li>
                <li>Out of core computing power</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <br>
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <li>pip install xgboost</li>
                <li>pip install scikit-learn</li>
                <li>pip install ucimlrepo</li>
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )

        self._diabetes_model()

        if 'sepal_length' not in st.session_state:
            st.session_state.sepal_length = 0.0

        if 'sepal_width' not in st.session_state:
            st.session_state.sepal_width = 0.0

        if 'petal_length' not in st.session_state:
            st.session_state.petal_length = 0.0

        if 'petal_width' not in st.session_state:
            st.session_state.petal_width = 0.0

        if 'iris_model' not in st.session_state:
            st.session_state.iris_model = None

        if st.button("Iris") or st.session_state.iris_model:
            if not st.session_state.iris_model:
                st.session_state.iris_model = self._basic_iris_dataset()
            with st.expander('Code', expanded=False):
                code = '''
                #imports
                from xgboost import XGBClassifier
                from sklearn.preprocessing import LabelEncoder
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score

                #get dataset
                def load_iris_dataset():
                    # URL of the Iris dataset on the UCI repository
                    iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

                    # Define column names for the dataset
                    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

                    # Load the Iris dataset into a Pandas DataFrame
                    iris_df = pd.read_csv(iris_url, header=None, names=column_names)
                    return iris_df

                df_iris = load_iris_dataset()
                # Select features (X) and target variable (Y)
                X = df_iris.drop('class', axis=1)
                label_encoder = LabelEncoder()
                label_encoder = label_encoder.fit(df_iris['class'])
                y = label_encoder.transform(df_iris['class'])

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create a XGBClassifier model
                model = XGBClassifier()

                #Fit and make predictions on test set
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                #print accuracy
                print(accuracy_score(y_test, predictions))
                '''
                st.code(code, language='python')

            col1, col2 = st.columns(2)
            with col1:
                st.session_state.sepal_length = st.number_input("Enter Sepal Length:", step=0.1)
                st.session_state.sepal_width = st.number_input("Enter Sepal Width:", step=0.1)
            with col2:
                st.session_state.petal_length = st.number_input("Enter Petal Length:", step=0.1)
                st.session_state.petal_width = st.number_input("Enter Petal Width:", step=0.1)

            if st.button("Predict"):
                user_input = [[st.session_state.sepal_length, st.session_state.sepal_width, st.session_state.petal_length, st.session_state.petal_width]]
                prediction = st.session_state.iris_model.predict(user_input)[0]

                iris_class = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                # Display the prediction
                st.success(f"Predicted Iris species: {iris_class[prediction]}")


    def _data_prep(self):
        st.subheader("Label Encoder")

        if st.button("Perform Label Encoder"):
            df = load_iris_dataset()
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(df['class'])
            encoded_labels = label_encoder.transform(df['class'])

            col1, col2 = st.columns(2)
            # Create a DataFrame with original and encoded labels
            with col1:
                st.write(df['class'])
            with col2:
                st.write(
                     pd.DataFrame(
                         {
                        'Encoded_Label': encoded_labels
                        }
                    )
                )

        with st.expander('Code', expanded=False):
            code = '''
            #imports
            import pandas as pd
            from sklearn.preprocessing import LabelEncoder

            # URL of the Iris dataset on the UCI repository
            iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

            # Define column names for the dataset
            column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

            # Load the Iris dataset into a Pandas DataFrame
            iris_df = pd.read_csv(iris_url, header=None, names=column_names)
            label_encoder = LabelEncoder()
            label_encoded_y = label_encoder.fit_transform(df['class'])
            '''
            st.code(code, language='python')

        st.subheader("OneHot Encoder")
        if st.button("Perform OneHot Encoder"):
            df = load_iris_dataset()
            one_hot_encoder = OneHotEncoder(sparse_output=False)
            one_hot_encoder.set_output(transform='pandas')
            species_2d = df['class'].values.reshape(-1, 1)
            encoded_labels = one_hot_encoder.fit_transform(species_2d)

            col1, col2 = st.columns(2)
            # Create a DataFrame with original and encoded labels
            with col1:
                st.write(df['class'])
            with col2:
                st.write(encoded_labels)

        with st.expander('Code', expanded=False):
            code = '''
            #imports
            import pandas as pd
            from sklearn.preprocessing import OneHotEncoder, LabelEncoder

            # URL of the Iris dataset on the UCI repository
            iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

            # Define column names for the dataset
            column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

            # Load the Iris dataset into a Pandas DataFrame
            iris_df = pd.read_csv(iris_url, header=None, names=column_names)
            one_hot_encoder = OneHotEncoder(sparse_output=False)
            one_hot_encoder.set_output(transform='pandas')
            species_2d = iris_df['class'].values.reshape(-1, 1)
            encoded_labels = one_hot_encoder.fit_transform(species_2d)
            '''
            st.code(code, language='python')

        st.subheader("Missing Values")
        if st.button("Perform Handling Missing Values"):
            iris_df = load_iris_dataset()
            iris_df.loc[2, 'sepal_length'] = np.nan
            iris_df.loc[17, 'sepal_length'] = np.nan
            iris_df.loc[5, 'sepal_length'] = np.nan
            iris_df.drop(columns=['class'], inplace=True)
            # Extract a single column as a DataFrame
            column_to_impute = iris_df[['sepal_length']]
            # impute missing values as the mean
            imputer = SimpleImputer(strategy='mean')
            imputed_x = imputer.fit_transform(column_to_impute)

            col1, col2 = st.columns(2)
            # Create a DataFrame with original and encoded labels
            with col1:
                st.write(iris_df['sepal_length'])
            with col2:
                st.write(imputed_x)

            with st.expander('Code', expanded=False):
                code = '''
                #imports
                import pandas as pd
                from sklearn.impute impoer SimpleImputer

                # URL of the Iris dataset on the UCI repository
                iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

                # Define column names for the dataset
                column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

                # Load the Iris dataset into a Pandas DataFrame
                iris_df = pd.read_csv(iris_url, header=None, names=column_names)
                iris_df.loc[2, 'sepal_length'] = np.nan
                iris_df.loc[17, 'sepal_length'] = np.nan
                iris_df.loc[5, 'sepal_length'] = np.nan
                iris_df.drop(columns=['class'], inplace=True)
                # Extract a single row as a DataFrame
                column_to_impute = iris_df[['sepal_length']]
                # impute missing values as the mean
                imputer = SimpleImputer(strategy='mean')
                imputed_x = imputer.fit_transform(column_to_impute)
                '''
                st.code(code, language='python')

    def _evaluate(self):
        st.markdown('<h3 style="color: orange;">Cross Validation</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <p>Cross-validation can be used to estimate the performance of a machine learning algorithm with less variance than a single train-test set split.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        # Create a DataFrame with original and encoded labels
        with col1:
            st.subheader("KFold")

            st.markdown(
                """
                <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <p>It works by splitting the dataset into k-parts (e.g. k = 5 or k = 10). Each split of the data is called a fold. The
                        algorithm is trained on k - 1 folds with one held back and tested on the held back fold.This is
                        repeated so that each fold of the dataset is given a chance to be the held back test set.</p>
                </div>
                <br>
                """,
                unsafe_allow_html=True
            )
            with st.expander('Code', expanded=False):
                code = '''
                import pandas as pd
                from xgboost import XGBClassifier
                from sklearn.preprocessing import LabelEncoder
                from sklearn.model_selection import KFold
                from sklearn.model_selection import cross_val_score

                # URL of the Iris dataset on the UCI repository
                iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

                # Define column names for the dataset
                column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

                # Load the Iris dataset into a Pandas DataFrame
                iris_df = pd.read_csv(iris_url, header=None, names=column_names)
                label_encoder = LabelEncoder()
                label_encoded_y = label_encoder.fit_transform(iris_df['class'])

                # Split the dataset into training and test sets
                model = XGBClassifier()
                kfold = KFold(n_splits=10)
                results = cross_val_score(model, iris_df.drop('class', axis=1), label_encoded_y, cv=kfold)
                '''
                st.code(code, language='python')
        
        # Create a DataFrame with original and encoded labels
        with col2:
            st.subheader("StratifiedKFold")

            st.markdown(
                """
                <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <p>If class distribution is imbalanced go for StratifiedKFold</p>
                <p>It works by enforcing the same distribution of classes in each fold as in the whole training dataset when performing the cross-validation evaluation.</p>
                </div>
                <br>
                """,
                unsafe_allow_html=True
            )
            with st.expander('Code', expanded=False):
                code = '''
                import pandas as pd
                from xgboost import XGBClassifier
                from sklearn.preprocessing import LabelEncoder
                from sklearn.model_selection import StratifiedKFold
                from sklearn.model_selection import cross_val_score

                # URL of the Iris dataset on the UCI repository
                iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

                # Define column names for the dataset
                column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

                # Load the Iris dataset into a Pandas DataFrame
                iris_df = pd.read_csv(iris_url, header=None, names=column_names)
                label_encoder = LabelEncoder()
                label_encoded_y = label_encoder.fit_transform(iris_df['class'])

                # Split the dataset into training and test sets
                model = XGBClassifier()
                kfold = StratifiedKFold(n_splits=10)
                results = cross_val_score(model, iris_df.drop('class', axis=1), label_encoded_y, cv=kfold)
                '''
                st.code(code, language='python')

    def _visualize(self):
        if st.button("Perform Visualization"):
             # Load the diabetes dataset
            df_diabetes = load_diabetes_dataset()
            st.write(df_diabetes.head())
            # Select features (X) and target variable (Y)
            X = df_diabetes.drop('target', axis=1)
            # Binarize the target variable
            threshold = 150  # You can adjust the threshold based on your criteria
            df_diabetes['target_class'] = (df_diabetes['target'] > threshold).astype(int)
            y = df_diabetes['target_class']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a RandomForestRegressor model
            model = XGBClassifier()
            
            # Train the model and capture the logs
            with st.spinner("Training the model..."):
                model.fit(X_train, y_train)

            fig, ax = plt.subplots(figsize=(8, 6))
            plot_tree(model, ax=ax)
            st.pyplot(fig)

        with st.expander('Code', expanded=False):
            code = '''
            #imports
            from xgboost import XGBClassifier, plot_tree
            from matplotlib import pyplot
            from sklearn.datasets import load_diabetes
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            #get dataset
            def load_diabetes_dataset():
                diabetes = load_diabetes()
                df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
                df_diabetes['target'] = diabetes.target
                return df_diabetes

            df_diabetes = load_diabetes_dataset()
            # Select features (X) and target variable (Y)
            X = df_diabetes.drop('target', axis=1)
            # Binarize the target variable
            threshold = 150  # You can adjust the threshold based on your criteria
            df_diabetes['target_class'] = (df_diabetes['target'] > threshold).astype(int)
            y = df_diabetes['target_class']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a XGBClassifier model
            model = XGBClassifier()

            #Fit and make predictions on test set
            model.fit(X_train, y_train)
            plot_tree(model)
            pyplot.show()
            '''
            st.code(code, language='python')

    def _feature_importance(self):
        self._diabetes_model(show_feature_imp=True)

    def _monitor(self):
        st.markdown('<h3 style="color: orange;">Monitoring Training Performance</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <p>Monitoring can be done using eval_metric and eval_set</p>
                <ol type="1">
                <li>Eval Metrics</li>
                <li>rmse - regression - Root Mean Squared Error</li>
                <li>mae - regression - Mean Absolute Error</li>
                <li>logloss - Binaryclassification</li>
                <li>mlogloss - Multi-class classification</li>
                <li>error - classification</li>
                <li>auc - classification - Area Under ROC</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<h3 style="color: orange;">Early Stopping</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <p>Avoids overfitting</p>
                <p>Works by monitoring test set and stops training once the performance has not improved after specified iterations</p>
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )

        with st.expander('Code', expanded=False):
            code = '''
            #imports
            from xgboost import XGBClassifier, plot_importance
            from sklearn.feature_selection import SelectFromModel
            from sklearn.datasets import load_diabetes
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            #get dataset
            def load_diabetes_dataset():
                diabetes = load_diabetes()
                df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
                df_diabetes['target'] = diabetes.target
                return df_diabetes

            df_diabetes = load_diabetes_dataset()
            # Select features (X) and target variable (Y)
            X = df_diabetes.drop('target', axis=1)
            # Binarize the target variable
            threshold = 150  # You can adjust the threshold based on your criteria
            df_diabetes['target_class'] = (df_diabetes['target'] > threshold).astype(int)
            y = df_diabetes['target_class']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create a XGBClassifier model
            model = XGBClassifier()

            #Fit and make predictions on test set
            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(X_train, y_train, eval_metric=["error", "logloss"], early_stopping_rounds=10, eval_set=eval_set,
            verbose=True)
            predictions = model.predict(X_test)

            #print accuracy
            print(accuracy_score(y_test, predictions))

            # retrieve performance metrics
            results = model.evals_result()
            epochs = len(results['validation_0']['error'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = pyplot.subplots()
            ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
            ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
            ax.legend()
            pyplot.ylabel('Log Loss')
            pyplot.title('XGBoost Log Loss')
            pyplot.show()
            # plot classification error
            fig, ax = pyplot.subplots()
            ax.plot(x_axis, results['validation_0']['error'], label='Train')
            ax.plot(x_axis, results['validation_1']['error'], label='Test')
            ax.legend()
            pyplot.ylabel('Classification Error')
            pyplot.title('XGBoost Classification Error')
            pyplot.show()
            '''
            st.code(code, language='python')

        st.markdown('<h3 style="color: orange;">Multi Threading</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color: black; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #d3d3d3;">
                <ol type="1">
                    <li>2 ways</li>
                    <li>XGBoost model (default uses all cores for parallel thread)</li>
                    <li>cross_val_score</li>
                </ol>
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )

        with st.expander('Code', expanded=False):
            code = '''
            from xgboost import XGBClassifier
            from sklearn.model_selection import cross_val_score

            XGBClassifier(nthread=-1)

            #(or)
            #cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1, verbose=1)
            '''
            st.code(code, language='python')

    def _tuning(self):
        if 'otto_file' not in st.session_state:
            st.session_state.otto_file = None

        if not st.session_state.otto_file:
            # File uploader component
            uploaded_file = st.file_uploader("Choose otto-product kaggle dataset", type=["csv"])

            # Display information about the uploaded file
            if uploaded_file is not None:
                st.session_state.otto_file = uploaded_file
                st.header("Uploaded File Details:")

                # Display file type and size
                file_details = {
                    "File Name": uploaded_file.name,
                    "File Type": uploaded_file.type,
                    "File Size (bytes)": uploaded_file.size,
                }
                st.write(file_details)
        
        st.markdown('<h3 style="color: orange;">No of Trees</h3>', unsafe_allow_html=True)

        # Slider component
        n_estimators = st.slider("No of Trees", min_value=50, max_value=400, value=50, step=50)
        if n_estimators and not st.session_state.otto_file:
            st.warning("Otto Dataset needed to be uploaded")

        if st.session_state.otto_file:
            with st.spinner("Training the model..."):
                data = load_custom_dataset(st.session_state.otto_file)
                dataset = data.values
                # split data into X and y
                X = dataset[:, 0:94]
                y = dataset[:, 94]
                # encode string class values as integers
                label_encoded_y = LabelEncoder().fit_transform(y)
                # grid search
                model = XGBClassifier()
                kfold = KFold(n_splits=10)
                results = cross_val_score(model, X, label_encoded_y, cv=kfold)
                
                # summarize results
                st.success(f"No.of trees: {n_estimators} -> {max(results)}")
        
        with st.expander('Grid Search No of trees', expanded=False):
            code = '''
            from pandas import read_csv
            from xgboost import XGBClassifier
            from sklearn.model_selection import GridSearchCV,  StratifiedKFold
            from sklearn.preprocessing import LabelEncoder
            from matplotlib import pyplot

            # load data
            data = read_csv('/content/sample_data/otto.csv')
            dataset = data.values
            # split data into X and y
            X = dataset[:,0:94]
            y = dataset[:,94]
            # encode string class values as integers
            label_encoded_y = LabelEncoder().fit_transform(y)

            # grid search
            model = XGBClassifier()
            n_estimators = range(50, 400, 50)
            param_grid = dict(n_estimators=n_estimators)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
            grid_result = grid_search.fit(X, label_encoded_y)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            # plot
            pyplot.errorbar(n_estimators, means, yerr=stds)
            pyplot.title("XGBoost n_estimators vs Log Loss")
            pyplot.xlabel('n_estimators')
            pyplot.ylabel('Log Loss')
            pyplot.savefig('n_estimators.png')
            '''
            st.code(code, language='python')

        max_depth = st.slider("Depth of Trees", min_value=1, max_value=11, value=3, step=2)
        if max_depth and not st.session_state.otto_file:
            st.warning("Otto Dataset needed to be uploaded")

        if st.session_state.otto_file:
            with st.spinner("Training the model..."):
                data = load_custom_dataset(st.session_state.otto_file)
                dataset = data.values
                # split data into X and y
                X = dataset[:, 0:94]
                y = dataset[:, 94]
                # encode string class values as integers
                label_encoded_y = LabelEncoder().fit_transform(y)
                # grid search
                model = XGBClassifier(max_depth=max_depth)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
                results = cross_val_score(model, X, label_encoded_y, cv=kfold)
                
                # summarize results
                st.success(f"Max Depth: {max_depth} -> {max(results)}")
        
        with st.expander('Grid Search Max Depth', expanded=False):
            code = '''
            from pandas import read_csv
            from xgboost import XGBClassifier
            from sklearn.model_selection import GridSearchCV,  StratifiedKFold
            from sklearn.preprocessing import LabelEncoder
            from matplotlib import pyplot

            # load data
            data = read_csv('/content/sample_data/otto.csv')
            dataset = data.values
            # split data into X and y
            X = dataset[:,0:94]
            y = dataset[:,94]
            # encode string class values as integers
            label_encoded_y = LabelEncoder().fit_transform(y)

            # grid search
            model = XGBClassifier()
            max_depth = range(1, 11, 2)
            param_grid = dict(max_depth=max_depth)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
            grid_result = grid_search.fit(X, label_encoded_y)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            # plot
            pyplot.errorbar(max_depth, means, yerr=stds)
            pyplot.title("XGBoost max_depth vs Log Loss")
            pyplot.xlabel('max_depth')
            pyplot.ylabel('Log Loss')
            '''
            st.code(code, language='python')

        learning_rates = (0.0001, 0.001, 0.01, 0.1, 0.2, 0.3)

        learning_rate = st.radio("Learning Rate", learning_rates)

        if learning_rate and not st.session_state.otto_file:
            st.warning("Otto Dataset needed to be uploaded")

        if st.session_state.otto_file:
            with st.spinner("Training the model..."):
                data = load_custom_dataset(st.session_state.otto_file)
                dataset = data.values
                # split data into X and y
                X = dataset[:, 0:94]
                y = dataset[:, 94]
                # encode string class values as integers
                label_encoded_y = LabelEncoder().fit_transform(y)
                # grid search
                model = XGBClassifier(learning_rate=learning_rate)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
                results = cross_val_score(model, X, label_encoded_y, cv=kfold)
                
                # summarize results
                st.success(f"Learning Rate: {learning_rate} -> {max(results)}")
        
        with st.expander('Grid Search Learning Rate', expanded=False):
            code = '''
            from pandas import read_csv
            from xgboost import XGBClassifier
            from sklearn.model_selection import GridSearchCV,  StratifiedKFold
            from sklearn.preprocessing import LabelEncoder
            from matplotlib import pyplot

            # load data
            data = read_csv('/content/sample_data/otto.csv')
            dataset = data.values
            # split data into X and y
            X = dataset[:,0:94]
            y = dataset[:,94]
            # encode string class values as integers
            label_encoded_y = LabelEncoder().fit_transform(y)

            # grid search
            model = XGBClassifier()
            learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
            param_grid = dict(learning_rate=learning_rate)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
            grid_result = grid_search.fit(X, label_encoded_y)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            # plot
            pyplot.errorbar(learning_rate, means, yerr=stds)
            pyplot.title("XGBoost learning_rate vs Log Loss")
            pyplot.xlabel('learning_rate')
            pyplot.ylabel('Log Loss')
            '''
            st.code(code, language='python')
        

        row_subsampling = st.slider("Row Subsample", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        if row_subsampling and not st.session_state.otto_file:
            st.warning("Otto Dataset needed to be uploaded")

        if st.session_state.otto_file:
            with st.spinner("Training the model..."):
                data = load_custom_dataset(st.session_state.otto_file)
                dataset = data.values
                # split data into X and y
                X = dataset[:, 0:94]
                y = dataset[:, 94]
                # encode string class values as integers
                label_encoded_y = LabelEncoder().fit_transform(y)
                # grid search
                model = XGBClassifier(subsample=row_subsampling)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
                results = cross_val_score(model, X, label_encoded_y, cv=kfold)
                
                # summarize results
                st.success(f"Row Sampling: {row_subsampling} -> {max(results)}")
        
        with st.expander('Grid Search Row Subsample', expanded=False):
            code = '''
            from pandas import read_csv
            from xgboost import XGBClassifier
            from sklearn.model_selection import GridSearchCV,  StratifiedKFold
            from sklearn.preprocessing import LabelEncoder
            from matplotlib import pyplot

            # load data
            data = read_csv('/content/sample_data/otto.csv')
            dataset = data.values
            # split data into X and y
            X = dataset[:,0:94]
            y = dataset[:,94]
            # encode string class values as integers
            label_encoded_y = LabelEncoder().fit_transform(y)

            # grid search
            model = XGBClassifier()
            subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            param_grid = dict(subsample=subsample)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
            grid_result = grid_search.fit(X, label_encoded_y)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            # plot
            pyplot.errorbar(subsample, means, yerr=stds)
            pyplot.title("XGBoost subsample vs Log Loss")
            pyplot.xlabel('subsample')
            pyplot.ylabel('Log Loss')
            '''
            st.code(code, language='python')

        col_subsampling = st.slider("Column Subsample", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        if col_subsampling and not st.session_state.otto_file:
            st.warning("Otto Dataset needed to be uploaded")

        if st.session_state.otto_file:
            with st.spinner("Training the model..."):
                data = load_custom_dataset(st.session_state.otto_file)
                dataset = data.values
                # split data into X and y
                X = dataset[:, 0:94]
                y = dataset[:, 94]
                # encode string class values as integers
                label_encoded_y = LabelEncoder().fit_transform(y)
                # grid search
                model = XGBClassifier(colsample_bytree=col_subsampling)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
                results = cross_val_score(model, X, label_encoded_y, cv=kfold)
                
                # summarize results
                st.success(f"Column Sampling: {col_subsampling} -> {max(results)}")
        
        with st.expander('Grid Search Column Subsample', expanded=False):
            code = '''
            from pandas import read_csv
            from xgboost import XGBClassifier
            from sklearn.model_selection import GridSearchCV,  StratifiedKFold
            from sklearn.preprocessing import LabelEncoder
            from matplotlib import pyplot

            # load data
            data = read_csv('/content/sample_data/otto.csv')
            dataset = data.values
            # split data into X and y
            X = dataset[:,0:94]
            y = dataset[:,94]
            # encode string class values as integers
            label_encoded_y = LabelEncoder().fit_transform(y)

            # grid search
            model = XGBClassifier()
            colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            param_grid = dict(colsample_bytree=colsample_bytree)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
            grid_result = grid_search.fit(X, label_encoded_y)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            # plot
            pyplot.errorbar(colsample_bytre, means, yerr=stds)
            pyplot.title("XGBoost subsample vs Log Loss")
            pyplot.xlabel('colsample_bytre')
            pyplot.ylabel('Log Loss')
            '''
            st.code(code, language='python')

        col_subsample_by_split = st.slider("Column Subsample By split", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        if col_subsample_by_split and not st.session_state.otto_file:
            st.warning("Otto Dataset needed to be uploaded")

        if st.session_state.otto_file:
            with st.spinner("Training the model..."):
                data = load_custom_dataset(st.session_state.otto_file)
                dataset = data.values
                # split data into X and y
                X = dataset[:, 0:94]
                y = dataset[:, 94]
                # encode string class values as integers
                label_encoded_y = LabelEncoder().fit_transform(y)
                # grid search
                model = XGBClassifier(olsample_bylevel=col_subsample_by_split)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
                results = cross_val_score(model, X, label_encoded_y, cv=kfold)
                
                # summarize results
                st.success(f"Column Sampling per split: {col_subsample_by_split} -> {max(results)}")
        
        with st.expander('Grid Search Column Subsample By split', expanded=False):
            code = '''
            from pandas import read_csv
            from xgboost import XGBClassifier
            from sklearn.model_selection import GridSearchCV,  StratifiedKFold
            from sklearn.preprocessing import LabelEncoder
            from matplotlib import pyplot

            # load data
            data = read_csv('/content/sample_data/otto.csv')
            dataset = data.values
            # split data into X and y
            X = dataset[:,0:94]
            y = dataset[:,94]
            # encode string class values as integers
            label_encoded_y = LabelEncoder().fit_transform(y)

            # grid search
            model = XGBClassifier()
            colsample_bylevel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            param_grid = dict(colsample_bylevel=colsample_bylevel)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
            grid_result = grid_search.fit(X, label_encoded_y)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            # plot
            pyplot.errorbar(colsample_bylevel, means, yerr=stds)
            pyplot.title("XGBoost colsample_bylevel vs Log Loss")
            pyplot.xlabel('colsample_bylevel')
            pyplot.ylabel('Log Loss')
            '''
            st.code(code, language='python')

boost = XGBoost(title="XGBoost")
