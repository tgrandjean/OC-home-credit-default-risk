import os
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

BASE_PATH = Path(__file__).resolve().parents[2]


class LabelEncoder(LabelEncoder):
    """Override the LabelEncoder in order to use it on pipeline."""

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y)

    def transform(self, y, *args, **kwargs):
        return super().transform(y)


class FeaturesBuilder():
    """Manage features proprely."""

    BASICS = [
        'SK_ID_CURR',
        'DAYS_BIRTH',
        'CODE_GENDER',
        'OCCUPATION_TYPE',
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'NAME_CONTRACT_TYPE',
        'AMT_ANNUITY',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3'
    ]

    FILES = [
        'application_train.csv',
        'application_test.csv',
        'bureau.csv',
        'bureau_balance.csv',
        'credit_card_balance.csv',
        'installments_payments.csv',
        'previous_application.csv',
        'POS_CASH_balance.csv'
    ]

    def __init__(self, data_path):
        self.data_path = Path(data_path).resolve()
        if not os.path.exists(self.data_path):
            raise ValueError("Path %s does not exist")

    def __load(self, file):
        return pd.read_csv(self.data_path.joinpath(file))

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, dataframe):
        if type(dataframe) != pd.DataFrame:
            raise ValueError("Must be a pandas's dataframe.")
        self.__data = dataframe

    @property
    def application_train(self):
        """Return only relevant columns from the application_train file."""
        self.app_train = self.__load(self.FILES[0])
        return self.app_train[self.BASICS]

    def handle_categorical_features(self, strategy='le'):
        """Handle categorical data.

        strategy: should be one of "le" (LabelEncoder), "ohe" (OneHotEncoder)
        or "auto" ("le" if only two modalities and "ohe" for the others)
        """
        for col in self.__data:
            if self.__data[col].dtype.name == 'category':
                pass

    def build(self):
        """Main entrypoint."""
        self.data = self.application_train

# Data loading
def load_data(src=BASE_PATH.joinpath('data', 'processed')):
    data = pd.read_csv(src.joinpath('features.csv'))
    return data.drop(columns='TARGET'), data['TARGET']

# Define Pipeline here
# In order to reuse the pipeline with various models, do not add the
# estimator to the pipeline.
categ_le_transformer = ColumnTransformer(
    transformers=[
        ('occupation_type', LabelEncoder(), ['OCCUPATION_TYPE',
                                             'CODE_GENDER',
                                             'NAME_CONTRACT_TYPE'])
    ], remainder='passthrough'
)

lgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
])

if __name__ == '__main__':
    X, y = load_data()
    X_enc = categ_le_transformer.fit_transform(X)
    # print(lgb_pipeline.fit_transform(X_enc))
