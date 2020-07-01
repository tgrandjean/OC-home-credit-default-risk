import os
from pathlib import Path
import pandas as pd
import warnings

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
        """Load CSV file as pandas.DataFrame."""
        return pd.read_csv(self.data_path.joinpath(file))

    def __assert_dataframe(self, dataframe):
        """Ensure data are provided as pandas dataframe.

        raises : ValueError if dataframe is not a pandas.DataFrame object.
        """
        if type(dataframe) != pd.DataFrame:
            raise ValueError("Invalid data format,"
                             " should be a pandas's dataframe")

    @property
    def data(self):
        """Internal representation of the features."""
        return self.__data

    @data.setter
    def data(self, dataframe):
        self.__assert_dataframe(dataframe)
        if 'SK_ID_CURR' in dataframe.columns:
            dataframe.set_index('SK_ID_CURR', inplace=True, drop=True)
        self.__data = dataframe

    @property
    def test_data(self):
        """Internale representation of the features for testing purpose."""
        return self.__test_data

    @test_data.setter
    def test_data(self, dataframe):
        self.__assert_dataframe(dataframe)
        if 'SK_ID_CURR' in dataframe.columns:
            dataframe.set_index('SK_ID_CURR', inplace=True, drop=True)
        self.__test_data = dataframe

    @property
    def full_data(self):
        return pd.concat([self.data, self.test_data], axis=0)

    @full_data.setter
    def full_data(self, dataframe):
        train_data = dataframe.loc[self.data.index]
        test_data = dataframe.loc[self.test_data.index]
        self.data = train_data
        self.test_data = test_data

    @property
    def application_train(self):
        """Return only relevant columns from the application_train file."""
        self.app_train = self.__load(self.FILES[0])
        return self.app_train[self.BASICS]

    @property
    def application_test(self):
        """Return only relevant columns from the application_test file."""
        self.app_test = self.__load(self.FILES[1])
        return self.app_test[self.BASICS]

    def handle_categorical_features(self, strategy='auto'):
        """Handle categorical data.

        strategy: should be one of "le" (LabelEncoder), "ohe" (OneHotEncoder)
        or "auto" ("le" if only two modalities and "ohe" for the others)
        """
        for col in self.__data:
            if self.__data[col].dtype.name == 'category':
                if strategy == 'le':
                    # LabelEncoder
                    pass
                elif strategy == 'ohe':
                    # OneHotEncoder
                    pass
                elif strategy == 'auto':
                    # Infer
                    pass
                else:
                    raise ValueError('Strategy must be one of'
                                     ' "le", "ohe" or "auto"')

    def handle_missing(self, strategy='auto'):
        """Handle missing values.

        strategy: should be "auto", "impute" or "drop"
        """
        for col in self.data.select_dtypes(exclude=["category", "object"]):
            if self.data[col].dtype.name == 'int64':
                warnings.warn('Integer data detected.'
                             ' Data must be converted to float')
            elif self.data[col].dtype.name == 'float64':
                # TODO: impute
                pass

    # TODO: Finish this. (Build all features that already exists in notebooks)
    ## Aggregated data
    @property
    def _history(self):
        """Return customer's history."""
        # Load concerned file

        # Merge on SK_ID_CURR

        return None

    def _active_credit(self):
        """For each appliant, we count all credits that are active."""
        sk_id_curr = self.data.reset_index(inplace=False,
                                           drop=False)['SK_ID_CURR']
        bureau = self.__load(self.FILES[2])
        # previous_application = self.__load(self.FILES[-2])
        # previous_application.set_index('SK_ID_CURR', inplace=True)
        active_cred_bur = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
        nb_active_cred = active_cred_bur[['SK_ID_CURR', 'CREDIT_ACTIVE']]\
            .groupby('SK_ID_CURR').count()
        self.full_data = self.full_data.join(nb_active_cred)

    def aggregate(self):
        """Call all function that make a feature by aggregate."""
        self.data['history'] = self._history

    def build(self):
        """Main entrypoint."""
        self.data = self.application_train
        self.test_data = self.application_test

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
