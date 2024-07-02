import pandas as pd
from pandera import Check, Column, DataFrameSchema, Index, errors


class DiamondData:
    valid_cut = ["Ideal", "Premium", "Very Good", "Good", "Fair"]
    valid_col = ["H", "I", "F", "G", "E", "D", "J"]
    valid_cla = ["SI2", "SI1", "VS2", "IF", "VVS2", "VS1", "I1", "VVS1"]

    def __init__(self, data_source: str, drop_rows: bool) -> None:
        self.df = self._load_and_validate_data(data_source, drop_rows)

    def _load_and_validate_data(
        self, data_source: str, drop_rows: bool
    ) -> pd.DataFrame:
        """Load a csv dataset into a pandas data-frame and validate
        its schema i.e. check that all of the specified columns are
        there as well as their dtypes and values.
        If `drop_invalid_rows` is `True`, rows containing data not
        conforming to the schema are dropped.

        Args:
            `data_source` (str): path to a csv diamond dataset file.
            `drop_rows` (bool): if `True`, invalid data-frame rows
                are dropped.
        Returns:
            `validated_df` (pd.DataFrame)
        Raises:
            `SchemaErrors`: if the data-frame do not conform to the
                given schema and `drop_rows` is set to `False`.
            `ValueError`: if the resulting data-frame is empty as a
                consequence of dropping all of its rows because they
                do not conform to the given schema.
        """
        df = pd.read_csv(data_source)

        schema = DataFrameSchema(
            {
                "carat": Column(float, Check.gt(0)),
                "cut": Column(object, Check.isin(self.valid_cut)),
                "color": Column(object, Check.isin(self.valid_col)),
                "clarity": Column(object, Check.isin(self.valid_cla)),
                "depth": Column(float, Check.gt(0)),
                "table": Column(float, Check.gt(0)),
                "price": Column(int, Check.gt(0)),
                "x": Column(float, Check.gt(0)),
                "y": Column(float, Check.gt(0)),
                "z": Column(float, Check.gt(0)),
            },
            index=Index(int),
            strict=True,
            drop_invalid_rows=drop_rows,
        )

        try:
            validated_df = schema.validate(df, lazy=True)
            if validated_df.empty:
                raise ValueError(
                    "The validated DataFrame is empty. All rows were dropped."
                )
        except errors.SchemaErrors as e:
            print("\nSchema errors and failure cases:")
            print(e.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(e.data)
            exit(1)
        except ValueError as e:
            print(e)
            exit(1)
        else:        
            validated_df.reset_index(drop=True, inplace=True)
            return validated_df

    @property
    def lin_reg_data(self) -> pd.DataFrame:
        """Preprocess the dataset `self.df` to accomodate the changes
        required to fit a `sklearn.linear_model.LinearRegression` instance.

        Returns:
            pd.DataFrame
        """
        lr_df = self.df.drop(columns=["depth", "table", "y", "z"])
        lr_df = pd.get_dummies(
            lr_df, columns=["cut", "color", "clarity"], drop_first=True
        )

        return lr_df

    @property
    def xgboost_data(self) -> pd.DataFrame:
        """Preprocess the dataset `self.df` to accomodate the changes
        required to fit an `xgboost.XGBRegressor` instance.

        Returns:
            pd.DataFrame
        """
        xgb_df = self.df.copy(deep=True)
        xgb_df["cut"] = pd.Categorical(
            xgb_df["cut"], self.valid_cut, ordered=True)
        xgb_df["color"] = pd.Categorical(
            xgb_df["color"], self.valid_col, ordered=True)
        xgb_df["clarity"] = pd.Categorical(
            xgb_df["clarity"], self.valid_cla, ordered=True
        )

        return xgb_df
