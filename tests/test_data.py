"""
Unit tests for data loading and preprocessing.
"""
import pandas as pd
import pytest

from src.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from src.data_loader import COLUMN_NAMES, load_raw_data
from src.preprocessing import clean_data, create_preprocessing_pipeline, split_data


class TestDataLoader:
    """Tests for data loading functionality."""

    def test_load_raw_data_returns_dataframe(self):
        """Verify that load_raw_data returns a pandas DataFrame."""
        df = load_raw_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_raw_data_has_correct_columns(self):
        """Verify that loaded data has expected columns."""
        df = load_raw_data()
        expected_columns = set(COLUMN_NAMES)
        actual_columns = set(df.columns)
        assert expected_columns == actual_columns

    def test_load_raw_data_not_empty(self):
        """Verify that the dataset is not empty."""
        df = load_raw_data()
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_target_column_is_binary(self):
        """Verify that target column contains only 0 and 1."""
        df = load_raw_data()
        unique_values = df[TARGET_COLUMN].unique()
        assert set(unique_values).issubset({0, 1})


class TestPreprocessing:
    """Tests for data preprocessing functionality."""

    @pytest.fixture
    def raw_data(self):
        """Load raw data once for all tests in this class."""
        return load_raw_data()

    def test_clean_data_returns_dataframe(self, raw_data):
        """Verify that clean_data returns a DataFrame."""
        cleaned = clean_data(raw_data)
        assert isinstance(cleaned, pd.DataFrame)

    def test_clean_data_same_row_count(self, raw_data):
        """Verify that clean_data doesn't drop rows unexpectedly."""
        cleaned = clean_data(raw_data)
        # Allow for some row drops due to missing values, but most should remain
        assert len(cleaned) >= len(raw_data) * 0.9

    def test_preprocessor_creates_pipeline(self):
        """Verify that preprocessing pipeline is created correctly."""
        preprocessor = create_preprocessing_pipeline()
        assert hasattr(preprocessor, "fit_transform")
        assert hasattr(preprocessor, "transform")

    def test_split_data_creates_correct_splits(self, raw_data):
        """Verify that data is split correctly."""
        cleaned = clean_data(raw_data)
        X_train, X_test, y_train, y_test = split_data(cleaned)

        # Check shapes
        assert len(X_train) > len(X_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # Check target values
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_preprocessor_handles_all_features(self, raw_data):
        """Verify that preprocessor can transform all features."""
        cleaned = clean_data(raw_data)
        X_train, X_test, _, _ = split_data(cleaned)

        preprocessor = create_preprocessing_pipeline()
        X_train_transformed = preprocessor.fit_transform(X_train)

        # Transformed data should have more columns due to one-hot encoding
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_train_transformed.shape[1] >= len(NUMERICAL_FEATURES)

    def test_split_data_stratified(self, raw_data):
        """Verify that split maintains class distribution."""
        cleaned = clean_data(raw_data)
        X_train, X_test, y_train, y_test = split_data(cleaned)

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        # Class distribution should be similar (within 10%)
        assert abs(train_ratio - test_ratio) < 0.1


class TestFeatureConsistency:
    """Tests for feature configuration consistency."""

    def test_all_features_list_correct(self):
        """Verify ALL_FEATURES contains numerical and categorical features."""
        expected = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        actual = set(ALL_FEATURES)
        assert expected == actual

    def test_no_duplicate_features(self):
        """Verify no duplicate feature names."""
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        assert len(all_features) == len(set(all_features))

    def test_features_exist_in_data(self):
        """Verify all configured features exist in the dataset."""
        df = load_raw_data()
        for feature in ALL_FEATURES:
            assert feature in df.columns, f"Feature {feature} not found in data"
