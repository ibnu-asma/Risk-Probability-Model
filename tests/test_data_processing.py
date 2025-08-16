
import pandas as pd
import pytest
import sys
sys.path.append('src')
from data_processing import get_day_of_week

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'TransactionStartTime': ['2025-01-01 10:00:00', '2025-01-02 12:00:00'],
        'other_col': [1, 2]
    }
    return pd.DataFrame(data)

def test_get_day_of_week_column_creation(sample_dataframe):
    """Test that the 'day_of_week' column is created."""
    df = get_day_of_week(sample_dataframe)
    assert 'day_of_week' in df.columns

def test_get_day_of_week_correctness(sample_dataframe):
    """Test that the day of the week is calculated correctly."""
    df = get_day_of_week(sample_dataframe)
    # 2025-01-01 is a Wednesday (2), 2025-01-02 is a Thursday (3)
    assert df['day_of_week'].tolist() == [2, 3]
