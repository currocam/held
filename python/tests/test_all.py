import held
import pytest
import numpy as np
def test_streaming_stats_basic():
    """Test basic streaming statistics functionality."""
    from held import StreamingStats
    
    # Create a streaming stats object for 3 columns
    stats = StreamingStats(n_cols=3)
    
    # Update with first batch
    batch1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    stats.update(batch1)
    
    # Update with second batch
    batch2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
    stats.update(batch2)
    
    # Finalize and get means
    means = stats.finalize()
    
    # Expected means: (1+4+7+10)/4=5.5, (2+5+8+11)/4=6.5, (3+6+9+12)/4=7.5
    expected = np.array([5.5, 6.5, 8.5])
    
    np.testing.assert_array_almost_equal(means, expected)
