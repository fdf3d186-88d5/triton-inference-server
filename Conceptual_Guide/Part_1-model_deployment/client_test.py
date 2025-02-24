import requests
import pytest

# URLs
TRITON_METRICS_URL = "http://localhost:8002/metrics"
PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"

# List of expected Triton Prometheus metrics
EXPECTED_METRICS = [
    "nv_inference_request_success",
    "nv_model_load_count",
    "nv_gpu_memory_used_bytes",
    "nv_gpu_utilization",
    "nv_inference_compute_time_us",
    "nv_inference_request_duration_us"
]

def test_triton_metrics_endpoint():
    """Check if Triton Server's Prometheus metrics endpoint is available"""
    response = requests.get(TRITON_METRICS_URL)
    assert response.status_code == 200, "Triton metrics endpoint is unreachable"

def test_triton_metrics_exist():
    """Verify that expected Triton metrics are present"""
    response = requests.get(TRITON_METRICS_URL)
    metrics_data = response.text
    missing_metrics = [metric for metric in EXPECTED_METRICS if metric not in metrics_data]

    assert not missing_metrics, f"Missing metrics: {missing_metrics}"

@pytest.mark.parametrize("metric", EXPECTED_METRICS)
def test_prometheus_can_scrape_triton(metric):
    """Ensure Prometheus has collected Triton metrics"""
    query = {"query": metric}
    response = requests.get(PROMETHEUS_QUERY_URL, params=query)
    
    assert response.status_code == 200, f"Prometheus query failed for {metric}"
    
    result_data = response.json()
    assert "data" in result_data and "result" in result_data["data"], f"Invalid response format for {metric}"
    assert len(result_data["data"]["result"]) > 0, f"Metric {metric} not found in Prometheus"

if __name__ == "__main__":
    pytest.main(["-v", "test_triton_prometheus.py"])
