#!/usr/bin/env python3
"""
Direct test of dashboard endpoint functionality
"""
import requests
import traceback

def test_dashboard_endpoint():
    """Test dashboard endpoint directly"""
    try:
        # Test health endpoint first
        health_response = requests.get('http://localhost:3005/api/health', timeout=5)
        print(f"Health Status: {health_response.status_code}")
        print(f"Health Data: {health_response.json()}")
        
        # Test dashboard data endpoint
        print("\nTesting dashboard data endpoint...")
        response = requests.get('http://localhost:3005/api/dashboard-data', timeout=10)
        print(f"Dashboard Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Portfolio Balance: ${data['portfolio']['total_balance']:.2f}")
            print(f"Signals Count: {len(data['signals'])}")
            print(f"Data Source: {data['portfolio']['source']}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Connection Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_dashboard_endpoint()