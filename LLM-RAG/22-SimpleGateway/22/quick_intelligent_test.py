#!/usr/bin/env python3
"""Intelligent Gateway Test"""
import requests


def test_gateway():
    """Test intelligent model selection"""
    base_url = "http://localhost:8000"

    # Health check
    try:
        if requests.get(f"{base_url}/health", timeout=5).status_code != 200:
            print("❌ Gateway service is unhealthy")
            return
        print("✅ Gateway service is healthy")
    except Exception:
        print("❌ Unable to connect to the gateway service")
        return

    # Test cases - Fix: add expected selection notes
    test_cases = [
        {"question": "What is AI?", "expected": "Fast model"},
        {"question": "Please explain the basic principles of machine learning", "expected": "Balanced model"},
        {"question": "Design a complete distributed recommendation system architecture", "expected": "Premium model"},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['question']}")
        print(f"Expected: {case['expected']}")

        try:
            print("Processing request...")
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": case["question"]}]},
                timeout=30,  # Increase timeout to 30 seconds
            )

            if response.status_code == 200:
                result = response.json()
                print(f"Actual selected: {result['model']}")

                # Show response content length
                if "choices" in result and result["choices"]:
                    content_len = len(result["choices"][0]["message"]["content"])
                    print(f"Response length: {content_len} characters")
            else:
                print(f"Request failed: {response.status_code}")
                print(f"Response body: {response.text}")
        except requests.exceptions.Timeout:
            print("Request timed out - server response took too long")
        except requests.exceptions.ConnectionError:
            print("Connection error - unable to reach the server")
        except Exception as e:
            print(f"Other exception: {e}")

    # Show stats
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("\nSystem stats:")
            print(f"Available engines: {stats['engine_types']}")
            print(f"API status: {'Available' if stats['api_available'] else 'Simulated'}")
    except requests.exceptions.Timeout:
        print("Stats request timed out")
    except Exception as e:
        print(f"Unable to fetch stats: {e}")

    print("\nTest completed")


if __name__ == "__main__":
    test_gateway()
