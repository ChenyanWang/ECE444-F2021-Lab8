"""
This script will test the API for lab 8
"""
import time
import requests

api_url = "http://ece444servesentiment37-env.eba-iviaijmx.us-east-1.elasticbeanstalk.com/invocations"
ITERATIONS = 100

# List of tests and their expected sentiments
tests = [
    {"Sentiment" : "FAKE",
    "text" : "The sky is orange! It's a tremendous day hahaha"},
    {"Sentiment" : "FAKE",
    "text" : "I believe that not only can pigs fly, but humans can too. Humans can fly by growing wings and flapping them! What is physics anyways?"},
    {"Sentiment" : "REAL",
    "text" : "Candles provide heat."},
    {"Sentiment" : "REAL",
    "text" : "Breakfast is an important meal of the day. Smoothies make a great breakfast."}
    ]

def run_test(text, iterations):
    total_time = 0
    avg_time = 0
    data = "{\"text\" : \"" +  text + "\"}"
    headers = {"Content-Type" : "application/json"}
    result = None

    for i in range(iterations):
        start = time.time_ns()
        response = requests.request("GET", api_url, data=data, headers=headers)
        response_json = response.json()
        response_json = response_json["Sentiment"]
        total_time += time.time_ns() - start

        if result is None:
            result = response_json
        elif result != response_json:
            result = "INVALID"

    avg_time = total_time/iterations
    return avg_time, result

def run_all_tests():

    for i in range(len(tests)):
        print("Running test %d:\nExpected sentiment: %s\nText: %s" %(i, tests[i]["Sentiment"], tests[i]["text"]))
        avg_time, result = run_test(tests[i]["text"], ITERATIONS)
        print("Average time (ns): %d, Result: %s\n" % (avg_time, result))

if __name__ == "__main__":
    run_all_tests()