#!/usr/bin/env python3
"""
Flask Proxy Server for Gemini - NO TOKEN LIMITS
"""

import os
import re
import time
import requests
import json
from flask import Flask, request, jsonify, Response
from collections import defaultdict, deque

app = Flask(__name__)

KEYS_FILE_PATH = "/Users/williamwhite/myapikeys/old/apikeys"

RATE_LIMITS = {
    "gemini-2.5-flash-preview-05-20": {"requests_per_minute": 1000, "requests_per_day": 10000},
    "gemini-2.5-flash-preview-04-17": {"requests_per_minute": 1000, "requests_per_day": 10000},
    "gemini-1.5-pro":                {"requests_per_minute":   50, "requests_per_day":  1000},
    "gemini-1.5-flash":             {"requests_per_minute": 1000, "requests_per_day": 10000},
    "gemini-1.0-pro":               {"requests_per_minute":   60, "requests_per_day":  1500}
}

API_KEYS = []
key_usage = defaultdict(lambda: {
    "requests_today": 0,
    "requests_this_minute": deque(),
    "last_reset": time.time()
})

def load_api_keys_from_file():
    global API_KEYS
    print(f"[DEBUG] Loading API keys from {KEYS_FILE_PATH}")
    if not os.path.exists(KEYS_FILE_PATH):
        print(f"[ERROR] Keys file not found: {KEYS_FILE_PATH}")
        return False
    content = open(KEYS_FILE_PATH, 'r', encoding='utf-8').read()
    matches = re.findall(r"(AIza[0-9A-Za-z_-]{35})", content)
    if matches:
        API_KEYS = matches
        print(f"[DEBUG] Extracted {len(API_KEYS)} keys via regex")
        return True
    print(f"[ERROR] No API keys found via regex")
    return False

def validate_api_keys():
    if not API_KEYS:
        print("[ERROR] No API keys to validate")
        return False
    print("[DEBUG] Validating first few API keys...")
    valid = []
    for key in API_KEYS[:5]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        try:
            r = requests.get(url, timeout=5)
            print(f"[DEBUG] Key …{key[-8:]} status {r.status_code}")
            if r.status_code == 200:
                valid.append(key)
        except Exception as e:
            print(f"[DEBUG] Key …{key[-8:]} validation error: {e}")
    API_KEYS[:] = valid + [k for k in API_KEYS if k not in valid]
    print(f"[DEBUG] {len(valid)}/{min(5,len(API_KEYS))} keys valid; {len(API_KEYS)} total kept")
    return bool(API_KEYS)

def get_available_key(model):
    now = time.time()
    limits = RATE_LIMITS.get(model, RATE_LIMITS["gemini-2.5-flash-preview-05-20"])
    for key in API_KEYS:
        usage = key_usage[key]
        if now - usage["last_reset"] >= 86400:
            usage["requests_today"] = 0
            usage["last_reset"] = now
        while usage["requests_this_minute"] and now - usage["requests_this_minute"][0] >= 60:
            usage["requests_this_minute"].popleft()
        if (usage["requests_today"] < limits["requests_per_day"]
            and len(usage["requests_this_minute"]) < limits["requests_per_minute"]):
            usage["requests_today"] += 1
            usage["requests_this_minute"].append(now)
            return key
    return None

@app.route('/v1beta/models/<model>:generateContent', methods=['POST'])
def generate_content(model):
    print(f"\n[INCOMING] generateContent for model={model}")
    try:
        payload = request.get_json(force=True)
        
        # REMOVE ANY TOKEN LIMITS - Let Gemini use its full capacity
        if "generationConfig" not in payload:
            payload["generationConfig"] = {}
        
        # Set maxOutputTokens to maximum if not specified or too low
        if "maxOutputTokens" not in payload["generationConfig"] or payload["generationConfig"]["maxOutputTokens"] < 1000000:
            payload["generationConfig"]["maxOutputTokens"] = 1048576  # 1M tokens
            
        print("[INCOMING] Payload (with maxOutputTokens set to 1M):", json.dumps(payload))
    except Exception as e:
        print("[ERROR] Invalid JSON body:", e)
        return jsonify(error="Invalid JSON"), 400

    key = get_available_key(model)
    if not key:
        print("[WARN] No available API key → 429 back to client")
        return jsonify(error="Rate limit exceeded for all keys"), 429
    print(f"[DEBUG] Selected key …{key[-8:]}")

    def call_upstream(api_key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        print(f"[UPSTREAM] POST {url}")
        print("[UPSTREAM] Headers:", headers)
        print("[UPSTREAM] Body:", json.dumps(payload))
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        print(f"[UPSTREAM] Status: {resp.status_code}")
        print("[UPSTREAM] Response length:", len(resp.text))
        return resp

    resp = call_upstream(key)
    if resp.status_code == 429:
        print(f"[WARN] Quota hit for key …{key[-8:]}, marking as exhausted and rotating")
        key_usage[key]["requests_today"] = RATE_LIMITS[model]["requests_per_day"]
        new_key = get_available_key(model)
        if new_key:
            print(f"[DEBUG] Rotated to key …{new_key[-8:]}")
            resp = call_upstream(new_key)
        else:
            print("[WARN] No keys left after rotation → 429")
            return jsonify(error="Rate limit exceeded for all keys"), 429

    try:
        result = resp.json()
    except Exception as e:
        print("[ERROR] Failed to parse JSON from upstream:", e)
        result = {"error": "Invalid JSON from upstream"}
    return jsonify(result), resp.status_code

@app.route('/v1beta/models/<model>:streamGenerateContent', methods=['POST'])
def stream_generate_content(model):
    print(f"\n[INCOMING] streamGenerateContent for model={model}")
    try:
        payload = request.get_json(force=True)
        
        # REMOVE ANY TOKEN LIMITS for streaming too
        if "generationConfig" not in payload:
            payload["generationConfig"] = {}
        payload["generationConfig"]["maxOutputTokens"] = 1048576  # 1M tokens
        
        print("[INCOMING] Payload (with maxOutputTokens set to 1M):", json.dumps(payload))
    except Exception as e:
        print("[ERROR] Invalid JSON body:", e)
        return jsonify(error="Invalid JSON"), 400

    key = get_available_key(model)
    if not key:
        print("[WARN] No available API key → 429 back to client")
        return jsonify(error="Rate limit exceeded for all keys"), 429
    print(f"[DEBUG] Selected key …{key[-8:]}")

    def start_streaming(api_key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
        params = dict(request.args, key=api_key)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        print(f"[UPSTREAM] POST {url}?{params}")
        print("[UPSTREAM] Headers:", headers)
        print("[UPSTREAM] Body:", json.dumps(payload))
        return requests.post(url, headers=headers, json=payload, params=params, timeout=120, stream=True)

    resp = start_streaming(key)
    if resp.status_code == 429:
        print(f"[WARN] Quota hit for key …{key[-8:]}, rotating")
        key_usage[key]["requests_today"] = RATE_LIMITS[model]["requests_per_day"]
        new_key = get_available_key(model)
        if new_key:
            print(f"[DEBUG] Rotated to key …{new_key[-8:]}")
            resp = start_streaming(new_key)
        else:
            print("[WARN] No keys left after rotation → 429")
            return jsonify(error="Rate limit exceeded for all keys"), 429

    if resp.status_code != 200:
        print(f"[ERROR] Upstream error {resp.status_code}: {resp.text}")
        return jsonify(error=f"API error: {resp.status_code}", details=resp.text), resp.status_code

    def generate():
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                yield chunk
    return Response(generate(), mimetype=resp.headers.get('content-type', 'text/plain'))

@app.route('/reload-keys', methods=['POST'])
def reload_keys():
    print("\n[ADMIN] reload-keys called")
    ok = load_api_keys_from_file()
    if ok:
        validate_api_keys()
        return jsonify(status="success", keys_loaded=len(API_KEYS))
    else:
        return jsonify(status="error", message="Failed to load keys"), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin','*')
    response.headers.add('Access-Control-Allow-Headers','Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods','GET,POST,OPTIONS')
    return response

def main():
    print("🚀 Starting Gemini Proxy - NO TOKEN LIMITS")
    if not load_api_keys_from_file() or not validate_api_keys():
        print("❌ Startup failed: could not load/validate API keys")
        return
    print(f"✅ Server ready with {len(API_KEYS)} keys - Using FULL 1M token capacity")
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

if __name__ == '__main__':
    main()