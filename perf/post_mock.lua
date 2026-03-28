-- wrk lua script: POST to mock upstream via MaxLLM gateway
-- Measures pure proxy overhead: auth, routing, body translation, metrics, logging
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["Authorization"] = "Bearer sk-maxllm-dev-key"
wrk.body = '{"model":"mock-model","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}]}'
