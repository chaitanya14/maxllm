-- wrk lua script: streaming POST to MaxLLM gateway
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["Authorization"] = "Bearer sk-maxllm-dev-key"
wrk.body = '{"model":"gemma3:1b","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"stream":true}'
