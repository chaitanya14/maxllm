-- wrk lua script: bare proxy — no auth header, minimal payload
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}'
