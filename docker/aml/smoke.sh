#!/usr/bin/env bash
# Contract smoke test against a running service.
# Usage: MNEMOTREE_API_KEY=... docker/aml/smoke.sh https://memory.example.com
set -euo pipefail

base=${1:-http://127.0.0.1:8000}
key=${MNEMOTREE_API_KEY:?set MNEMOTREE_API_KEY}
user="smoke-$(date +%s)-$$"
auth=(-H "Authorization: Bearer ${key}" -H "Content-Type: application/json")

echo "health:  $(curl -fsS "${base}/health")"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${base}/v1/memories/search" \
  -H "Content-Type: application/json" -d '{"query":"x","user_id":"u","top_k":1}')
[[ ${code} == 401 ]] || { echo "expected 401 without auth, got ${code}" >&2; exit 1; }
echo "no-auth: 401"

add=$(curl -fsS -X POST "${base}/v1/memories/add" "${auth[@]}" -d @- <<EOF
{"request_id":"smoke-r1","user_id":"${user}","session_id":"s1","messages":[
 {"role":"user","content":"I adopted a beagle named Pixel last spring.","timestamp":1683554160000},
 {"role":"assistant","content":"Congratulations on adopting Pixel!","timestamp":1683554220000}]}
EOF
)
echo "add:     ${add}"
echo "${add}" | grep -q '"success":true' || { echo "add failed" >&2; exit 1; }

start=$(date +%s)
search=$(curl -fsS -X POST "${base}/v1/memories/search" "${auth[@]}" \
  -d "{\"query\":\"What is the name of my beagle?\",\"user_id\":\"${user}\",\"top_k\":100}")
echo "search:  $(( $(date +%s) - start ))s ${search:0:300}"
echo "${search}" | grep -q 'Pixel' || { echo "search did not return the added memory" >&2; exit 1; }
echo "${search}" | grep -q '"created_at":"2023-05-08T' || { echo "created_at missing" >&2; exit 1; }
echo "OK"
