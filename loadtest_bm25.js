import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 25,
  duration: '120s',
  thresholds: {
    http_req_failed: ['rate<0.01'],
  },
};

const baseUrl = __ENV.BASE_URL;

const payload = JSON.stringify({
  query: 'python list comprehension',
  candidates: [
    { id: 'c1', text: 'weather forecast for tomorrow in chicago' },
    { id: 'c2', text: 'python list comprehension tutorial and examples' },
    { id: 'c3', text: 'paris is the capital city of france' },
    { id: 'c4', text: 'generator expressions are similar to list comprehensions' }
  ]
});

export default function () {
  const res = http.post(`${baseUrl}/rerank_bm25`, payload, {
    headers: { 'Content-Type': 'application/json' },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
