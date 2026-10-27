import http from 'k6/http';
import exec from 'k6/execution';
import { sleep } from 'k6';
import { Counter, Trend } from 'k6/metrics';

const POLL_EVERY_SECS = 2;
const POLL_LIMIT_SECS = 300;

export const BASE_URL = __ENV.BASE_URL || 'https://api.danielemartinez.com';
const REUSE_SHARE = Number(__ENV.REUSE_SHARE || 0.2);
const OUT = __ENV.OUT || 'query-ids.txt';

const queriesNew = new Counter('queries_new');
const queriesRepeat = new Counter('queries_repeat');
const queriesDone = new Counter('queries_done');
const queriesFailed = new Counter('queries_failed');
const queriesUnfinished = new Counter('queries_unfinished');
const queryDuration = new Trend('query_duration', true);

export function num(name, fallback) {
  return __ENV[name] === undefined ? fallback : Number(__ENV[name]);
}

// plan: { startRate, stages: [{ secs, target }], abort }. Rates are iterations per second.
export function buildOptions(plan) {
  const peak = Math.max(plan.startRate, ...plan.stages.map((stage) => stage.target));
  const maxVUs = num('MAX_VUS', Math.ceil(peak * (POLL_LIMIT_SECS + 10)) + 10);
  const preAllocatedVUs = Math.min(maxVUs, num('PRE_ALLOCATED_VUS', Math.ceil(peak * 30) + 10));
  const stop = plan.abort
    ? { abortOnFail: true, delayAbortEval: `${num('ABORT_DELAY_SECS', 10)}s` }
    : {};
  return {
    summaryTrendStats: ['avg', 'p(95)', 'p(99)', 'max'],
    scenarios: {
      load: {
        executor: 'ramping-arrival-rate',
        startRate: plan.startRate,
        timeUnit: '1s',
        preAllocatedVUs,
        maxVUs,
        stages: plan.stages.map((stage) => ({ duration: `${stage.secs}s`, target: stage.target })),
        gracefulStop: `${POLL_LIMIT_SECS + 30}s`,
      },
    },
    thresholds: {
      'http_req_duration{op:post}': [{ threshold: 'p(99)<500', ...stop }],
      http_req_failed: [{ threshold: 'rate<0.01', ...stop }],
      queries_unfinished: [{ threshold: 'count==0', ...stop }],
      // These two entries only make k6 report each request kind on its own.
      'http_reqs{op:post}': ['count>=0'],
      'http_reqs{op:get}': ['count>=0'],
      'http_req_duration{op:get}': ['p(99)>=0'],
    },
  };
}

// A mixed 32-bit hash, so iteration i always makes the same choices.
function hash(i, salt) {
  let h = (i ^ Math.imul(salt, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function isRepeat(i) {
  return i > 0 && hash(i, 1) < REUSE_SHARE;
}

// A repeat takes the text of a random earlier iteration that sent a new text.
function textIndex(i) {
  let index = i;
  while (isRepeat(index)) {
    index = Math.floor(hash(index, 2) * index);
  }
  return index;
}

export function setup() {
  return { runId: __ENV.RUN_ID || Date.now().toString(36) };
}

export function run(data) {
  const i = exec.scenario.iterationInTest;
  const index = textIndex(i);
  (index === i ? queriesNew : queriesRepeat).add(1);

  const headers = { Authorization: `Bearer ${__ENV.API_KEY}`, 'Content-Type': 'application/json' };
  const post = http.post(
    `${BASE_URL}/queries`,
    JSON.stringify({ text: `loadtest ${data.runId} ${index}` }),
    { headers, tags: { op: 'post', name: 'POST /queries' } },
  );
  if (post.status !== 202) {
    return;
  }
  const queryId = post.json('query_id');

  const started = Date.now();
  let status = 'unfinished';
  while (Date.now() - started < POLL_LIMIT_SECS * 1000) {
    sleep(POLL_EVERY_SECS);
    const get = http.get(`${BASE_URL}/queries/${queryId}`, {
      headers,
      tags: { op: 'get', name: 'GET /queries/{id}' },
    });
    if (get.status === 200) {
      const current = get.json('status');
      if (current === 'done' || current === 'failed') {
        status = current;
        break;
      }
    }
  }

  if (status === 'done') {
    queriesDone.add(1);
    queryDuration.add(Date.now() - started);
  } else if (status === 'failed') {
    queriesFailed.add(1);
  } else {
    queriesUnfinished.add(1);
  }
  console.log(`qid=${queryId} status=${status}`);
}

function rateAt(plan, elapsedSecs) {
  let from = plan.startRate;
  let start = 0;
  for (const stage of plan.stages) {
    if (elapsedSecs <= start + stage.secs) {
      return from + ((stage.target - from) * (elapsedSecs - start)) / stage.secs;
    }
    from = stage.target;
    start += stage.secs;
  }
  return from;
}

function value(data, metric, key) {
  const found = data.metrics[metric];
  return found && found.values[key] !== undefined ? found.values[key] : 0;
}

export function summaryFor(name, plan) {
  return function handleSummary(data) {
    const elapsedSecs = data.state.testRunDurationMs / 1000;
    const failed = Object.keys(data.metrics).filter((metric) =>
      (data.metrics[metric].thresholds ? Object.values(data.metrics[metric].thresholds) : []).some(
        (threshold) => threshold.ok === false,
      ),
    );
    const fresh = value(data, 'queries_new', 'count');
    const repeat = value(data, 'queries_repeat', 'count');
    const report = {
      scenario: name,
      base_url: BASE_URL,
      elapsed_secs: Math.round(elapsedSecs),
      rate_at_end_per_sec: Number(rateAt(plan, elapsedSecs).toFixed(2)),
      break_point: failed.length > 0,
      failed_thresholds: failed,
      queries_sent: fresh + repeat,
      query_reuse_rate: fresh + repeat > 0 ? Number((repeat / (fresh + repeat)).toFixed(3)) : 0,
      post_per_sec: Number(value(data, 'http_reqs{op:post}', 'rate').toFixed(2)),
      get_per_sec: Number(value(data, 'http_reqs{op:get}', 'rate').toFixed(2)),
      post_p99_ms: Number(value(data, 'http_req_duration{op:post}', 'p(99)').toFixed(1)),
      get_p99_ms: Number(value(data, 'http_req_duration{op:get}', 'p(99)').toFixed(1)),
      error_rate: Number(value(data, 'http_req_failed', 'rate').toFixed(4)),
      queries_done: value(data, 'queries_done', 'count'),
      queries_failed: value(data, 'queries_failed', 'count'),
      queries_unfinished: value(data, 'queries_unfinished', 'count'),
      query_duration_p95_ms: Number(value(data, 'query_duration', 'p(95)').toFixed(0)),
      dropped_iterations: value(data, 'dropped_iterations', 'count'),
      vus_max: value(data, 'vus_max', 'max'),
    };
    const text = JSON.stringify(report, null, 2);
    return {
      stdout: `\n${text}\n`,
      [`${OUT}.summary.json`]: text,
    };
  };
}
