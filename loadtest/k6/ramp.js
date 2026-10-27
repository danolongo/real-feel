import { buildOptions, num, run, setup, summaryFor } from './common.js';

const start = num('START_RATE', 1);
const max = num('MAX_RATE', 100);
const steps = num('STEPS', 10);
const stepSecs = num('STEP_SECS', 60);
const plan = { startRate: start, stages: [], abort: true };
for (let step = 1; step <= steps; step++) {
  plan.stages.push({ secs: stepSecs, target: Math.round(start + ((max - start) * step) / steps) });
}

export const options = buildOptions(plan);
export { setup, run as default };
export const handleSummary = summaryFor('ramp', plan);
