import { buildOptions, num, run, setup, summaryFor } from './common.js';

const rate = num('RATE', 1);
const plan = { startRate: rate, stages: [{ secs: num('DURATION_SECS', 60), target: rate }], abort: false };

export const options = buildOptions(plan);
export { setup, run as default };
export const handleSummary = summaryFor('smoke', plan);
