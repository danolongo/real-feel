import { buildOptions, num, run, setup, summaryFor } from './common.js';

const rate = num('RATE', 5);
const plan = { startRate: rate, stages: [{ secs: num('DURATION_SECS', 3600), target: rate }], abort: false };

export const options = buildOptions(plan);
export { setup, run as default };
export const handleSummary = summaryFor('soak', plan);
