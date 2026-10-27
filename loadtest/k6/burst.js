import { buildOptions, num, run, setup, summaryFor } from './common.js';

const base = num('BASE_RATE', 2);
const burst = num('BURST_RATE', 50);
const plan = {
  startRate: base,
  stages: [
    { secs: num('BASE_SECS', 60), target: base },
    { secs: 1, target: burst },
    { secs: num('BURST_SECS', 60), target: burst },
    { secs: 1, target: base },
    { secs: num('RECOVERY_SECS', 120), target: base },
  ],
  abort: false,
};

export const options = buildOptions(plan);
export { setup, run as default };
export const handleSummary = summaryFor('burst', plan);
