import assert from "node:assert/strict";
import test from "node:test";

import { featureDensity } from "./density.ts";

test("feature density follows milestone thresholds", () => {
  assert.equal(featureDensity(1501), "GOOD");
  assert.equal(featureDensity(1500), "OK");
  assert.equal(featureDensity(500), "OK");
  assert.equal(featureDensity(499), "POOR");
});

