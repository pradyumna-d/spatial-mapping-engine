export type FeatureDensity = "GOOD" | "OK" | "POOR";

export function featureDensity(count: number): FeatureDensity {
  if (count > 1500) return "GOOD";
  if (count >= 500) return "OK";
  return "POOR";
}

