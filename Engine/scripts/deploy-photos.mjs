// Deploys with Plate's photo bucket switched on, without changing wrangler.jsonc (the
// template stays deployable on accounts without R2). Uncomments the r2_buckets block into
// a throwaway config beside the real one, deploys with it, and removes it.
import { readFileSync, writeFileSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";

const src = readFileSync("wrangler.jsonc", "utf8");
const on = src.replace(/  \/\/ "r2_buckets": \[\n  \/\/ (.*)\n  \/\/ \],/, '  "r2_buckets": [\n  $1\n  ],');
if (on === src) { console.error("No commented r2_buckets block found in wrangler.jsonc."); process.exit(1); }
writeFileSync(".wrangler.photos.jsonc", on);
const r = spawnSync("npx", ["wrangler", "deploy", "--config", ".wrangler.photos.jsonc", ...process.argv.slice(2)], { stdio: "inherit" });
rmSync(".wrangler.photos.jsonc", { force: true });
process.exit(r.status ?? 1);
