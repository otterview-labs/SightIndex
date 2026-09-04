/*
 * Screenshots every page at two widths and compares against a baseline.
 *
 * The CSS carries a decade of "add a rule that overrides the last one", and an earlier attempt
 * to tidy it silently moved a selector past an equal-specificity rule in between, changing what
 * won. Nothing catches that but pixels.
 *
 *   node scripts/pixel-diff.mjs baseline   # capture the reference
 *   node scripts/pixel-diff.mjs check      # compare against it
 */
import { mkdirSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import pw from "playwright";

const MODE = process.argv[2] ?? "check";
const BASE = process.env.SIGHTINDEX_URL ?? "http://127.0.0.1:8000";
const OUT = process.env.SHOT_DIR ?? "/tmp/sightindex-pixels";
const WIDTHS = [1440, 1024];
const PAGES = [
  ["monitor", "/"],
  ["search", "/search"],
  ["observations", "/observations"],
  ["faces", "/faces"],
  ["reid", "/reid"],
  ["chat", "/chat-ui"],
];

const dir = join(OUT, MODE === "baseline" ? "baseline" : "current");
mkdirSync(dir, { recursive: true });

const browser = await pw.chromium.launch();
const results = [];
for (const [name, path] of PAGES) {
  for (const width of WIDTHS) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    await page.goto(BASE + path, { waitUntil: "networkidle", timeout: 60000 });
    // Timestamps and live frames differ between runs; freeze what we can and let the rest be
    // compared as a ratio rather than demanding an exact match.
    // Live camera frames and thumbnails change between runs, so a baseline taken hours earlier
    // reported 17-22% on the monitor and search pages when nothing had changed. Hiding the
    // image content keeps every box exactly where it was -- a wrong aspect ratio still moves
    // the layout and still gets caught -- while removing the only thing that varies on its own.
    await page.addStyleTag({
      content: `*{animation:none!important;transition:none!important}
                img,video,canvas{opacity:0!important}`,
    });
    await page.waitForTimeout(2500);
    const file = join(dir, `${name}-${width}.png`);
    await page.screenshot({ path: file, fullPage: true });
    results.push({ name, width, file });
    await page.close();
  }
}
await browser.close();

if (MODE === "baseline") {
  console.log(`baseline captured: ${results.length} shots in ${dir}`);
  process.exit(0);
}

// Compare by decoding both PNGs with the browser itself -- no image library needed.
const compare = await pw.chromium.launch();
const page = await compare.newPage();
let failed = 0;
for (const shot of results) {
  const reference = shot.file.replace("/current/", "/baseline/");
  if (!existsSync(reference)) {
    console.log(`SKIP  ${shot.name}-${shot.width}  (no baseline)`);
    continue;
  }
  const ratio = await page.evaluate(
    async ([a, b]) => {
      const load = (src) =>
        new Promise((resolve) => {
          const image = new Image();
          image.onload = () => resolve(image);
          image.src = src;
        });
      const [one, two] = await Promise.all([load(a), load(b)]);
      const width = Math.min(one.width, two.width);
      const height = Math.min(one.height, two.height);
      const draw = (image) => {
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext("2d");
        context.drawImage(image, 0, 0);
        return context.getImageData(0, 0, width, height).data;
      };
      const [x, y] = [draw(one), draw(two)];
      let differing = 0;
      for (let i = 0; i < x.length; i += 4) {
        if (Math.abs(x[i] - y[i]) > 8 || Math.abs(x[i + 1] - y[i + 1]) > 8 ||
            Math.abs(x[i + 2] - y[i + 2]) > 8) differing++;
      }
      const sizeGap = Math.abs(one.height - two.height) / Math.max(one.height, two.height);
      return { pixels: differing / (x.length / 4), sizeGap };
    },
    [
      `data:image/png;base64,${readFileSync(reference).toString("base64")}`,
      `data:image/png;base64,${readFileSync(shot.file).toString("base64")}`,
    ],
  );
  // Live camera frames and clocks move on their own; a few percent is the floor, not a defect.
  // With the imagery neutralised the floor is genuine noise, so the bar can be much tighter.
  const bad = ratio.pixels > 0.01 || ratio.sizeGap > 0.005;
  if (bad) failed++;
  console.log(
    `${bad ? "DIFF " : "ok   "} ${shot.name}-${shot.width}  ` +
      `变化像素 ${(ratio.pixels * 100).toFixed(2)}%  高度差 ${(ratio.sizeGap * 100).toFixed(2)}%`,
  );
}
await compare.close();
console.log(failed ? `\n${failed} 个页面超出阈值` : "\n全部在阈值内");
process.exit(failed ? 1 : 0);
