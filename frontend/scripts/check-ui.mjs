/*
 * Asserts the visual contract of the refinements in src/styles/refine.css, in a real browser.
 *
 * Four stylesheets with eight generations of "override the last rule" sit under every view, and
 * which one wins depends on load order, so a rule can hold on one page and lose on another. The
 * component tests never render a template and the type checker cannot see computed style, so this
 * walks the actual pages and measures the properties the refinements are about.
 *
 *   node scripts/check-ui.mjs
 *   SIGHTINDEX_URL=http://127.0.0.1:5180 node scripts/check-ui.mjs
 *
 * Anything that depends on live data (a trajectory card only exists once a person is selected) is
 * reported as SKIP rather than passed, so the output never claims coverage it did not have.
 */
import pw from "playwright";

const BASE = process.env.SIGHTINDEX_URL ?? "http://127.0.0.1:8000";
const rows = [];
const check = (label, ok, detail = "") => rows.push({ label, ok, detail });
const skip = (label, why) => rows.push({ label, skip: true, detail: why });

const browser = await pw.chromium.launch();

async function open(path, settle = 2000) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, locale: "zh-CN" });
  try {
    await page.goto(BASE + path, { waitUntil: "networkidle", timeout: 30000 });
  } catch {
    // A live MJPEG preview keeps the connection open, so networkidle is not always reachable.
    await page.goto(BASE + path, { waitUntil: "domcontentloaded", timeout: 30000 });
  }
  await page.addStyleTag({ content: "*{animation:none!important;transition:none!important}" });
  await page.waitForTimeout(settle);
  return page;
}

/* ---- accent: one blue, and it is the accent ------------------------------------------------- */

{
  const page = await open("/");
  const theme = await page.evaluate(() => {
    const root = getComputedStyle(document.documentElement);
    return {
      accent: root.getPropertyValue("--accent").trim(),
      strong: root.getPropertyValue("--accent-strong").trim(),
      dark: root.getPropertyValue("--accent-dark").trim(),
    };
  });
  check(
    "主色令牌 --accent-strong 指向蓝色家族",
    theme.strong === theme.dark,
    `--accent ${theme.accent} · --accent-strong ${theme.strong} · --accent-dark ${theme.dark}`,
  );

  const primary = await page.$(".button.primary");
  if (!primary) {
    skip("主按钮悬停为深蓝", "当前页面没有主按钮");
  } else {
    await primary.hover();
    await page.waitForTimeout(150);
    const hover = await primary.evaluate((el) => getComputedStyle(el).backgroundColor);
    // Teal #087267 was what --accent-strong used to hold; hovering must not leave the blue family.
    const [r, g, b] = (hover.match(/\d+/g) ?? []).map(Number);
    check(
      "主按钮悬停为深蓝",
      b > r && b - g > 60,
      `hover background ${hover}`,
    );
  }

  const input = await page.$("input");
  if (input) {
    await input.focus();
    const ring = await input.evaluate((el) => getComputedStyle(el).boxShadow);
    const [r, g, b] = (ring.match(/\d+/g) ?? []).map(Number);
    check("输入框聚焦环属于蓝色家族", ring === "none" || (b > g && b - r > 40), `focus box-shadow ${ring}`);
  }
  await page.close();
}

/* ---- hooks that are not dead: dashed boxes only where you can drop or draw ------------------ */

for (const path of ["/faces", "/search", "/observations", "/chat-ui"]) {
  const page = await open(path, path === "/faces" ? 3500 : 2500);
  const dashed = await page.evaluate(() => {
    const zones = [...document.querySelectorAll(".empty, .message.system, .face-result:empty")];
    const read = (el) => {
      const r = el.getBoundingClientRect();
      const before = getComputedStyle(el, "::before").content;
      return {
        cls: String(el.className) || el.tagName,
        // Only rendered boxes count: getComputedStyle still reports the border of a
        // display:none element, so the flags have to come with a visibility test.
        rendered: r.width > 0 && r.height > 0,
        dashed: getComputedStyle(el).borderStyle.includes("dashed"),
        hint: before && before !== "none" ? before.replace(/^"|"$/g, "") : "",
        text: (el.textContent || "").trim().slice(0, 24),
      };
    };
    const rows = zones.map(read);
    const dropTargets = [...document.querySelectorAll(".file-drop")];
    return {
      rows,
      dropDashed: dropTargets.filter((el) => getComputedStyle(el).borderStyle.includes("dashed")).length,
      dropCount: dropTargets.length,
    };
  });
  const dashedBoxes = dashed.rows.filter((z) => z.rendered && z.dashed);
  check(
    `${path} 空状态不再用虚线框表达"未完成"`,
    dashedBoxes.length === 0,
    dashedBoxes.length ? `仍有虚线：${dashedBoxes.map((z) => z.cls).join(", ")}` : `检查了 ${dashed.rows.length} 个空状态`,
  );
  // A placeholder that has something to say must actually render -- display:none on one of these
  // silently deletes guidance, which is how the 人脸库 hint went missing during the first pass.
  const silenced = dashed.rows.filter((z) => !z.rendered && (z.hint || z.text));
  check(
    `${path} 有提示文字的空状态没有被隐藏`,
    silenced.length === 0,
    silenced.length ? `被隐藏：${silenced.map((z) => `${z.cls}("${z.hint || z.text}")`).join(", ")}` : `${dashed.rows.filter((z) => z.rendered).length} 个空状态可见`,
  );
  if (dashed.dropCount) {
    check(
      `${path} 拖拽区保留虚线（正对照）`,
      dashed.dropDashed === dashed.dropCount,
      `${dashed.dropDashed}/${dashed.dropCount} 个 .file-drop 仍为虚线`,
    );
  }
  await page.close();
}

/* ---- empty status chip must not render as a broken control ---------------------------------- */

{
  const page = await open("/observations", 2500);
  const chips = await page.evaluate(() =>
    [...document.querySelectorAll(".table-status")].map((el) => ({
      text: (el.textContent || "").trim(),
      display: getComputedStyle(el).display,
    })),
  );
  const visible = chips.filter((c) => c.text === "" && c.display !== "none");
  check("空的状态药丸不渲染", visible.length === 0, `${chips.length} 个状态容器，空的 ${visible.length} 个仍在显示`);
  await page.close();
}

/* ---- the search prefix is an icon, not a glyph ---------------------------------------------- */

{
  const page = await open("/search", 3000);
  const prefix = await page.evaluate(() => {
    const el = document.querySelector(".semantic-prefix");
    if (!el) return null;
    return {
      svg: el.querySelectorAll("svg").length,
      text: (el.textContent || "").trim(),
      glyphInPage: document.body.innerText.includes("⌕"),
    };
  });
  if (!prefix) {
    skip("检索框前缀是真图标", "找不到 .semantic-prefix");
  } else {
    check("检索框前缀是真图标", prefix.svg === 1 && !prefix.glyphInPage, `svg ${prefix.svg} 个，页面残留字形 ${prefix.glyphInPage}`);
  }
  await page.close();
}

/* ---- datetime fields speak the document's language ------------------------------------------ */

for (const path of ["/search", "/observations"]) {
  const page = await open(path, 3000);
  const before = await page.evaluate(() =>
    [...document.querySelectorAll("input[type='datetime-local']")].map((el) => {
      const hint = el.parentElement?.querySelector(".datetime-field-hint");
      return {
        hasHint: Boolean(hint),
        hintText: hint ? (hint.textContent || "").trim() : "",
        hintShown: hint ? getComputedStyle(hint).display !== "none" && getComputedStyle(hint).opacity !== "0" : false,
      };
    }),
  );
  const empty = before.every((f) => f.hasHint && f.hintText && f.hintShown);
  check(
    `${path} 日期控件显示中文占位`,
    before.length > 0 && empty,
    `${before.length} 个控件；${JSON.stringify(before[0] ?? null)}`,
  );

  const first = await page.$("input[type='datetime-local']");
  if (first) {
    await first.focus();
    const focused = await first.evaluate((el) => {
      const hint = el.parentElement?.querySelector(".datetime-field-hint");
      return hint ? getComputedStyle(hint).opacity : null;
    });
    check(`${path} 聚焦日期控件时让位给原生编辑器`, focused === "0", `hint opacity on focus ${focused}`);

    await first.fill("2026-10-16T10:00");
    await page.waitForTimeout(120);
    const after = await first.evaluate((el) => el.parentElement?.querySelectorAll(".datetime-field-hint").length ?? 0);
    check(`${path} 有值时中文占位消失`, after === 0, `剩余 hint 节点 ${after}`);
  }
  await page.close();
}

/* ---- person-crop tiles fill their box -------------------------------------------------------- */

{
  const page = await open("/", 6000);
  const tiles = await page.evaluate(() => {
    const items = [...document.querySelectorAll(".media-grid-crops .media-item")].slice(0, 8);
    return items.map((el) => {
      const img = el.querySelector("img");
      const meta = el.querySelector(".media-meta");
      const strong = meta?.querySelector("strong");
      const ids = meta?.querySelector(".media-ids");
      const r = img.getBoundingClientRect();
      // How much of the box the crop actually covers once object-fit has been applied. The
      // previous 4:3 contain box filled 30% of its width with a portrait crop.
      const natural = img.naturalWidth / img.naturalHeight;
      const contentW = Math.min(r.width, r.height * natural);
      return {
        fit: getComputedStyle(img).objectFit,
        box: [Math.round(r.width), Math.round(r.height)],
        fill: contentW / r.width,
        firstLine: (strong?.textContent || "").trim(),
        firstLineWeight: strong ? getComputedStyle(strong).fontWeight : null,
        idsText: (ids?.textContent || "").trim(),
        idsSize: ids ? parseFloat(getComputedStyle(ids).fontSize) : null,
        strongSize: strong ? parseFloat(getComputedStyle(strong).fontSize) : null,
      };
    });
  });
  if (!tiles.length) {
    skip("人物裁剪卡片填满缩略框", "监控页当前没有裁剪卡片");
  } else {
    // Real crops run 0.34-0.58 wide over tall, so no fixed box fills all of them: in a 1:2 box
    // the tallest leaves ~32% of the width as matte. The old 4:3 contain box filled 31% for even
    // a mid-range crop, so 0.65 separates "matched to the subject" from "letterboxed".
    const letterboxed = tiles.filter((t) => t.fit !== "contain" || t.fill < 0.65);
    check(
      "人物裁剪卡片填满缩略框",
      letterboxed.length === 0,
      `${tiles.length} 张卡片，${tiles.map((t) => `${t.box.join("x")} 填充 ${(t.fill * 100).toFixed(0)}%`).join("，")}`,
    );
    const portrait = tiles.every((t) => t.box[1] / t.box[0] >= 1.6);
    check("裁剪缩略框与人物裁剪同为竖长条", portrait, tiles.map((t) => (t.box[1] / t.box[0]).toFixed(2)).join(" "));
    const times = tiles.filter((t) => /\d{1,2}:\d{2}:\d{2}/.test(t.firstLine));
    check("卡片首行是拍摄时间而不是 id", times.length === tiles.length, `首行：${tiles[0].firstLine}`);
    check(
      "裁剪 id 降级为次要小字",
      tiles.every((t) => t.idsSize !== null && t.strongSize !== null && t.idsSize < t.strongSize),
      `strong ${tiles[0].strongSize}px / ids ${tiles[0].idsSize}px`,
    );
  }
  await page.close();
}

/* ---- stopped sources stop eating the rail ---------------------------------------------------- */

{
  const page = await open("/", 6000);
  const rail = await page.evaluate(() => {
    const groups = [...document.querySelectorAll(".stream-group")].map((el) => ({
      label: (el.querySelector("summary")?.textContent || "").trim(),
      open: el.open,
      items: el.querySelectorAll(".stream-item").length,
    }));
    const all = document.querySelectorAll(".stream-list .stream-item").length;
    const visible = [...document.querySelectorAll(".stream-list .stream-item")].filter((el) => {
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    }).length;
    const list = document.querySelector(".stream-list");
    return {
      groups,
      all,
      visible,
      height: Math.round(list?.getBoundingClientRect().height ?? 0),
      overflow: list ? list.scrollHeight - list.clientHeight : 0,
    };
  });
  if (!rail.groups.length) {
    skip("视频源按运行中/已停止分组", "没有 .stream-group（可能是旧构建）");
  } else {
    const running = rail.groups.find((g) => g.label.startsWith("运行中"));
    const stopped = rail.groups.find((g) => g.label.startsWith("已停止"));
    check("视频源按运行中/已停止分组", Boolean(running) && Boolean(stopped), rail.groups.map((g) => `${g.label}${g.open ? "(展开)" : "(收起)"}`).join(" · "));
    check("运行中默认展开", Boolean(running?.open), `运行中 open=${running?.open}`);
    check("已停止默认收起", stopped ? !stopped.open : true, `已停止 open=${stopped?.open}`);
    check(
      "右侧栏不再被死流占满",
      rail.height <= 420,
      `列表高 ${rail.height}px，内部溢出 ${rail.overflow}px，共 ${rail.all} 条源`,
    );
  }
  await page.close();
}

/* ---- 人脸库 trajectory card: face box on the face, chip clipped, no sideways void ----------- */

{
  const page = await open("/faces", 4000);
  // Only the read-only 轨迹 action; the card also carries destructive buttons.
  const person = page.locator(".person-item button", { hasText: "轨迹" }).first();
  if (await person.count()) {
    await person.click();
    await page.waitForTimeout(6000);
  }
  const card = await page.evaluate(() => {
    const list = document.querySelector(".trajectory-list");
    const frame = document.querySelector(".trajectory-image-frame");
    const zoom = document.querySelector(".face-zoom-frame");
    const bbox = frame?.querySelector(".face-bbox");
    const inside = (() => {
      if (!bbox || !frame) return null;
      const a = bbox.getBoundingClientRect();
      const b = frame.getBoundingClientRect();
      return a.left >= b.left - 2 && a.right <= b.right + 2 && a.top >= b.top - 2 && a.bottom <= b.bottom + 2;
    })();
    return {
      hasList: Boolean(list),
      track: list ? getComputedStyle(list).gridTemplateColumns : null,
      scrollVoid: list ? list.scrollWidth - list.clientWidth : 0,
      framePos: frame ? getComputedStyle(frame).position : null,
      bboxInside: inside,
      bboxSize: bbox ? [Math.round(bbox.getBoundingClientRect().width), Math.round(bbox.getBoundingClientRect().height)] : null,
      zoom: zoom
        ? {
            pos: getComputedStyle(zoom).position,
            overflow: getComputedStyle(zoom).overflow,
            img: getComputedStyle(zoom.querySelector("img")).position,
            box: [Math.round(zoom.getBoundingClientRect().width), Math.round(zoom.getBoundingClientRect().height)],
          }
        : null,
    };
  });
  if (!card.hasList) {
    skip("轨迹卡片布局", "人脸库没有 .trajectory-list");
  } else {
    check(
      "轨迹列表不横向溢出",
      card.scrollVoid <= 2,
      `scrollWidth-clientWidth = ${card.scrollVoid}px，列为 ${card.track}`,
    );
    if (card.framePos === null) {
      skip("人脸框落在人脸所在的图片内", "没有轨迹卡片（需要选择一位人员）");
    } else {
      check(
        "人脸框落在人脸所在的图片内",
        card.framePos === "relative" && card.bboxInside !== false,
        `frame position ${card.framePos}，bbox 是否在图片内 ${card.bboxInside}，bbox ${card.bboxSize?.join("x")}`,
      );
    }
    if (card.zoom) {
      check(
        "人脸放大切片被裁剪且尺寸正常",
        card.zoom.pos === "relative" &&
          card.zoom.overflow === "hidden" &&
          card.zoom.img === "absolute" &&
          card.zoom.box[0] <= 200,
        `frame ${card.zoom.pos}/${card.zoom.overflow}，img ${card.zoom.img}，盒子 ${card.zoom.box.join("x")}`,
      );
    }
  }
  await page.close();
}

await browser.close();

/* ---- report ---------------------------------------------------------------------------------- */

let failed = 0;
for (const row of rows) {
  if (row.skip) {
    console.log(`SKIP  ${row.label}  (${row.detail})`);
    continue;
  }
  if (!row.ok) failed++;
  console.log(`${row.ok ? "ok   " : "FAIL "} ${row.label}${row.detail ? `  —  ${row.detail}` : ""}`);
}
const passed = rows.filter((r) => r.ok).length;
const skipped = rows.filter((r) => r.skip).length;
console.log(
  failed
    ? `\n${failed} 项不通过（通过 ${passed}，跳过 ${skipped}）`
    : `\n全部通过：${passed} 项${skipped ? `，跳过 ${skipped} 项` : ""}`,
);
process.exit(failed ? 1 : 0);
