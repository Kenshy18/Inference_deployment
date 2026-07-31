import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { _electron as electron } from "playwright";

const guiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(guiRoot, "..");
const video =
  process.env.MASK_STUDIO_TEST_VIDEO ??
  path.join(repositoryRoot, "data", "codino_trt_3min_simple150_input.mp4");
const outputRoot =
  process.env.MASK_STUDIO_TEST_OUTPUT ??
  path.join(repositoryRoot, "output", "gui_smoke");
const artifactRoot =
  process.env.MASK_STUDIO_TEST_ARTIFACTS ??
  path.join(repositoryRoot, "output", "gui_qa_latest");

if (!fs.existsSync(video)) {
  throw new Error(`GUI smoke input does not exist: ${video}`);
}
fs.mkdirSync(outputRoot, { recursive: true });
fs.mkdirSync(artifactRoot, { recursive: true });
const collisionRoot = path.join(outputRoot, `collision-${process.pid}`);
const videoTitle = path.parse(video).name;
const occupiedOutput = path.join(collisionRoot, videoTitle);
fs.mkdirSync(occupiedOutput, { recursive: true });
fs.writeFileSync(path.join(occupiedOutput, "run_manifest.json"), "{}\n");
const userData = fs.mkdtempSync(path.join(os.tmpdir(), "mask-studio-gui-"));
const runtimeLibraries = path.join(
  guiRoot,
  ".runtime-libs",
  "usr",
  "lib",
  "x86_64-linux-gnu",
);

const errors = [];
let app;
try {
  app = await electron.launch({
    args: [
      ".",
      "--software-rendering",
      `--automation-video=${video}`,
      `--automation-output=${collisionRoot}`,
      `--user-data-dir=${userData}`,
    ],
    cwd: guiRoot,
    env: {
      ...process.env,
      MASK_STUDIO_AUTOMATION_NO_EXTERNAL: "1",
      LD_LIBRARY_PATH: [
        runtimeLibraries,
        process.env.LD_LIBRARY_PATH,
      ]
        .filter(Boolean)
        .join(":"),
    },
  });
  const window = await app.firstWindow();
  window.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  window.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  await window.waitForLoadState("domcontentloaded");
  for (const label of ["性器推論", "顔推論", "後処理", "オーバーレイ"]) {
    await window
      .locator(".phase-progress")
      .filter({ hasText: label })
      .waitFor({ state: "visible" });
  }
  for (const label of [
    "入力",
    "性器推論",
    "顔推論",
    "後処理",
    "オーバーレイ",
    "出力",
  ]) {
    await window
      .locator(".flow__node")
      .filter({ hasText: label })
      .waitFor({ state: "visible" });
  }
  if (
    (await window
      .locator(".metric > span")
      .filter({ hasText: /^compute$/i })
      .count()) !== 0
  ) {
    throw new Error("Monitor still exposed the obsolete Compute metric");
  }
  await window
    .locator(".metric")
    .filter({ hasText: /^faces/i })
    .waitFor({ state: "visible" });

  const modeRow = window.locator(".row").filter({ hasText: "処理モード" });
  await modeRow.getByRole("button", { name: "性器", exact: true }).click();
  if (
    (await window
      .locator(".flow__node")
      .filter({ hasText: "顔推論" })
      .count()) !== 0
  ) {
    throw new Error("Face flow node remained visible in segmentation-only mode");
  }
  await modeRow.getByRole("button", { name: "顔", exact: true }).click();
  for (const label of ["性器推論", "後処理"]) {
    if (
      (await window
        .locator(".flow__node")
        .filter({ hasText: label })
        .count()) !== 0
    ) {
      throw new Error(`${label} flow node remained visible in face-only mode`);
    }
  }
  await modeRow.getByRole("button", { name: "両方", exact: true }).click();
  await window
    .locator(".flow__node")
    .filter({ hasText: "顔推論" })
    .waitFor({ state: "visible" });
  await window.screenshot({
    path: path.join(artifactRoot, "01-startup.png"),
    fullPage: true,
  });

  const source = window.locator(".panel").filter({ hasText: "Source" });
  await source.getByRole("button", { name: "参照" }).first().click();
  await source.getByRole("button", { name: /追加/ }).click();
  const queueItem = window
    .locator(".qitem")
    .filter({ hasText: path.parse(video).name });
  await queueItem.waitFor();
  await queueItem.getByText(/1920×1080/).waitFor();
  await window.getByText("予測所要", { exact: true }).waitFor();
  const etaValues = window.locator(".eta--expanded b");
  if ((await etaValues.count()) !== 2) {
    throw new Error("Elapsed/predicted-duration readout is incomplete");
  }
  if ((await etaValues.nth(1).innerText()).includes("--")) {
    throw new Error("Predicted duration was not calculated");
  }
  for (const obsolete of ["実測速度で補正", "このPCの基準速度", "完了予定"]) {
    if ((await window.getByText(obsolete, { exact: true }).count()) !== 0) {
      throw new Error(`Monitor still exposed obsolete label: ${obsolete}`);
    }
  }

  const dryRun = window.getByRole("button", { name: /Dry Run/ });
  await dryRun.waitFor({ state: "visible" });
  if (await dryRun.isDisabled()) {
    throw new Error("Dry Run remained disabled after selecting input and output");
  }

  const simpleClassRules = window.locator(
    ".simple-policy-editor__rule",
  );
  if ((await simpleClassRules.count()) !== 3) {
    throw new Error("Simple settings did not expose all three class rules");
  }
  if (
    (await window
      .locator(".panel--inspector .row__label")
      .filter({ hasText: "カット検出" })
      .count()) !== 0
  ) {
    throw new Error("Simple settings still exposed the cut detection toggle");
  }
  const simpleMaleRule = simpleClassRules.filter({ hasText: "男性器" });
  await simpleMaleRule.getByRole("button", { name: "楕円", exact: true }).click();
  await simpleMaleRule
    .getByRole("button", { name: "ポリゴン", exact: true })
    .click();
  const simpleMaleKeyframe = simpleMaleRule.locator("input[type=number]");
  await simpleMaleKeyframe.fill("4");
  await simpleMaleKeyframe.fill("2");

  await window.getByRole("button", { name: "詳細", exact: true }).click();
  await window.waitForTimeout(50);
  if (
    (await window.locator(".panel--inspector [data-app-tooltip]").count()) !== 0
  ) {
    throw new Error("Inspector still exposes hover tooltips");
  }
  const faceModelRow = window.locator(".row").filter({ hasText: "顔モデル" });
  await faceModelRow.locator(".select__btn").click();
  await window.getByRole("option", { name: "Face V1" }).click();
  const faceEngineRow = window
    .locator(".row")
    .filter({ hasText: "顔推論エンジン" });
  await faceEngineRow.getByText("低速（安定）").waitFor();

  // Focusing a control deep in the scrolled Inspector used to scroll the
  // document root as well, moving the fixed-height app out of the viewport
  // and leaving a black window. Exercise a real pointer click and ensure the
  // setting is restored after the regression check.
  const deepInspectorCheck = window.getByText("AI生出力（raw.mp4）", {
    exact: true,
  });
  await deepInspectorCheck.scrollIntoViewIfNeeded();
  await deepInspectorCheck.click();
  const rootScrollAfterDeepClick = await window.evaluate(() => window.scrollY);
  await deepInspectorCheck.click();
  if (rootScrollAfterDeepClick !== 0) {
    throw new Error(
      `Inspector focus scrolled the document root to ${rootScrollAfterDeepClick}px`,
    );
  }

  await dryRun.click();
  await window.getByText("検証済み", { exact: true }).last().waitFor({
    timeout: 30_000,
  });
  await window.getByText("exit").last().waitFor();
  const body = await window.locator("body").innerText();
  if (!body.includes("--face-backend") || !body.includes("pytorch")) {
    throw new Error("Dry Run did not forward the selected face backend");
  }
  if (!body.includes("exit") || !body.includes("0")) {
    throw new Error("Dry Run did not finish successfully");
  }
  const japaneseLog = window
    .locator(".line > code")
    .filter({ hasText: "既存の出力を保護" })
    .first();
  await japaneseLog.waitFor({ state: "visible" });
  const consoleFont = await japaneseLog.evaluate(
    (element) => getComputedStyle(element).fontFamily,
  );
  if (!consoleFont.includes("Noto Sans JP")) {
    throw new Error(`Console lost its Japanese font fallback: ${consoleFont}`);
  }
  const allocatedOutput = `${occupiedOutput}_2`;
  if (!body.includes(allocatedOutput)) {
    throw new Error(
      `existing output was not versioned: expected ${allocatedOutput}`,
    );
  }
  await window.screenshot({
    path: path.join(artifactRoot, "02-dry-run.png"),
    fullPage: true,
  });

  // Seed the completed state that a successful real run writes, then verify
  // the dedicated output queue and its Explorer click path. Automation mode
  // acknowledges the IPC without launching an external Windows process.
  await window.evaluate(
    ({ completedOutput }) => {
      const key = "mask-studio-queue";
      const queue = JSON.parse(window.localStorage.getItem(key) ?? "[]");
      queue[0] = {
        ...queue[0],
        status: "done",
        outputDir: completedOutput,
        summary: "v3-lite + Face V2 · ポリゴン · overlay fast",
        completedAt: new Date().toISOString(),
        artifactCount: 4,
        error: null,
      };
      window.localStorage.setItem(key, JSON.stringify(queue));
    },
    { completedOutput: occupiedOutput },
  );
  await window.reload();
  await window.waitForLoadState("domcontentloaded");
  const outputEntry = window.locator(".output-queue .qitem--output");
  await outputEntry.waitFor({ state: "visible" });
  await outputEntry.getByText("4成果物", { exact: false }).waitFor();
  if ((await outputEntry.getAttribute("title")) !== null) {
    throw new Error("Output queue still uses the native GTK tooltip");
  }
  await outputEntry.hover();
  const tooltip = window.locator(".app-tooltip");
  await tooltip.waitFor({ state: "visible" });
  await tooltip.getByText("クリックして出力フォルダを開く", {
    exact: false,
  }).waitFor();
  const tooltipFont = await tooltip.evaluate(
    (element) => getComputedStyle(element).fontFamily,
  );
  if (!tooltipFont.includes("Noto Sans JP")) {
    throw new Error(`Tooltip lost its Japanese font: ${tooltipFont}`);
  }
  const completedInputEntry = window.locator(
    ".queue:not(.output-queue) .qitem.is-done",
  );
  if ((await completedInputEntry.count()) !== 1) {
    throw new Error("Completed item did not remain in the input queue");
  }
  await outputEntry.click();
  await outputEntry.click({ button: "right" });
  if ((await window.locator(".ctxmenu").count()) !== 0) {
    throw new Error("Output history exposed the input re-processing menu");
  }
  await window.screenshot({
    path: path.join(artifactRoot, "03-output-queue.png"),
    fullPage: true,
  });
  await completedInputEntry.click({ button: "right" });
  await window
    .getByRole("button", { name: "再処理（未処理に戻す）" })
    .click();
  await window
    .locator(".queue:not(.output-queue) .qitem.is-pending")
    .waitFor({ state: "visible" });
  if ((await window.locator(".output-queue .qitem").count()) !== 1) {
    throw new Error("Re-queueing discarded the previous output history");
  }
  const secondOutput = `${occupiedOutput}_2`;
  await window.evaluate(
    ({ nextOutput }) => {
      const key = "mask-studio-queue";
      const queue = JSON.parse(window.localStorage.getItem(key) ?? "[]");
      queue[0].outputs.push({
        id: "second-run",
        outputDir: nextOutput,
        summary: queue[0].summary,
        completedAt: new Date().toISOString(),
        artifactCount: 4,
      });
      window.localStorage.setItem(key, JSON.stringify(queue));
    },
    { nextOutput: secondOutput },
  );
  await window.reload();
  await window.waitForLoadState("domcontentloaded");
  if ((await window.locator(".output-queue .qitem").count()) !== 2) {
    throw new Error("Repeated run did not append a second output history item");
  }
  await window
    .locator(".output-queue .qitem")
    .filter({ hasText: `${videoTitle}_2` })
    .waitFor({ state: "visible" });

  const report = {
    status: "passed",
    title: await window.title(),
    video,
    outputRoot: collisionRoot,
    occupiedOutput,
    allocatedOutput,
    queueItems: await window.locator(".qitem").count(),
    outputQueueItemsBeforeRequeue: 1,
    outputQueueItemsAfterSecondRun: 2,
    explorerClick: "passed",
    contextMenuRequeue: "passed",
    outputContextMenu: "disabled",
    faceBackend: "pytorch",
    progressPanels: await window.locator(".phase-progress").count(),
    dryRunExitCode: 0,
    rendererErrors: errors,
  };
  fs.writeFileSync(
    path.join(artifactRoot, "report.json"),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  if (errors.length > 0) {
    throw new Error(`renderer errors: ${errors.join(" | ")}`);
  }
  console.log(JSON.stringify(report, null, 2));
} finally {
  await app?.close();
  fs.rmSync(userData, { recursive: true, force: true });
  fs.rmSync(collisionRoot, { recursive: true, force: true });
}
