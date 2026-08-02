import process from "node:process";
import { chromium } from "playwright";

const endpoint = process.argv.find((value) => value.startsWith("--endpoint="))?.slice(11);
const screenshot = process.argv.find((value) => value.startsWith("--screenshot="))?.slice(13);
if (!endpoint) throw new Error("--endpoint is required");
const browser = await chromium.connectOverCDP(endpoint);
try {
  const page = browser.contexts()[0]?.pages()[0];
  if (!page) throw new Error("Electron renderer was not found");
  const snapshot = await page.evaluate(async () => {
    const bootstrap = await window.maskStudio.bootstrap();
    return {
      job: bootstrap.job,
      queue: JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]").map((item) => ({
        title: item.title,
        status: item.status,
        outputDir: item.outputDir,
        error: item.error,
      })),
      appVisible: Boolean(document.querySelector(".app")),
      bodyTextLength: document.body.innerText.length,
      rootScrollY: window.scrollY,
      topbarButtons: Array.from(document.querySelectorAll(".topbar button")).map((button) => ({
        text: button.textContent?.trim() ?? "",
        disabled: button.disabled,
      })),
      inspectorDisabledControls: document.querySelectorAll(
        ".panel--inspector button:disabled,.panel--inspector input:disabled",
      ).length,
    };
  });
  console.log(JSON.stringify(snapshot, null, 2));
  if (screenshot) await page.screenshot({ path: screenshot, fullPage: true });
} finally {
  await browser.close();
}
