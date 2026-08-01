// Screenshot helper for design proposal mockups.
// Usage: cd /home/kenshin/inference_backend2/gui && node design-proposals/_shared/screenshot.mjs <abs-html-path> <out-prefix>
// Produces <out-prefix>-1920.png and <out-prefix>-1366.png and warns on page overflow.
import { chromium } from 'playwright';
import { pathToFileURL } from 'node:url';

const [, , htmlPath, outPrefix] = process.argv;
if (!htmlPath || !outPrefix) {
  console.error('usage: node screenshot.mjs <abs-html-path> <out-prefix>');
  process.exit(1);
}

const browser = await chromium.launch();
try {
  for (const [width, height] of [[1920, 1080], [1366, 900]]) {
    const page = await browser.newPage({ viewport: { width, height } });
    page.on('console', (msg) => {
      if (msg.type() === 'error') console.log(`CONSOLE-ERROR ${width}x${height}: ${msg.text()}`);
    });
    page.on('pageerror', (err) => console.log(`PAGE-ERROR ${width}x${height}: ${err.message}`));
    await page.goto(pathToFileURL(htmlPath).href);
    await page.waitForTimeout(400);
    const m = await page.evaluate(() => ({
      sw: document.scrollingElement.scrollWidth,
      iw: window.innerWidth,
      sh: document.scrollingElement.scrollHeight,
      ih: window.innerHeight,
    }));
    if (m.sw > m.iw) console.log(`WARN ${width}x${height}: BODY HORIZONTAL OVERFLOW scrollWidth=${m.sw} > innerWidth=${m.iw}`);
    if (m.sh > m.ih) console.log(`INFO ${width}x${height}: body vertical overflow scrollHeight=${m.sh} > innerHeight=${m.ih} (app should be 100vh; expected only below min sizes)`);
    await page.screenshot({ path: `${outPrefix}-${width}.png` });
    await page.close();
    console.log(`ok ${outPrefix}-${width}.png`);
  }
} finally {
  await browser.close();
}
