import fs from "node:fs";

function hasContent(target: string): boolean {
  if (!fs.existsSync(target)) {
    return false;
  }
  try {
    return !fs.statSync(target).isDirectory() || fs.readdirSync(target).length > 0;
  } catch {
    // An unreadable path is not safe to reuse.
    return true;
  }
}

/** Select a non-destructive job root. Explicit resume keeps the requested
 * directory; a new run gets `_2`, `_3`, ... when a prior result or partial
 * output already occupies the requested path. */
export function availableOutputRoot(
  requested: string,
  resume: boolean,
): string {
  if (resume || !hasContent(requested)) {
    return requested;
  }
  for (let index = 2; ; index += 1) {
    const candidate = `${requested}_${index}`;
    if (!hasContent(candidate)) {
      return candidate;
    }
  }
}
