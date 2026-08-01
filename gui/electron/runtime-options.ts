import path from "node:path";

export interface RuntimeOptions {
  automationPort: number | null;
  automationAddress: "127.0.0.1" | "0.0.0.0";
  automationVideos: string[];
  automationOutput: string | null;
  softwareRendering: boolean;
  qaE2eInput: string | null;
  qaE2eOutput: string | null;
  qaE2eReport: string | null;
  qaE2eMaxFrames: number;
  qaE2eCancelAfterMs: number | null;
}

function values(argv: string[], name: string): string[] {
  const prefix = `--${name}=`;
  const output: string[] = [];
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument.startsWith(prefix)) {
      output.push(argument.slice(prefix.length));
    } else if (argument === `--${name}` && argv[index + 1]) {
      output.push(argv[index + 1]);
      index += 1;
    }
  }
  return output.filter(Boolean);
}

function hasFlag(argv: string[], name: string): boolean {
  return argv.includes(`--${name}`);
}

function parsePort(value: string | undefined): number | null {
  if (value === undefined) {
    return null;
  }
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error(`--automation-port must be an integer in [1, 65535]`);
  }
  return port;
}

function enabled(value: string | undefined): boolean {
  return value === "1" || value?.toLowerCase() === "true";
}

export function parseRuntimeOptions(
  argv: string[],
  environment: NodeJS.ProcessEnv,
): RuntimeOptions {
  const automationPort = parsePort(
    values(argv, "automation-port").at(-1) ??
      environment.MASK_STUDIO_AUTOMATION_PORT,
  );
  const requestedAddress =
    values(argv, "automation-address").at(-1) ??
    environment.MASK_STUDIO_AUTOMATION_ADDRESS ??
    "127.0.0.1";
  if (requestedAddress !== "127.0.0.1" && requestedAddress !== "0.0.0.0") {
    throw new Error(
      "--automation-address must be 127.0.0.1 or 0.0.0.0",
    );
  }
  if (
    requestedAddress === "0.0.0.0" &&
    !enabled(environment.MASK_STUDIO_ALLOW_REMOTE_AUTOMATION)
  ) {
    throw new Error(
      "--automation-address=0.0.0.0 requires MASK_STUDIO_ALLOW_REMOTE_AUTOMATION=1",
    );
  }
  const automationVideos = values(argv, "automation-video").map((value) =>
    path.resolve(value),
  );
  const automationOutputValue =
    values(argv, "automation-output").at(-1) ??
    environment.MASK_STUDIO_AUTOMATION_OUTPUT;
  const isWsl = Boolean(environment.WSL_DISTRO_NAME || environment.WSL_INTEROP);
  const hardwareRequested = hasFlag(argv, "hardware-rendering");
  const softwareRendering =
    !hardwareRequested &&
    (hasFlag(argv, "software-rendering") ||
      enabled(environment.MASK_STUDIO_SOFTWARE_RENDERING) ||
      isWsl);
  const qaE2eInput = values(argv, "qa-e2e-input").at(-1) ?? null;
  const qaE2eOutput = values(argv, "qa-e2e-output").at(-1) ?? null;
  const qaE2eReport = values(argv, "qa-e2e-report").at(-1) ?? null;
  const qaFramesValue = values(argv, "qa-e2e-max-frames").at(-1) ?? "120";
  const qaE2eMaxFrames = Number.parseInt(qaFramesValue, 10);
  if (!Number.isInteger(qaE2eMaxFrames) || qaE2eMaxFrames < 1) {
    throw new Error("--qa-e2e-max-frames must be a positive integer");
  }
  const qaCancelValue = values(argv, "qa-e2e-cancel-after-ms").at(-1);
  const qaE2eCancelAfterMs =
    qaCancelValue === undefined ? null : Number.parseInt(qaCancelValue, 10);
  if (
    qaE2eCancelAfterMs !== null &&
    (!Number.isInteger(qaE2eCancelAfterMs) || qaE2eCancelAfterMs < 100)
  ) {
    throw new Error("--qa-e2e-cancel-after-ms must be an integer >= 100");
  }
  return {
    automationPort,
    automationAddress: requestedAddress,
    automationVideos,
    automationOutput: automationOutputValue
      ? path.resolve(automationOutputValue)
      : null,
    softwareRendering,
    qaE2eInput: qaE2eInput ? path.resolve(qaE2eInput) : null,
    qaE2eOutput: qaE2eOutput ? path.resolve(qaE2eOutput) : null,
    qaE2eReport: qaE2eReport ? path.resolve(qaE2eReport) : null,
    qaE2eMaxFrames,
    qaE2eCancelAfterMs,
  };
}
