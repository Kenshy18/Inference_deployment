import path from "node:path";

export interface RuntimeOptions {
  automationPort: number | null;
  automationVideos: string[];
  automationOutput: string | null;
  softwareRendering: boolean;
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
  return {
    automationPort,
    automationVideos,
    automationOutput: automationOutputValue
      ? path.resolve(automationOutputValue)
      : null,
    softwareRendering,
  };
}
