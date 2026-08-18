import fs from "node:fs";
import path from "node:path";
import type { AppSettings } from "../shared/types";

export interface DeploymentProfile {
  schemaVersion: 1;
  releaseId: string;
  userDataPath: string;
  backendRoot: string;
  runtimePython: string;
  wslDistro: string;
  sourcePath: string;
}

const SAFE_ID = /^[A-Za-z0-9_.-]+$/;
const PROFILE_FILENAME = "deployment-profile.json";

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

function candidatePaths(
  argv: string[],
  environment: NodeJS.ProcessEnv,
  executablePath: string,
): string[] {
  const candidates: string[] = [];
  const explicit =
    values(argv, "deployment-profile").at(-1) ??
    environment.MASK_STUDIO_DEPLOYMENT_PROFILE;
  if (explicit) {
    candidates.push(path.resolve(explicit));
  }
  for (const directory of [
    environment.PORTABLE_EXECUTABLE_DIR,
    environment.PORTABLE_EXECUTABLE_FILE
      ? path.dirname(environment.PORTABLE_EXECUTABLE_FILE)
      : undefined,
    path.dirname(executablePath),
  ]) {
    if (directory) {
      candidates.push(path.join(directory, PROFILE_FILENAME));
    }
  }
  return [...new Set(candidates)];
}

function requiredString(
  value: unknown,
  name: string,
  sourcePath: string,
): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${sourcePath}: ${name} must be a non-empty string`);
  }
  return value.trim();
}

export function loadDeploymentProfile(
  argv: string[],
  environment: NodeJS.ProcessEnv,
  executablePath: string,
): DeploymentProfile | null {
  const explicit = Boolean(
    values(argv, "deployment-profile").at(-1) ??
      environment.MASK_STUDIO_DEPLOYMENT_PROFILE,
  );
  const sourcePath = candidatePaths(argv, environment, executablePath).find(
    (candidate) => fs.existsSync(candidate),
  );
  if (!sourcePath) {
    if (explicit) {
      throw new Error("The requested deployment profile does not exist");
    }
    return null;
  }

  let value: Record<string, unknown>;
  try {
    value = JSON.parse(fs.readFileSync(sourcePath, "utf8")) as Record<
      string,
      unknown
    >;
  } catch (error) {
    throw new Error(
      `${sourcePath}: invalid deployment profile: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  if (value.schema_version !== 1) {
    throw new Error(`${sourcePath}: unsupported deployment profile schema`);
  }
  const releaseId = requiredString(value.release_id, "release_id", sourcePath);
  const wslDistro = requiredString(
    value.wsl_distribution,
    "wsl_distribution",
    sourcePath,
  );
  if (!SAFE_ID.test(releaseId) || !SAFE_ID.test(wslDistro)) {
    throw new Error(`${sourcePath}: unsafe release or distribution identifier`);
  }
  const userDataPath = requiredString(
    value.user_data_path,
    "user_data_path",
    sourcePath,
  );
  if (!path.isAbsolute(userDataPath)) {
    throw new Error(`${sourcePath}: user_data_path must be absolute`);
  }
  return {
    schemaVersion: 1,
    releaseId,
    userDataPath: path.resolve(userDataPath),
    backendRoot: requiredString(
      value.backend_root,
      "backend_root",
      sourcePath,
    ),
    runtimePython: requiredString(
      value.runtime_python,
      "runtime_python",
      sourcePath,
    ),
    wslDistro,
    sourcePath,
  };
}

export function bindDeploymentSettings(
  settings: AppSettings,
  profile: DeploymentProfile | null,
): AppSettings {
  if (!profile) {
    return settings;
  }
  return {
    ...settings,
    backendMode: "wsl",
    backendRoot: profile.backendRoot,
    runtimePython: profile.runtimePython,
    wslDistro: profile.wslDistro,
  };
}
