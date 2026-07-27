import type { MaskStudioApi } from "../shared/types";

declare global {
  interface Window {
    maskStudio?: MaskStudioApi;
  }
}

export {};

