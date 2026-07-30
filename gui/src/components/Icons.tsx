import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 16 16"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {children}
    </svg>
  );
}

export const PlayIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 3.2 12.4 8 5 12.8V3.2Z" fill="currentColor" stroke="none" />
  </Icon>
);

export const StopIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="4" y="4" width="8" height="8" rx="0.5" fill="currentColor" stroke="none" />
  </Icon>
);

export const CheckIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m3.2 8.4 3 3L12.8 4.6" />
  </Icon>
);

export const ChevronIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m6 3.5 4.5 4.5L6 12.5" />
  </Icon>
);

export const FolderIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M1.8 4h4l1.4 1.6h7V12a1 1 0 0 1-1 1H2.8a1 1 0 0 1-1-1V4Z" />
  </Icon>
);

export const VideoIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="1.6" y="3.6" width="9" height="8.8" rx="1" />
    <path d="m10.6 7.4 3.8-2.1v5.4l-3.8-2.1" />
  </Icon>
);

export const CpuIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="4.2" y="4.2" width="7.6" height="7.6" rx="1" />
    <path d="M6.6 1.8v2.4M9.4 1.8v2.4M6.6 11.8v2.4M9.4 11.8v2.4M1.8 6.6h2.4M1.8 9.4h2.4M11.8 6.6h2.4M11.8 9.4h2.4" />
  </Icon>
);

export const LayersIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 1.8 14.4 5 8 8.2 1.6 5 8 1.8Z" />
    <path d="m1.6 8 6.4 3.2L14.4 8" />
    <path d="m1.6 11 6.4 3.2L14.4 11" />
  </Icon>
);

export const EyeIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M1.4 8S3.8 3.8 8 3.8 14.6 8 14.6 8 12.2 12.2 8 12.2 1.4 8 1.4 8Z" />
    <circle cx="8" cy="8" r="1.8" />
  </Icon>
);

export const DatabaseIcon = (p: IconProps) => (
  <Icon {...p}>
    <ellipse cx="8" cy="3.6" rx="5.4" ry="2" />
    <path d="M2.6 3.6v8.8c0 1.1 2.4 2 5.4 2s5.4-.9 5.4-2V3.6" />
    <path d="M2.6 8c0 1.1 2.4 2 5.4 2s5.4-.9 5.4-2" />
  </Icon>
);

export const SettingsIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="2" />
    <path d="M8 1.6v1.8M8 12.6v1.8M14.4 8h-1.8M3.4 8H1.6M12.5 3.5l-1.3 1.3M4.8 11.2l-1.3 1.3M12.5 12.5l-1.3-1.3M4.8 4.8 3.5 3.5" />
  </Icon>
);

export const TerminalIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="m2.6 4.4 3.2 3.2-3.2 3.2M8.4 11.4h5" />
  </Icon>
);

export const FilmIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="1.6" y="2.6" width="12.8" height="10.8" rx="1" />
    <path d="M4.8 2.6v10.8M11.2 2.6v10.8M1.6 8h12.8" />
  </Icon>
);

export const PlusIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 3v10M3 8h10" />
  </Icon>
);

export const AlertIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 2.4 14.6 13H1.4L8 2.4Z" />
    <path d="M8 6.6v3M8 11.4h.01" />
  </Icon>
);
