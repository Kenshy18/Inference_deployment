import { useCallback, useEffect, useRef, useState } from "react";

interface SplitOptions {
  min: number;
  max: number;
  /** "start" grows with the pointer, "end" grows against it. */
  side: "start" | "end";
  axis: "x" | "y";
}

/** Draggable panel divider with a persisted size, NLE-style. */
export function useSplit(
  key: string,
  initial: number,
  { min, max, side, axis }: SplitOptions,
) {
  const [size, setSize] = useState(() => {
    const stored = Number(window.localStorage.getItem(key));
    return Number.isFinite(stored) && stored >= min && stored <= max
      ? stored
      : initial;
  });
  const [dragging, setDragging] = useState(false);
  const origin = useRef({ pointer: 0, size: 0 });

  useEffect(() => {
    window.localStorage.setItem(key, String(size));
  }, [key, size]);

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId);
      origin.current = {
        pointer: axis === "x" ? event.clientX : event.clientY,
        size,
      };
      setDragging(true);
    },
    [axis, size],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) {
        return;
      }
      const pointer = axis === "x" ? event.clientX : event.clientY;
      const delta = pointer - origin.current.pointer;
      const next = origin.current.size + (side === "start" ? delta : -delta);
      setSize(Math.min(max, Math.max(min, Math.round(next))));
    },
    [axis, dragging, max, min, side],
  );

  const onPointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.currentTarget.releasePointerCapture(event.pointerId);
      setDragging(false);
    },
    [],
  );

  const handleProps = {
    className: `split split--${axis === "x" ? "v" : "h"} ${
      dragging ? "is-dragging" : ""
    }`,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    role: "separator" as const,
    "aria-orientation": (axis === "x" ? "vertical" : "horizontal") as
      | "vertical"
      | "horizontal",
  };

  return { size, setSize, handleProps };
}
