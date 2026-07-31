import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface TooltipState {
  text: string;
  x: number;
  y: number;
  above: boolean;
}

const TOOLTIP_ATTRIBUTE = "data-app-tooltip";

function promoteNativeTitle(element: Element): void {
  const title = element.getAttribute("title");
  if (!title) {
    return;
  }
  if (element.closest(".panel--inspector") !== null) {
    // Inspector labels and controls already explain themselves inline. Keep
    // the native popup disabled without replacing it with another popup.
    element.removeAttribute(TOOLTIP_ATTRIBUTE);
    element.removeAttribute("title");
    return;
  }
  element.setAttribute(TOOLTIP_ATTRIBUTE, title);
  element.removeAttribute("title");
}

function promoteTitles(root: ParentNode): void {
  if (root instanceof Element && root.hasAttribute("title")) {
    promoteNativeTitle(root);
  }
  root.querySelectorAll?.("[title]").forEach(promoteNativeTitle);
}

/**
 * Replaces Chromium/GTK native title popups with an in-app tooltip. Native
 * popups cannot use the bundled Noto Sans JP font and render Japanese as tofu
 * on a minimal WSLg installation.
 */
export function TooltipLayer() {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const anchor = useRef<Element | null>(null);
  const pointer = useRef({ x: 0, y: 0 });
  const timer = useRef<number | null>(null);

  useEffect(() => {
    promoteTitles(document.documentElement);
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        if (record.type === "attributes") {
          promoteNativeTitle(record.target as Element);
          continue;
        }
        record.addedNodes.forEach((node) => {
          if (node instanceof Element) {
            promoteTitles(node);
          }
        });
      }
    });
    observer.observe(document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["title"],
    });

    const clearTimer = () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current);
        timer.current = null;
      }
    };
    const hide = () => {
      clearTimer();
      anchor.current = null;
      setTooltip(null);
    };
    const show = (element: Element) => {
      clearTimer();
      anchor.current = element;
      timer.current = window.setTimeout(() => {
        const text = element.getAttribute(TOOLTIP_ATTRIBUTE);
        if (!text || anchor.current !== element) {
          return;
        }
        const { x, y } = pointer.current;
        setTooltip({
          text,
          x: Math.max(8, Math.min(x + 14, window.innerWidth - 388)),
          y: y > window.innerHeight * 0.7 ? y - 14 : y + 18,
          above: y > window.innerHeight * 0.7,
        });
      }, 280);
    };
    const onMouseOver = (event: MouseEvent) => {
      if (!(event.target instanceof Element)) {
        return;
      }
      const element = event.target.closest(`[${TOOLTIP_ATTRIBUTE}]`);
      if (element === anchor.current) {
        return;
      }
      if (element === null) {
        hide();
        return;
      }
      pointer.current = { x: event.clientX, y: event.clientY };
      show(element);
    };
    const onMouseMove = (event: MouseEvent) => {
      pointer.current = { x: event.clientX, y: event.clientY };
    };
    const onMouseOut = (event: MouseEvent) => {
      if (
        anchor.current !== null &&
        event.relatedTarget instanceof Node &&
        anchor.current.contains(event.relatedTarget)
      ) {
        return;
      }
      hide();
    };
    const onViewportChange = () => hide();

    document.addEventListener("mouseover", onMouseOver);
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseout", onMouseOut);
    window.addEventListener("blur", hide);
    window.addEventListener("scroll", onViewportChange, true);
    return () => {
      observer.disconnect();
      clearTimer();
      document.removeEventListener("mouseover", onMouseOver);
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseout", onMouseOut);
      window.removeEventListener("blur", hide);
      window.removeEventListener("scroll", onViewportChange, true);
    };
  }, []);

  if (tooltip === null) {
    return null;
  }
  return createPortal(
    <div
      className={`app-tooltip ${tooltip.above ? "is-above" : ""}`}
      role="tooltip"
      style={{ left: tooltip.x, top: tooltip.y }}
    >
      {tooltip.text}
    </div>,
    document.body,
  );
}
