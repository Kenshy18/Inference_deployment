import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/noto-sans-jp/japanese-400.css";
import "@fontsource/noto-sans-jp/japanese-500.css";
import "@fontsource/noto-sans-jp/japanese-600.css";
import "@fontsource/noto-sans-jp/latin-400.css";
import "@fontsource/noto-sans-jp/latin-500.css";
import "@fontsource/noto-sans-jp/latin-600.css";
import App from "./App";
import { TooltipLayer } from "./components/TooltipLayer";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
    <TooltipLayer />
  </StrictMode>,
);
