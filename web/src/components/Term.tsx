import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { GLOSSARY, type TermKey } from "../glossary";

/** A piece of jargon with a plain-language definition: tap on a phone, hover or focus on a laptop. */
export function Term({ k, children, align }: { k: TermKey; children?: ReactNode; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const [hover, setHover] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const id = useId();
  const e = GLOSSARY[k];
  const shown = open || hover;

  useEffect(() => {
    if (!open) return;
    const away = (ev: PointerEvent) => { if (!ref.current?.contains(ev.target as Node)) setOpen(false); };
    const esc = (ev: KeyboardEvent) => { if (ev.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", esc); };
  }, [open]);

  return (
    <span
      className="term" ref={ref}
      onPointerEnter={(ev) => { if (ev.pointerType === "mouse") setHover(true); }}
      onPointerLeave={(ev) => { if (ev.pointerType === "mouse") setHover(false); }}
    >
      <button type="button" aria-expanded={shown} aria-describedby={shown ? id : undefined} onClick={() => setOpen((o) => !o)}>
        {children ?? e.name}
      </button>
      {shown && (
        <span className={"term-pop" + (align ? ` ${align}` : "")} role="tooltip" id={id}>
          <b>{e.name}</b>
          {e.text}
          {"better" in e && e.better && <span className="better">{e.better}</span>}
        </span>
      )}
    </span>
  );
}
