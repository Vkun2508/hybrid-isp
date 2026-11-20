import { useRef, useState } from "react";
import { ACCEPT, MAX_UPLOAD_MB, type Sample } from "../api";

interface Props {
  samples: Sample[] | null;
  selectedId: string | null;
  fileName: string | null;
  onSample: (s: Sample) => void;
  onFile: (f: File) => void;
}

const kindLabel = (s: Sample) => (s.kind === "sidd" ? "Real phone RAW" : "Photo → simulated RAW");

/** Step 1: a filmstrip of samples, with your own photo as the last tile. */
export function InputPanel({ samples, selectedId, fileName, onSample, onFile }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div className="filmstrip-wrap">
      <ul className="filmstrip" aria-label="Sample photos">
        {samples === null
          ? Array.from({ length: 4 }, (_, i) => (
              <li key={i} aria-hidden="true">
                <div className="film"><div className="film-img skel" /><div className="skel" style={{ height: 12, width: "80%", marginTop: 8 }} /></div>
              </li>
            ))
          : samples.map((s) => (
              <li key={s.id}>
                <button className="film" aria-pressed={selectedId === s.id} onClick={() => onSample(s)}>
                  <span className="film-img"><img src={s.thumb_url} alt="" loading="lazy" /></span>
                  <span className="film-name">{s.name.split(" · ").slice(0, 1).join("")}</span>
                  <span className="film-meta">{s.name.split(" · ").slice(1).join(" · ") || kindLabel(s)}</span>
                  <span className={"film-kind" + (s.kind === "sidd" ? " raw" : "")}>{kindLabel(s)}</span>
                </button>
              </li>
            ))}
        <li>
          <button
            className={"film drop" + (over ? " over" : "") + (fileName ? " has-file" : "")}
            aria-pressed={!!fileName}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              const f = e.dataTransfer.files?.[0];
              if (f) onFile(f);
            }}
          >
            <span className="film-img drop-img" aria-hidden="true">
              <svg width="28" height="28" viewBox="0 0 28 28"><path d="M14 19V6M8.5 11.5L14 6l5.5 5.5M5 18v3.5A1.5 1.5 0 0 0 6.5 23h15a1.5 1.5 0 0 0 1.5-1.5V18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </span>
            <span className="film-name">{fileName ? "Your photo" : "Your own photo"}</span>
            <span className="film-meta" title={fileName ?? undefined}>{fileName ?? "RAW (DNG, NEF, CR2…) or JPG"}</span>
            <span className="film-kind">{fileName ? "Tap to change" : `Up to ${MAX_UPLOAD_MB} MB`}</span>
          </button>
        </li>
      </ul>
      <input
        ref={fileRef} type="file" accept={ACCEPT} className="sr-only" tabIndex={-1} aria-label="Upload a RAW file or photo"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}
