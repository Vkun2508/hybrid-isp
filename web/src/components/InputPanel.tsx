import { useRef, useState } from "react";
import { ACCEPT, MAX_UPLOAD_MB, type Sample } from "../api";

interface Props {
  samples: Sample[] | null;
  selectedId: string | null;
  fileName: string | null;
  onSample: (s: Sample) => void;
  onFile: (f: File) => void;
}

export function InputPanel({ samples, selectedId, fileName, onSample, onFile }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <section className="card o-input" aria-labelledby="input-h">
      <div className="card-head">
        <h2 id="input-h">Input</h2>
        <p>Real phone RAW or a photo</p>
      </div>
      <ul className="samples">
        {samples === null
          ? Array.from({ length: 5 }, (_, i) => (
              <li key={i}>
                <div className="sample" aria-hidden="true">
                  <div className="thumb-skel" />
                  <div>
                    <div className="skel" style={{ height: 14, width: "80%" }} />
                    <div className="skel" style={{ height: 12, width: "55%", marginTop: 6 }} />
                  </div>
                </div>
              </li>
            ))
          : samples.map((s) => (
              <li key={s.id}>
                <button className="sample" aria-pressed={selectedId === s.id} onClick={() => onSample(s)}>
                  <img src={s.thumb_url} alt="" width={52} height={52} loading="lazy" />
                  <span>
                    <span className="sample-name">{s.name}</span>
                    <span className="sample-meta">
                      {s.kind === "sidd" ? `Real RAW · unseen test scene` : "Photo → simulated RAW"}
                    </span>
                  </span>
                </button>
              </li>
            ))}
      </ul>
      <button
        className={"drop" + (over ? " over" : "")}
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
        <strong>{fileName ? `Using ${fileName}` : "Drop your own photo"}</strong>
        <span>{fileName ? "Drop another, or browse" : "or browse your files"}</span>
        <span className="small">DNG, NEF, CR2, ARW and other RAW, or JPG / PNG · up to {MAX_UPLOAD_MB} MB</span>
      </button>
      <input
        ref={fileRef}
        type="file"
        accept={ACCEPT}
        className="sr-only"
        tabIndex={-1}
        aria-label="Upload a RAW file or photo"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </section>
  );
}
